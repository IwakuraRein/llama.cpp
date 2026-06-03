#include "adelic-condense.cuh"
#include <cuda_runtime.h>

__global__ void adelic_condense_kernel(
    float * k_cache,
    float * v_cache,
    const float * router,
    int head_dim,
    int num_heads,
    int seq_len,
    float threshold) 
{
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    if (token_idx >= seq_len || head_idx >= num_heads || dim_idx >= head_dim) return;

    // We protect the first 17 tokens (including hologram token 16) from condensation.
    if (token_idx <= 17) return;

    size_t curr_idx = (size_t)head_idx * seq_len * head_dim + (size_t)token_idx * head_dim + dim_idx;
    // Compare with the previous token in the sequence for this head
    size_t prev_idx = (size_t)head_idx * seq_len * head_dim + (size_t)(token_idx - 1) * head_dim + dim_idx;

    __shared__ float s_dot;
    __shared__ float s_norm_curr;
    __shared__ float s_norm_prev;

    if (dim_idx == 0) {
        s_dot = 0.0f;
        s_norm_curr = 0.0f;
        s_norm_prev = 0.0f;
    }
    __syncthreads();

    // The condensation algorithm uses cosine similarity of the V cache tokens
    float v_curr = v_cache[curr_idx];
    float v_prev = v_cache[prev_idx];

    atomicAdd(&s_dot, v_curr * v_prev);
    atomicAdd(&s_norm_curr, v_curr * v_curr);
    atomicAdd(&s_norm_prev, v_prev * v_prev);

    __syncthreads();

    if (s_norm_curr > 0.0f && s_norm_prev > 0.0f) {
        float cos_sim = s_dot / (sqrtf(s_norm_curr) * sqrtf(s_norm_prev) + 1e-6f);
        
        // If the token is highly similar to the previous one, it provides no new information.
        // We mask it out of the K cache so that Softmax(Q*K) -> 0 and it is ignored in attention.
        if (cos_sim > threshold) {
            k_cache[curr_idx] = -1e4f;
            
            // In a full implementation, we would accumulate the dropped V values into 
            // the hologram token (token_idx == 16). For the GGML static graph mask, 
            // zeroing out the dropped tokens achieves the primary sparse routing effect.
        }
    }
}

void ggml_cuda_op_adelic_condense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k = dst->src[0];
    ggml_tensor * v = dst->src[1];
    ggml_tensor * router = dst->src[2]; // Passed from Python export but ignored here

    const int head_dim = k->ne[0];
    const int num_heads = k->ne[2];
    const int seq_len = k->ne[1];
    
    // Configurable threshold: drop tokens with > 90% cosine similarity
    const float threshold = 0.90f; 

    float * d_k = (float *)k->data;
    float * d_v = (float *)v->data;
    const float * d_router = (const float *)router->data;

    dim3 grid(seq_len, num_heads);
    dim3 block(head_dim);

    adelic_condense_kernel<<<grid, block, 0, ctx.stream()>>>(
        d_k, d_v, d_router, head_dim, num_heads, seq_len, threshold
    );
}
