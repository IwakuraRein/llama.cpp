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

    __shared__ float s_dot;
    __shared__ float s_norm_k;
    __shared__ float s_norm_r;

    if (dim_idx == 0) {
        s_dot = 0.0f;
        s_norm_k = 0.0f;
        s_norm_r = 0.0f;
    }
    __syncthreads();

    // Calculate dot products for cosine similarity
    float k_val = k_cache[k_idx];
    float r_val = router[dim_idx]; // assuming router is a vector of size head_dim

    atomicAdd(&s_dot, k_val * r_val);
    atomicAdd(&s_norm_k, k_val * k_val);
    atomicAdd(&s_norm_r, r_val * r_val);

    __syncthreads();

    if (s_norm_k > 0.0f && s_norm_r > 0.0f) {
        float cos_sim = s_dot / (sqrtf(s_norm_k) * sqrtf(s_norm_r) + 1e-6f);
        if (cos_sim < threshold) {
            // Mask out the token so attention ignores it.
            // Setting K to a large negative number ensures softmax(K*Q) -> 0
            k_cache[k_idx] = -1e4f;
            v_cache[k_idx] = 0.0f;
        }
    }
}

void ggml_cuda_op_adelic_condense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    // Boilerplate for dispatch
    ggml_tensor * k = dst->src[0];
    ggml_tensor * v = dst->src[1];
    ggml_tensor * router = dst->src[2];

    const int head_dim = k->ne[0];
    const int num_heads = k->ne[2];
    const int seq_len = k->ne[1];
    const float threshold = 0.5f;

    float * d_k = (float *)k->data;
    float * d_v = (float *)v->data;
    const float * d_router = (const float *)router->data;

    dim3 grid(seq_len, num_heads);
    dim3 block(head_dim);

    adelic_condense_kernel<<<grid, block, 0, ctx.stream()>>>(
        d_k, d_v, d_router, head_dim, num_heads, seq_len, threshold
    );
}
