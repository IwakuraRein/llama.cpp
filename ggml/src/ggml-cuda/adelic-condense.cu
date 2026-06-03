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

    size_t k_idx = (size_t)head_idx * seq_len * head_dim + (size_t)token_idx * head_dim + dim_idx;

    __shared__ float s_dot;
    __shared__ float s_norm_k;
    __shared__ float s_norm_r;

    if (dim_idx == 0) {
        s_dot = 0.0f;
        s_norm_k = 0.0f;
        s_norm_r = 0.0f;
    // Mathematical correction: The 'router' passed here is actually a [n_embd, n_embd] weight matrix,
    // not a centroid vector. Treating it as a vector and taking the dot product with the KV cache 
    // results in random noise, which arbitrarily zeroed out the KV cache and caused gibberish generation.
    // Full DynamicTopologyRouter port to GGML requires significant graph surgery. 
    // For now, this kernel acts as a no-op (dense attention fallback) so the model generates coherently.
    return;
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
