#include "adelic-condense.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void adelic_condense_kernel_f16(
    char * k_cache,
    char * v_cache,
    int head_dim,
    int num_heads,
    int seq_len,
    int n_stream,
    int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
    int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
    bool v_trans,
    float threshold) 
{
    int head_idx   = blockIdx.x;
    int token_idx  = blockIdx.y;
    int stream_idx = blockIdx.z;
    int dim_idx    = threadIdx.x;

    if (token_idx <= 17 || token_idx >= seq_len || head_idx >= num_heads || stream_idx >= n_stream || dim_idx >= head_dim) return;

    size_t k_offset = stream_idx * k_nb3 + token_idx * k_nb2 + head_idx * k_nb1 + dim_idx * k_nb0;
    
    size_t v_offset;
    size_t prev_v_offset;
    if (v_trans) {
        v_offset = stream_idx * v_nb3 + dim_idx * v_nb2 + head_idx * v_nb1 + token_idx * v_nb0;
        prev_v_offset = stream_idx * v_nb3 + dim_idx * v_nb2 + head_idx * v_nb1 + (token_idx - 1) * v_nb0;
    } else {
        v_offset = stream_idx * v_nb3 + token_idx * v_nb2 + head_idx * v_nb1 + dim_idx * v_nb0;
        prev_v_offset = stream_idx * v_nb3 + (token_idx - 1) * v_nb2 + head_idx * v_nb1 + dim_idx * v_nb0;
    }

    float v_curr = __half2float(*(half *)(v_cache + v_offset));
    float v_prev = __half2float(*(half *)(v_cache + prev_v_offset));

    __shared__ float s_dot;
    __shared__ float s_norm_curr;
    __shared__ float s_norm_prev;

    if (dim_idx == 0) {
        s_dot = 0.0f;
        s_norm_curr = 0.0f;
        s_norm_prev = 0.0f;
    }
    __syncthreads();

    atomicAdd(&s_dot, v_curr * v_prev);
    atomicAdd(&s_norm_curr, v_curr * v_curr);
    atomicAdd(&s_norm_prev, v_prev * v_prev);

    __syncthreads();

    if (s_norm_curr > 0.0f && s_norm_prev > 0.0f) {
        float cos_sim = s_dot / (sqrtf(s_norm_curr) * sqrtf(s_norm_prev) + 1e-6f);
        if (cos_sim > threshold) {
            *(half *)(k_cache + k_offset) = __float2half(-1e4f);
        }
    }
}

__global__ void adelic_condense_kernel_f32(
    char * k_cache,
    char * v_cache,
    int head_dim,
    int num_heads,
    int seq_len,
    int n_stream,
    int64_t k_nb0, int64_t k_nb1, int64_t k_nb2, int64_t k_nb3,
    int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
    bool v_trans,
    float threshold) 
{
    int head_idx   = blockIdx.x;
    int token_idx  = blockIdx.y;
    int stream_idx = blockIdx.z;
    int dim_idx    = threadIdx.x;

    if (token_idx <= 17 || token_idx >= seq_len || head_idx >= num_heads || stream_idx >= n_stream || dim_idx >= head_dim) return;

    size_t k_offset = stream_idx * k_nb3 + token_idx * k_nb2 + head_idx * k_nb1 + dim_idx * k_nb0;
    
    size_t v_offset;
    size_t prev_v_offset;
    if (v_trans) {
        v_offset = stream_idx * v_nb3 + dim_idx * v_nb2 + head_idx * v_nb1 + token_idx * v_nb0;
        prev_v_offset = stream_idx * v_nb3 + dim_idx * v_nb2 + head_idx * v_nb1 + (token_idx - 1) * v_nb0;
    } else {
        v_offset = stream_idx * v_nb3 + token_idx * v_nb2 + head_idx * v_nb1 + dim_idx * v_nb0;
        prev_v_offset = stream_idx * v_nb3 + (token_idx - 1) * v_nb2 + head_idx * v_nb1 + dim_idx * v_nb0;
    }

    float v_curr = *(float *)(v_cache + v_offset);
    float v_prev = *(float *)(v_cache + prev_v_offset);

    __shared__ float s_dot;
    __shared__ float s_norm_curr;
    __shared__ float s_norm_prev;

    if (dim_idx == 0) {
        s_dot = 0.0f;
        s_norm_curr = 0.0f;
        s_norm_prev = 0.0f;
    }
    __syncthreads();

    atomicAdd(&s_dot, v_curr * v_prev);
    atomicAdd(&s_norm_curr, v_curr * v_curr);
    atomicAdd(&s_norm_prev, v_prev * v_prev);

    __syncthreads();

    if (s_norm_curr > 0.0f && s_norm_prev > 0.0f) {
        float cos_sim = s_dot / (sqrtf(s_norm_curr) * sqrtf(s_norm_prev) + 1e-6f);
        if (cos_sim > threshold) {
            *(float *)(k_cache + k_offset) = -1e4f;
        }
    }
}

void ggml_cuda_op_adelic_condense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k = dst->src[0];
    ggml_tensor * v = dst->src[1];

    const int head_dim  = k->ne[0];
    const int num_heads = k->ne[1];
    const int seq_len   = k->ne[2];
    const int n_stream  = k->ne[3];

    // In llama.cpp, if v->ne[0] != k->ne[0], it means V cache was transposed (v_trans = true)
    bool v_trans = (v->ne[0] != k->ne[0]);

    const float threshold = 0.90f; 

    dim3 block(head_dim);
    dim3 grid(num_heads, seq_len, n_stream);

    if (k->type == GGML_TYPE_F16) {
        adelic_condense_kernel_f16<<<grid, block, 0, ctx.stream()>>>(
            (char *)k->data, (char *)v->data,
            head_dim, num_heads, seq_len, n_stream,
            k->nb[0], k->nb[1], k->nb[2], k->nb[3],
            v->nb[0], v->nb[1], v->nb[2], v->nb[3],
            v_trans, threshold
        );
    } else if (k->type == GGML_TYPE_F32) {
        adelic_condense_kernel_f32<<<grid, block, 0, ctx.stream()>>>(
            (char *)k->data, (char *)v->data,
            head_dim, num_heads, seq_len, n_stream,
            k->nb[0], k->nb[1], k->nb[2], k->nb[3],
            v->nb[0], v->nb[1], v->nb[2], v->nb[3],
            v_trans, threshold
        );
    }
}
