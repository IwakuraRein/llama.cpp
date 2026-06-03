#include "adelic-condense.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void adelic_condense_mask_kernel(
    char * kq_mask,
    char * v_cache,
    int mask_type,
    int v_type,
    int head_dim,
    int n_kv_capacity,
    int n_tokens,
    int n_stream,
    int64_t mask_nb0, int64_t mask_nb1, int64_t mask_nb2, int64_t mask_nb3,
    int64_t v_nb0, int64_t v_nb1, int64_t v_nb2, int64_t v_nb3,
    bool v_trans,
    float threshold) 
{
    int k_idx      = blockIdx.x; 
    int q_idx      = blockIdx.y; 
    int stream_idx = blockIdx.z;
    int dim_idx    = threadIdx.x;
    int head_idx   = 0; // only use head 0 for similarity proxy

    if (k_idx <= 17 || k_idx >= n_kv_capacity || q_idx >= n_tokens || stream_idx >= n_stream || dim_idx >= head_dim) return;

    size_t v_offset;
    size_t prev_v_offset;
    if (v_trans) {
        v_offset      = stream_idx * v_nb3 + dim_idx * v_nb2 + head_idx * v_nb1 + k_idx * v_nb0;
        prev_v_offset = stream_idx * v_nb3 + dim_idx * v_nb2 + head_idx * v_nb1 + (k_idx - 1) * v_nb0;
    } else {
        v_offset      = stream_idx * v_nb3 + k_idx * v_nb2 + head_idx * v_nb1 + dim_idx * v_nb0;
        prev_v_offset = stream_idx * v_nb3 + (k_idx - 1) * v_nb2 + head_idx * v_nb1 + dim_idx * v_nb0;
    }

    float v_curr = 0.0f;
    float v_prev = 0.0f;

    // v_type: 0 for F32, 1 for F16 (GGML_TYPE_F32 = 0, GGML_TYPE_F16 = 1)
    if (v_type == 1) { // F16
        v_curr = __half2float(*(half *)(v_cache + v_offset));
        v_prev = __half2float(*(half *)(v_cache + prev_v_offset));
    } else if (v_type == 0) { // F32
        v_curr = *(float *)(v_cache + v_offset);
        v_prev = *(float *)(v_cache + prev_v_offset);
    }

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
            size_t mask_offset = stream_idx * mask_nb3 + q_idx * mask_nb1 + k_idx * mask_nb0;
            if (mask_type == 1) {
                *(half *)(kq_mask + mask_offset) = __float2half(-65504.0f);
            } else {
                *(float *)(kq_mask + mask_offset) = -65504.0f;
            }
        }
    }
}

void ggml_cuda_op_adelic_condense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * kq_mask = dst->src[0];
    ggml_tensor * v       = dst->src[1];

    // kq_mask: [n_kv, n_tokens, 1, n_stream]
    const int n_kv_capacity = kq_mask->ne[0];
    const int n_tokens      = kq_mask->ne[1];
    const int n_stream      = kq_mask->ne[3];

    // v is directly from mctx->get_v(), so it's not transposed yet.
    // v: [head_dim, num_heads, max_kv, n_stream]
    const int head_dim  = v->ne[0];
    const int num_heads = v->ne[1];
    const bool v_trans  = false;

    const float threshold = 0.90f; 

    dim3 block(head_dim);
    dim3 grid(n_kv_capacity, n_tokens, n_stream);

    adelic_condense_mask_kernel<<<grid, block, 0, ctx.stream()>>>(
        (char *)kq_mask->data, (char *)v->data,
        (int)kq_mask->type, (int)v->type,
        head_dim, n_kv_capacity, n_tokens, n_stream,
        kq_mask->nb[0], kq_mask->nb[1], kq_mask->nb[2], kq_mask->nb[3],
        v->nb[0], v->nb[1], v->nb[2], v->nb[3],
        v_trans, threshold
    );
}
