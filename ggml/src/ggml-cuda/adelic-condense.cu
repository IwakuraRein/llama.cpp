#include "adelic-condense.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void adelic_condense_kernel_f16(
    half * k_cache,
    half * v_cache,
    int head_dim,
    int seq_len,
    float threshold) 
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stream_idx = blockIdx.y;

    if (token_idx <= 17 || token_idx >= seq_len) return;

    size_t curr_base = (size_t)stream_idx * seq_len * head_dim + (size_t)token_idx * head_dim;
    size_t prev_base = (size_t)stream_idx * seq_len * head_dim + (size_t)(token_idx - 1) * head_dim;

    float s_dot = 0.0f;
    float s_norm_curr = 0.0f;
    float s_norm_prev = 0.0f;

    for (int i = 0; i < head_dim; ++i) {
        float v_curr = __half2float(v_cache[curr_base + i]);
        float v_prev = __half2float(v_cache[prev_base + i]);

        s_dot += v_curr * v_prev;
        s_norm_curr += v_curr * v_curr;
        s_norm_prev += v_prev * v_prev;
    }

    if (s_norm_curr > 0.0f && s_norm_prev > 0.0f) {
        float cos_sim = s_dot / (sqrtf(s_norm_curr) * sqrtf(s_norm_prev) + 1e-6f);
        if (cos_sim > threshold) {
            for (int i = 0; i < head_dim; ++i) {
                k_cache[curr_base + i] = __float2half(-1e4f);
            }
        }
    }
}

__global__ void adelic_condense_kernel_f32(
    float * k_cache,
    float * v_cache,
    int head_dim,
    int seq_len,
    float threshold) 
{
    int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stream_idx = blockIdx.y;

    if (token_idx <= 17 || token_idx >= seq_len) return;

    size_t curr_base = (size_t)stream_idx * seq_len * head_dim + (size_t)token_idx * head_dim;
    size_t prev_base = (size_t)stream_idx * seq_len * head_dim + (size_t)(token_idx - 1) * head_dim;

    float s_dot = 0.0f;
    float s_norm_curr = 0.0f;
    float s_norm_prev = 0.0f;

    for (int i = 0; i < head_dim; ++i) {
        float v_curr = v_cache[curr_base + i];
        float v_prev = v_cache[prev_base + i];

        s_dot += v_curr * v_prev;
        s_norm_curr += v_curr * v_curr;
        s_norm_prev += v_prev * v_prev;
    }

    if (s_norm_curr > 0.0f && s_norm_prev > 0.0f) {
        float cos_sim = s_dot / (sqrtf(s_norm_curr) * sqrtf(s_norm_prev) + 1e-6f);
        if (cos_sim > threshold) {
            for (int i = 0; i < head_dim; ++i) {
                k_cache[curr_base + i] = -1e4f;
            }
        }
    }
}

void ggml_cuda_op_adelic_condense(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * k = dst->src[0];
    ggml_tensor * v = dst->src[1];

    const int head_dim = k->ne[0];
    const int seq_len = k->ne[1];
    const int n_stream = k->ne[2];
    
    // Configurable threshold: drop tokens with > 90% cosine similarity
    const float threshold = 0.90f; 

    dim3 block(256);
    dim3 grid((seq_len + block.x - 1) / block.x, n_stream);

    if (k->type == GGML_TYPE_F16) {
        adelic_condense_kernel_f16<<<grid, block, 0, ctx.stream()>>>(
            (half *)k->data, (half *)v->data, head_dim, seq_len, threshold
        );
    } else if (k->type == GGML_TYPE_F32) {
        adelic_condense_kernel_f32<<<grid, block, 0, ctx.stream()>>>(
            (float *)k->data, (float *)v->data, head_dim, seq_len, threshold
        );
    }
}
