#include "common.cuh"

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "ggml-cuda.h"
#undef GGML_COMMON_DECL_C

void ggml_cuda_op_adelic_condense(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
