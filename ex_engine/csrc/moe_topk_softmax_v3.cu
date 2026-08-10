// moe_topk_softmax_v3.cu — Fused softmax+topk for Qwen3.5 MoE routing
//
// VERIFIED on BI-V100 (ivcore10) 2026-08-10:
//   weights sum=1.0, no NaN, no duplicate ids, 881 tokens batch OK
//   Compiler: corex clang/16, --cuda-gpu-arch=ivcore10
//
// 64 experts, topk=8, warp shuffle only, zero shared memory
// Each warp handles one token row: 32 threads × 2 values = 64 experts
//
// Based on: TRT-LLM/vllm topk_softmax_kernels + xllm moe_topk_softmax_kernels.cuh
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    return val;
}
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    return val;
}

static constexpr int NUM_EXPERTS = 64;
static constexpr int VPT = 2;
static constexpr int THREADS_PER_ROW = NUM_EXPERTS / VPT;
static constexpr int WARPS_PER_CTA = 4;
static constexpr int ROWS_PER_CTA = WARPS_PER_CTA;

__global__ void topk_gating_softmax_kernel(
    const float* __restrict__ input,
    float* __restrict__ output_weights,
    int32_t* __restrict__ output_indices,
    int32_t* __restrict__ output_source_rows,
    int num_tokens, int k, bool renormalize
) {
    const int row = blockIdx.x * ROWS_PER_CTA + threadIdx.y;
    if (row >= num_tokens) return;

    const int tid = threadIdx.x;
    const float* row_input = input + row * NUM_EXPERTS;

    float vals[VPT];
    int my_indices[VPT];
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        int col = tid * VPT + i;
        vals[i] = row_input[col];
        my_indices[i] = col;
    }

    float tmax = vals[0];
    for (int i = 1; i < VPT; i++) tmax = fmaxf(tmax, vals[i]);
    float row_max = warp_reduce_max(tmax);

    float tsum = 0.0f;
    #pragma unroll
    for (int i = 0; i < VPT; i++) {
        vals[i] = expf(vals[i] - row_max);
        tsum += vals[i];
    }
    float row_sum = warp_reduce_sum(tsum);
    float inv_sum = 1.0f / row_sum;

    #pragma unroll
    for (int i = 0; i < VPT; i++) vals[i] *= inv_sum;

    float* out_w = output_weights + row * k;
    int32_t* out_idx = output_indices + row * k;
    int32_t* out_src = output_source_rows + row * k;

    float topk_sum = 0.0f;

    for (int ki = 0; ki < k; ki++) {
        float local_max = -1.0f;
        int local_idx = -1;
        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            if (vals[i] > local_max) {
                local_max = vals[i];
                local_idx = my_indices[i];
            }
        }

        float global_max = warp_reduce_max(local_max);
        bool is_winner = (local_max == global_max && local_max > 0.0f);
        unsigned winner_mask = __ballot_sync(0xFFFFFFFF, is_winner);
        int first_winner = __ffs(winner_mask) - 1;

        float winner_val = __shfl_sync(0xFFFFFFFF, local_max, first_winner);
        int winner_idx = __shfl_sync(0xFFFFFFFF, local_idx, first_winner);

        if (tid == 0) {
            out_w[ki] = winner_val;
            out_idx[ki] = winner_idx;
            out_src[ki] = row;
        }
        topk_sum += winner_val;

        #pragma unroll
        for (int i = 0; i < VPT; i++) {
            if (my_indices[i] == winner_idx)
                vals[i] = -1.0f;
        }
    }

    if (renormalize && tid == 0) {
        float inv_topk = 1.0f / (topk_sum + 1e-8f);
        for (int ki = 0; ki < k; ki++)
            out_w[ki] *= inv_topk;
    }
}

std::vector<torch::Tensor> moe_topk_softmax(
    torch::Tensor gating_output, int64_t topk, bool renormalize
) {
    int num_tokens = gating_output.size(0);
    TORCH_CHECK(gating_output.size(1) == 64, "Specialized for 64 experts");

    auto opts_f = torch::dtype(torch::kFloat32).device(gating_output.device());
    auto opts_i = torch::dtype(torch::kInt32).device(gating_output.device());
    auto topk_weights = torch::empty({num_tokens, topk}, opts_f);
    auto topk_ids = torch::empty({num_tokens, topk}, opts_i);
    auto token_expert_ids = torch::empty({num_tokens, topk}, opts_i);

    auto input_f32 = gating_output.to(torch::kFloat32);

    dim3 block(THREADS_PER_ROW, WARPS_PER_CTA);
    dim3 grid((num_tokens + ROWS_PER_CTA - 1) / ROWS_PER_CTA);

    topk_gating_softmax_kernel<<<grid, block, 0,
        c10::cuda::getCurrentCUDAStream()>>>(
        input_f32.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        token_expert_ids.data_ptr<int32_t>(),
        num_tokens, topk, renormalize);

    return {topk_weights, topk_ids, token_expert_ids};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_topk_softmax", &moe_topk_softmax,
          "Fused softmax+topk for MoE routing (64 experts, warp shuffle, zero SMEM)");
}
