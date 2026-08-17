// ex_engine/factors/factor_moe_combine.cu
//
// Layer 10: MoE weighted combine kernel
//
// Upstream parallel: kernels/cuda/moe/moe_combine.cu (105 lines)
//   Fused reorder + weighted sum replacing:
//     torch::zeros + index_copy_ + view + multiply + sum
//
// Algorithm per token (each block handles one output token):
//   For each of its topk experts:
//     1. Read expert output at the flat index position
//     2. Multiply by the router weight for this (token, expert) pair
//     3. Accumulate into output[token] in fp32
//   Then cast back to input dtype.
//
// Grid:  N blocks (one per output token)
// Block: 256 threads, each handling hidden_dim / 256 elements
//
// For Qwen3.5: hidden_size=3584, topk=8
//   Each block reads 8 × 3584 = 28672 values and produces 3584 outputs.
//   Compute: 8 FMA per element → 3584 × 8 = 28672 FMA → negligible.
//   Bandwidth: 28672 × 2 bytes (fp16 read) + 3584 × 2 (fp16 write) = ~61 KB.
//   At 900 GB/s: ~68 ns per block → fully bandwidth bound.
//
// BI-V100 SM70 adaptations:
//   - Template on scalar_t (half, bfloat16, float)
//   - fp32 accumulation to prevent overflow
//   - 256 threads per block (8 warps, good SM70 occupancy)
//   - Optional residual add (fused shared expert output)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

// =========================================================================
// Compile-time constants
// =========================================================================
static constexpr int32_t kCombineBlockSize = 256;

// =========================================================================
// Device helpers: type conversion to/from float
// =========================================================================
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

template <typename T>
__device__ __forceinline__ T from_float(float val);

template <>
__device__ __forceinline__ float from_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ __half from_float<__half>(float val) {
    return __float2half(val);
}

// =========================================================================
// Kernel: moe_combine_kernel
// =========================================================================
// Each block processes one output token.
// Threads stride over the hidden dimension.
// Accumulation in fp32 prevents overflow for fp16 inputs.
//
// Memory layout (after expert dispatch):
//   gemm2_out: (N × topk, H) — expert outputs in flat-index order
//     Token t's k-th expert output is at gemm2_out[(t × topk + k), :]
//   reduce_weight: (N × topk,) or (N, topk) — router weights
//   output: (N, H) — final combined output

template <typename scalar_t>
__global__ void moe_combine_kernel(
    const scalar_t* __restrict__ gemm2,          // (N*topk, H) expert outputs
    const float*    __restrict__ reduce_weight,   // (N*topk,) or (N, topk)
    scalar_t*       __restrict__ output,          // (N, H) final output
    const scalar_t* __restrict__ residual,        // (N, H) optional residual, NULL if none
    int64_t         N,                            // number of output tokens
    int32_t         topk,                         // experts per token
    int64_t         H                             // hidden dimension
) {
    const int64_t token_id = blockIdx.x;
    if (token_id >= N) return;

    const int32_t tid = threadIdx.x;
    const int32_t stride = kCombineBlockSize;

    // Process hidden dimension elements in strided fashion
    for (int64_t h = tid; h < H; h += stride) {
        float acc = 0.0f;

        // Accumulate over topk experts
        for (int32_t k = 0; k < topk; ++k) {
            int64_t flat_idx = token_id * topk + k;
            float w = reduce_weight[flat_idx];
            float val = to_float(gemm2[flat_idx * H + h]);
            acc += w * val;
        }

        // Add residual if present (shared expert output)
        if (residual != nullptr) {
            acc += to_float(residual[token_id * H + h]);
        }

        output[token_id * H + h] = from_float<scalar_t>(acc);
    }
}

// =========================================================================
// Host-side launcher
// =========================================================================
// Dispatches by dtype. Matches upstream xllm::kernel::cuda::moe_combine_result.
// The upstream version also supports bfloat16; we handle fp16 and fp32
// for BI-V100 (which lacks native bf16 tensor cores).

extern "C" int ex_moe_combine(
    const void*  gemm2_ptr,       // (N*topk, H) device pointer
    const float* reduce_weight,    // (N*topk,) device pointer
    void*        output_ptr,       // (N, H) device pointer
    const void*  residual_ptr,     // (N, H) device pointer, NULL if none
    int64_t      N,                // number of tokens
    int32_t      topk,             // experts per token
    int64_t      H,                // hidden dimension
    int          dtype,            // 0 = fp32, 1 = fp16
    cudaStream_t stream
) {
    if (dtype == 1) {
        // fp16 path — primary for Qwen3.5 inference
        moe_combine_kernel<__half>
            <<<N, kCombineBlockSize, 0, stream>>>(
                reinterpret_cast<const __half*>(gemm2_ptr),
                reduce_weight,
                reinterpret_cast<__half*>(output_ptr),
                residual_ptr ? reinterpret_cast<const __half*>(residual_ptr) : nullptr,
                N, topk, H);
    } else {
        // fp32 path — for debugging or fp32 inference
        moe_combine_kernel<float>
            <<<N, kCombineBlockSize, 0, stream>>>(
                reinterpret_cast<const float*>(gemm2_ptr),
                reduce_weight,
                reinterpret_cast<float*>(output_ptr),
                residual_ptr ? reinterpret_cast<const float*>(residual_ptr) : nullptr,
                N, topk, H);
    }

    return 0;
}
