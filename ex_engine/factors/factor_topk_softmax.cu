// ex_engine/factors/factor_topk_softmax.cu
//
// Layer 8: MoE topk_softmax CUDA kernel
//
// Upstream parallel: kernels/cuda/moe/moe_topk_softmax_kernels.cuh (867 lines)
//   Originally adapted from:
//     vllm v0.7.3 → csrc/moe/topk_softmax_kernels.cu
//     TensorRT-LLM v0.7.1 → moe_kernels.cu
//     xllm latest → moe_topk_softmax_kernels.cuh
//
// Three kernel paths in upstream:
//   1. topk_gating_softmax<T, VPT, NUM_EXPERTS, WARPS_PER_CTA, BYTES_PER_LDG>
//      → For power-of-2 expert counts (1..256), packs rows into warps
//      → Pure warp shuffle, zero shared memory
//      → Qwen3.5 uses this path: 64 experts → VPT=2, 32 threads/row
//
//   2. moe_topk_fast<TPB>
//      → For non-power-of-2 expert counts, k ≥ 2
//      → Uses CUB BlockReduce with TopKPair (finds 2 maxima per iter)
//      → Requires softmax_workspace for pre-computed softmax
//
//   3. moe_topK<TPB>
//      → For non-power-of-2 expert counts, k = 1
//      → Uses CUB BlockReduce with single cub::ArgMax
//
// BI-V100 SM70 adaptations:
//   - __shfl_xor_sync with full mask 0xFFFFFFFF (SM70 warp shuffle)
//   - No cp.async, no TMA — all loads are standard global loads
//   - cub::BlockReduce via cub/block/block_reduce.cuh (CUB ships with CUDA 10.2)
//   - __launch_bounds__ tuned for SM70: 128 threads, max occupancy
//
// For Qwen3.5-27B: NUM_EXPERTS=64, topk=8, all tokens route here.
// This kernel is called 64 times per forward pass (once per MoE layer).
// At ~6K tokens/batch: 64 × 6K = ~384K kernel launches amortized.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

// CUB for BlockReduce (non-power-of-2 fallback path)
#include <cub/block/block_reduce.cuh>

// =========================================================================
// SM70 warp shuffle macros (BI-V100 compatible)
// =========================================================================
// Upstream uses XLLM_SHFL_XOR_SYNC_WIDTH macro.
// On SM70, standard __shfl_xor_sync with full mask.
#define FULL_MASK 0xFFFFFFFFU
#define WARP_SIZE 32

#ifndef SHFL_XOR_SYNC
#define SHFL_XOR_SYNC(val, mask, width) \
    __shfl_xor_sync(FULL_MASK, (val), (mask), (width))
#endif

#ifndef SHFL_SYNC
#define SHFL_SYNC(val, src, width) \
    __shfl_sync(FULL_MASK, (val), (src), (width))
#endif

// =========================================================================
// Utility: convert generic type to float
// =========================================================================
template <typename T>
__device__ __forceinline__ float to_float(T val);

template <>
__device__ __forceinline__ float to_float<float>(float val) { return val; }

template <>
__device__ __forceinline__ float to_float<__half>(__half val) {
    return __half2float(val);
}

// Aligned array for vectorized loads (replaces CUTLASS dependency)
template <typename T, int N>
struct alignas(sizeof(T) * N) AlignedArray {
    T data[N];
    __device__ __forceinline__ T& operator[](int i) { return data[i]; }
    __device__ __forceinline__ const T& operator[](int i) const { return data[i]; }
};

// =========================================================================
// Compile-time constants
// =========================================================================
// TopkConstants: compute VPT and ROWS_PER_WARP from expert count and load width
template <typename T, int NUM_EXPERTS, int BYTES_PER_LDG>
struct TopkConstants {
    static constexpr int kEltsPerLdg = BYTES_PER_LDG / sizeof(T);
    static constexpr int kThreadsPerRow = NUM_EXPERTS / (sizeof(T) <= 2 ? 2 : 1);
    // Ensure threads_per_row does not exceed WARP_SIZE
    static constexpr int VPT = NUM_EXPERTS / (WARP_SIZE < (NUM_EXPERTS / 1) ? WARP_SIZE : (NUM_EXPERTS / 1));
    static constexpr int ROWS_PER_WARP = WARP_SIZE * VPT / NUM_EXPERTS;
};

// =========================================================================
// Kernel 1: topk_gating_softmax — power-of-2 experts (THE hot path)
// =========================================================================
// This is the primary kernel for Qwen3.5 (64 experts).
// Each warp processes kRowsPerWarp rows simultaneously.
// All reduces via warp shuffle — zero shared memory.

template <typename T, int VPT, int NUM_EXPERTS, int WARPS_PER_CTA, int BYTES_PER_LDG>
__launch_bounds__(WARPS_PER_CTA * WARP_SIZE)
__global__ void topk_gating_softmax_kernel(
    const T*    __restrict__ input,     // (num_rows, NUM_EXPERTS)
    const bool* __restrict__ finished,  // (num_rows,) or NULL
    float*      __restrict__ output,    // (num_rows, k)
    const int   num_rows,
    int*        __restrict__ indices,   // (num_rows, k)
    const int   k,
    const int   start_expert,
    const int   end_expert,
    const bool  renormalize
) {
    // Compile-time geometry
    static constexpr int kEltsPerLdg    = BYTES_PER_LDG / sizeof(T);
    static constexpr int kEltsPerRow    = NUM_EXPERTS;
    static constexpr int kThreadsPerRow = kEltsPerRow / VPT;
    static constexpr int kLdgPerThread  = VPT / kEltsPerLdg;
    static constexpr int kEltsPerWarp   = WARP_SIZE * VPT;
    static constexpr int kRowsPerWarp   = kEltsPerWarp / kEltsPerRow;
    static constexpr int kRowsPerCta    = WARPS_PER_CTA * kRowsPerWarp;
    static constexpr int kColsPerGroupLdg = kEltsPerLdg * kThreadsPerRow;

    // Row assignment
    const int cta_base_row = blockIdx.x * kRowsPerCta;
    const int warp_base_row = cta_base_row + threadIdx.y * kRowsPerWarp;
    const int thread_row_in_warp = threadIdx.x / kThreadsPerRow;
    const int thread_row = warp_base_row + thread_row_in_warp;

    if (thread_row >= num_rows) return;

    const bool row_active = finished ? !finished[thread_row] : true;

    // Read this thread's chunk
    const T* thread_row_ptr = input + thread_row * kEltsPerRow;
    const int thread_group_idx = threadIdx.x % kThreadsPerRow;
    const int first_elt = thread_group_idx * kEltsPerLdg;
    const T* read_ptr = thread_row_ptr + first_elt;

    // Vectorized load
    using AccessType = AlignedArray<T, kEltsPerLdg>;
    T row_chunk_raw[VPT];
    AccessType* vec_ptr = reinterpret_cast<AccessType*>(&row_chunk_raw);
    const AccessType* src_ptr = reinterpret_cast<const AccessType*>(read_ptr);
    #pragma unroll
    for (int ii = 0; ii < kLdgPerThread; ++ii) {
        vec_ptr[ii] = src_ptr[ii * kThreadsPerRow];
    }

    // Convert to float
    float row_chunk[VPT];
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = to_float(row_chunk_raw[ii]);
    }

    // ===== Softmax: max reduction via butterfly =====
    float thread_max = row_chunk[0];
    #pragma unroll
    for (int ii = 1; ii < VPT; ++ii) {
        thread_max = fmaxf(thread_max, row_chunk[ii]);
    }
    #pragma unroll
    for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
        thread_max = fmaxf(thread_max,
            SHFL_XOR_SYNC(thread_max, mask, kThreadsPerRow));
    }

    // ===== Softmax: exp and sum =====
    float row_sum = 0.0f;
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] = expf(row_chunk[ii] - thread_max);
        row_sum += row_chunk[ii];
    }
    #pragma unroll
    for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
        row_sum += SHFL_XOR_SYNC(row_sum, mask, kThreadsPerRow);
    }

    // ===== Normalize =====
    const float inv_sum = 1.0f / row_sum;
    #pragma unroll
    for (int ii = 0; ii < VPT; ++ii) {
        row_chunk[ii] *= inv_sum;
    }

    // ===== TopK via iterative warp argmax =====
    int start_col = first_elt;
    float renorm_sum = 0.0f;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
        // Thread-local argmax
        float max_val = row_chunk[0];
        int expert = start_col;
        #pragma unroll
        for (int ldg = 0, col = start_col; ldg < kLdgPerThread;
             ++ldg, col += kColsPerGroupLdg) {
            #pragma unroll
            for (int ii = 0; ii < kEltsPerLdg; ++ii) {
                float val = row_chunk[ldg * kEltsPerLdg + ii];
                if (val > max_val) {
                    max_val = val;
                    expert = col + ii;
                }
            }
        }

        // Butterfly argmax across thread group
        #pragma unroll
        for (int mask = kThreadsPerRow / 2; mask > 0; mask /= 2) {
            float other_max = SHFL_XOR_SYNC(max_val, mask, kThreadsPerRow);
            int other_expert = SHFL_XOR_SYNC(expert, mask, kThreadsPerRow);
            if (other_max > max_val ||
                (other_max == max_val && other_expert < expert)) {
                max_val = other_max;
                expert = other_expert;
            }
        }

        // Write result (lead thread only)
        if (thread_group_idx == 0) {
            const bool uses_expert = expert >= start_expert && expert < end_expert;
            const bool should_process = row_active && uses_expert;
            const int idx = k * thread_row + k_idx;
            output[idx] = max_val;
            indices[idx] = should_process ? (expert - start_expert) : NUM_EXPERTS;
            renorm_sum += max_val;
        }

        // Suppress winner for next iteration
        if (k_idx + 1 < k) {
            const int winner_ldg = expert / kColsPerGroupLdg;
            const int winner_thread = (expert / kEltsPerLdg) % kThreadsPerRow;
            if (thread_group_idx == winner_thread) {
                const int offset = expert % kEltsPerLdg;
                row_chunk[winner_ldg * kEltsPerLdg + offset] = -10000.0f;
            }
        }
    }

    // Renormalize
    if (renormalize && thread_group_idx == 0) {
        float inv = 1.0f / renorm_sum;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            const int idx = k * thread_row + k_idx;
            output[idx] *= inv;
        }
    }
}

// =========================================================================
// Kernel 2: moe_softmax — generic softmax for non-power-of-2 fallback
// =========================================================================
// Uses CUB BlockReduce for max and sum across arbitrary expert counts.
// Writes softmax probabilities to output buffer for subsequent topk.

template <typename T, int TPB>
__launch_bounds__(TPB)
__global__ void moe_softmax_kernel(
    const T*    __restrict__ input,   // (num_tokens, num_cols)
    float*      __restrict__ output,  // (num_tokens, num_cols)
    const int   num_cols
) {
    using BlockReduce = cub::BlockReduce<float, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;
    __shared__ float s_max;
    __shared__ float s_norm;

    const int row_offset = blockIdx.x * num_cols;

    // Pass 1: find max
    float thread_max = -FLT_MAX;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        float val = to_float(input[row_offset + ii]);
        output[row_offset + ii] = val;  // store converted value
        thread_max = fmaxf(thread_max, val);
    }
    float block_max = BlockReduce(tmp_storage).Reduce(thread_max, cub::Max());
    if (threadIdx.x == 0) s_max = block_max;
    __syncthreads();

    // Pass 2: exp and sum
    float thread_sum = 0.0f;
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        float val = expf(output[row_offset + ii] - s_max);
        output[row_offset + ii] = val;
        thread_sum += val;
    }
    float block_sum = BlockReduce(tmp_storage).Sum(thread_sum);
    if (threadIdx.x == 0) s_norm = 1.0f / block_sum;
    __syncthreads();

    // Pass 3: normalize
    for (int ii = threadIdx.x; ii < num_cols; ii += TPB) {
        output[row_offset + ii] *= s_norm;
    }
}

// =========================================================================
// Kernel 3: moe_topk_fast — topk from pre-computed softmax (k ≥ 2)
// =========================================================================
// Uses CUB BlockReduce with TopKPair to find 2 maxima per iteration.
// Upstream: moe_topk_fast, uses cub::KeyValuePair.

using cub_kvp = cub::KeyValuePair<int, float>;

template <int TPB>
__launch_bounds__(TPB)
__global__ void moe_topk_fast_kernel(
    float*       __restrict__ probs,      // (N, E) — modified in-place
    float*       __restrict__ output,     // (N, k)
    int*         __restrict__ indices,    // (N, k)
    const int    num_experts,
    const int    k,
    const int    start_expert,
    const int    end_expert,
    const bool   renormalize
) {
    using BlockReduce = cub::BlockReduce<cub_kvp, TPB>;
    __shared__ typename BlockReduce::TempStorage tmp_storage;

    const int row = blockIdx.x;
    const int row_offset = row * num_experts;
    float renorm_sum = 0.0f;

    cub::ArgMax arg_max;

    for (int k_idx = 0; k_idx < k; ++k_idx) {
        cub_kvp thread_kvp;
        thread_kvp.key = 0;
        thread_kvp.value = -1.0f;

        for (int e = threadIdx.x; e < num_experts; e += TPB) {
            cub_kvp inp;
            inp.key = e;
            inp.value = probs[row_offset + e];
            thread_kvp = arg_max(inp, thread_kvp);
        }

        cub_kvp result = BlockReduce(tmp_storage).Reduce(thread_kvp, arg_max);

        if (threadIdx.x == 0) {
            const int expert = result.key;
            const bool uses = expert >= start_expert && expert < end_expert;
            const int idx = k * row + k_idx;
            output[idx] = result.value;
            indices[idx] = uses ? (expert - start_expert) : num_experts;
            renorm_sum += result.value;
            // Suppress winner
            probs[row_offset + expert] = -1.0f;
        }
        __syncthreads();
    }

    if (renormalize && threadIdx.x == 0) {
        float inv = 1.0f / renorm_sum;
        for (int k_idx = 0; k_idx < k; ++k_idx) {
            output[k * row + k_idx] *= inv;
        }
    }
}

// =========================================================================
// Host-side launcher with template dispatch by expert count
// =========================================================================
// Matches upstream topk_gating_softmax_kernel_launcher pattern.
// Power-of-2 experts → topk_gating_softmax (zero shared mem, warp shuffle)
// Other → moe_softmax + moe_topk_fast (CUB path)

template <typename T, int EXPERTS, int WARPS_PER_TB>
void launch_topk_gating(
    const T* input, float* output, int* indices,
    int num_rows, int k, int start_expert, int end_expert,
    bool renormalize, cudaStream_t stream
) {
    // For SM70: BYTES_PER_LDG capped at min(16, sizeof(T)*EXPERTS)
    static constexpr int kBytesPerLdg =
        (16 < (int)(sizeof(T) * EXPERTS)) ? 16 : (int)(sizeof(T) * EXPERTS);
    static constexpr int kEltsPerLdg = kBytesPerLdg / sizeof(T);
    static constexpr int kVpt = EXPERTS / WARP_SIZE;
    // Ensure VPT ≥ 1
    static constexpr int VPT = (kVpt > 0) ? kVpt : 1;
    static constexpr int kRowsPerWarp = (WARP_SIZE * VPT) / EXPERTS;
    static constexpr int kRowsPerCta = WARPS_PER_TB * ((kRowsPerWarp > 0) ? kRowsPerWarp : 1);

    const int num_blocks = (num_rows + kRowsPerCta - 1) / kRowsPerCta;
    dim3 block(WARP_SIZE, WARPS_PER_TB);

    topk_gating_softmax_kernel<T, VPT, EXPERTS, WARPS_PER_TB, kBytesPerLdg>
        <<<num_blocks, block, 0, stream>>>(
            input, nullptr, output, num_rows, indices,
            k, start_expert, end_expert, renormalize);
}

// Macro for dispatch table
#define LAUNCH_GATING(TYPE, EXPERTS, WARPS)       \
    launch_topk_gating<TYPE, EXPERTS, WARPS>(     \
        gating_ptr, topk_weights, topk_indices,   \
        num_tokens, topk, 0, num_experts,         \
        renormalize, stream);

// =========================================================================
// Host entry point: topk_softmax (matches ixformer::infer::topk_softmax)
// =========================================================================

extern "C" void ex_topk_softmax(
    float*       topk_weights,    // (num_tokens, topk) output
    int*         topk_indices,    // (num_tokens, topk) output
    const float* gating_output,   // (num_tokens, num_experts) input
    int          num_tokens,
    int          num_experts,
    int          topk,
    bool         renormalize,
    cudaStream_t stream
) {
    const float* gating_ptr = gating_output;
    const bool is_pow2 = (num_experts & (num_experts - 1)) == 0;

    if (is_pow2 && num_experts <= 256) {
        // Fast path: topk_gating_softmax with warp shuffle
        static constexpr int kWarps = 4;
        switch (num_experts) {
            case 1:   LAUNCH_GATING(float, 1,   kWarps); break;
            case 2:   LAUNCH_GATING(float, 2,   kWarps); break;
            case 4:   LAUNCH_GATING(float, 4,   kWarps); break;
            case 8:   LAUNCH_GATING(float, 8,   kWarps); break;
            case 16:  LAUNCH_GATING(float, 16,  kWarps); break;
            case 32:  LAUNCH_GATING(float, 32,  kWarps); break;
            case 64:  LAUNCH_GATING(float, 64,  kWarps); break;
            case 128: LAUNCH_GATING(float, 128, kWarps); break;
            case 256: LAUNCH_GATING(float, 256, kWarps); break;
        }
    } else {
        // Fallback: softmax + topk via CUB
        static constexpr int kTpb = 256;

        // Allocate workspace for softmax output
        float* workspace;
        cudaMalloc(&workspace, (size_t)num_tokens * num_experts * sizeof(float));

        moe_softmax_kernel<float, kTpb>
            <<<num_tokens, kTpb, 0, stream>>>(
                gating_output, workspace, num_experts);

        moe_topk_fast_kernel<kTpb>
            <<<num_tokens, kTpb, 0, stream>>>(
                workspace, topk_weights, topk_indices,
                num_experts, topk, 0, num_experts, renormalize);

        cudaFree(workspace);
    }
}
