// ex_engine/csrc/factor_moe_topk_softmax.cu
//
// Factor 0: MOE_TOPK_SOFTMAX — fused softmax + top-k for MoE routing
//
// CCCL reference: cub/device/dispatch/tuning/tuning_topk.cuh
//   worker_policy levels 1-6 with items_per_thread = {64,32,16,12,8,2}
//   Selects smallest sufficient policy based on segment_size
//
// BI-V100 target: SM70, 16 SMs, 49152 bytes SMEM, no cp.async
// Input: router_logits (T, num_experts) where num_experts=64 for Qwen3.5-MoE
// Output: topk_weights (T, top_k), topk_ids (T, top_k) with top_k=8
//
// This replaces: torch.softmax(router_logits, dim=-1) → torch.topk(..., k=8)
// Fusing saves: 1 full pass over (T, 64) tensor + 1 partial sort

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <float.h>
#include <stdint.h>

// External C interface
extern "C" {
#include "ex_engine.h"
}

// ---------------------------------------------------------------------------
// Kernel: fused softmax + topk for MoE routing
//
// One CTA per token (T tokens total).
// Each CTA handles num_experts values, finds top_k winners.
// For num_experts=64, top_k=8: fits perfectly in 2 warps (64 threads).
//
// CCCL analogy: this is a single-tile reduce (num_experts fits in one tile)
// with a radix-select epilogue instead of a simple accumulate.
// ---------------------------------------------------------------------------

// Tuning for BI-V100: 64 experts → 64 threads (1 expert per thread)
// Each thread holds its logit, does warp shuffle for max/sum, then
// bitonic partial sort for top-k.
static constexpr int BLOCK_SIZE = 64;  // == num_experts
static constexpr int TOP_K = 8;

// Warp-level max reduction
__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Warp-level sum reduction
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__global__ void moe_topk_softmax_kernel(
    float* __restrict__ topk_weights,   // (T, top_k)
    int32_t* __restrict__ topk_ids,     // (T, top_k)
    const float* __restrict__ logits,   // (T, num_experts)
    int T,
    int num_experts,
    int top_k
) {
    int token_idx = blockIdx.x;
    if (token_idx >= T) return;

    int tid = threadIdx.x;
    const float* my_logits = logits + token_idx * num_experts;

    // Step 1: Load my logit (1 per thread for 64 experts)
    float my_val = (tid < num_experts) ? my_logits[tid] : -FLT_MAX;
    int my_id = tid;

    // Step 2: Online softmax — find max across all experts (2-warp reduction)
    __shared__ float s_max[2];
    __shared__ float s_sum[2];

    int warp_id = tid / 32;
    float warp_max = warp_reduce_max(my_val);
    if (tid % 32 == 0) s_max[warp_id] = warp_max;
    __syncthreads();

    float global_max = fmaxf(s_max[0], s_max[1]);

    // Step 3: Compute exp(x - max) — numerically stable softmax
    float my_exp = (tid < num_experts) ? expf(my_val - global_max) : 0.0f;

    // Step 4: Sum for normalization
    float warp_sum = warp_reduce_sum(my_exp);
    if (tid % 32 == 0) s_sum[warp_id] = warp_sum;
    __syncthreads();

    float global_sum = s_sum[0] + s_sum[1];
    float my_prob = my_exp / global_sum;  // softmax output

    // Step 5: Top-K selection via shared memory partial sort
    // Use shared memory to collect all (prob, id) pairs, then
    // do a register-based bitonic top-K.
    __shared__ float  s_probs[64];
    __shared__ int    s_ids[64];
    s_probs[tid] = my_prob;
    s_ids[tid]   = my_id;
    __syncthreads();

    // Thread 0 does a simple insertion sort for top_k=8 from 64 elements
    // This is faster than full bitonic for k << n.
    // 64 elements × 8 comparisons = 512 ops (fits in registers)
    if (tid < top_k) {
        // Each of the first top_k threads finds one winner
        // We use a parallel argmax approach: each thread looks for
        // the (tid+1)-th largest element.
        // Simple approach: tid=0 finds max, tid=1 finds 2nd max, etc.
        // Use iterative suppression in shared memory.

        // Actually, simpler: thread 0 does all work (64 experts is tiny)
    }

    if (tid == 0) {
        float* out_w = topk_weights + token_idx * top_k;
        int32_t* out_id = topk_ids + token_idx * top_k;

        // Insertion sort top-K from 64 elements
        // Initialize with -inf
        float best_w[8];
        int   best_id[8];
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) {
            best_w[k] = -1.0f;
            best_id[k] = -1;
        }

        for (int e = 0; e < num_experts; e++) {
            float p = s_probs[e];
            // Insert into sorted top-K
            if (p > best_w[TOP_K - 1]) {
                best_w[TOP_K - 1] = p;
                best_id[TOP_K - 1] = e;
                // Bubble up
                #pragma unroll
                for (int k = TOP_K - 1; k > 0; k--) {
                    if (best_w[k] > best_w[k-1]) {
                        float tw = best_w[k]; best_w[k] = best_w[k-1]; best_w[k-1] = tw;
                        int ti = best_id[k]; best_id[k] = best_id[k-1]; best_id[k-1] = ti;
                    }
                }
            }
        }

        // Renormalize top-K weights
        float sum_topk = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; k++) sum_topk += best_w[k];
        float inv_sum = (sum_topk > 0.0f) ? (1.0f / sum_topk) : 0.0f;

        #pragma unroll
        for (int k = 0; k < top_k; k++) {
            out_w[k]  = best_w[k] * inv_sum;
            out_id[k] = best_id[k];
        }
    }
}

// ---------------------------------------------------------------------------
// Factor entry point
// ---------------------------------------------------------------------------

static int moe_topk_softmax_dispatch(
    void*          output,
    const void*    input,
    const void*    aux_inputs[],
    int            n_aux,
    const int64_t  dims[],
    int            n_dims,
    void*          stream
) {
    // dims[0] = T (tokens), dims[1] = num_experts, dims[2] = top_k
    // output points to topk_weights buffer, aux_inputs[0] = topk_ids buffer
    if (n_dims < 3 || !output || !input || !aux_inputs || n_aux < 1) return -1;

    int T = (int)dims[0];
    int num_experts = (int)dims[1];
    int top_k = (int)dims[2];

    float* topk_weights = (float*)output;
    int32_t* topk_ids = (int32_t*)aux_inputs[0];
    const float* logits = (const float*)input;

    cudaStream_t cu_stream = (cudaStream_t)stream;

    dim3 grid(T);
    dim3 block(BLOCK_SIZE);

    moe_topk_softmax_kernel<<<grid, block, 0, cu_stream>>>(
        topk_weights, topk_ids, logits, T, num_experts, top_k
    );

    return 0;
}

// ---------------------------------------------------------------------------
// .so export
// ---------------------------------------------------------------------------

static ex_factor_t s_factor;

extern "C" ex_factor_t* ex_get_factor(const ex_hardware_t* hw) {
    s_factor.factor_id   = EX_FACTOR_MOE_TOPK_SOFTMAX;
    s_factor.name        = "moe_topk_softmax";
    s_factor.version     = "1.0.0";
    s_factor.tuning      = (ex_tuning_t){
        .threads_per_block = BLOCK_SIZE,  // 64 (== num_experts)
        .items_per_thread  = 1,
        .vec_size          = 1,
        .shared_mem_bytes  = 64 * (sizeof(float) + sizeof(int)) + 4 * sizeof(float),
        .num_warps         = 2,
        .num_stages        = 1  // no async on SM70
    };
    s_factor.kernel          = moe_topk_softmax_dispatch;
    s_factor.kernel_fallback = NULL;
    return &s_factor;
}
