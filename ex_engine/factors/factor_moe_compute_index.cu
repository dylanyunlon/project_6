// ex_engine/factors/factor_moe_compute_index.cu
//
// Layer 9: MoE token index computation — 3-phase CUDA kernel
//
// Upstream parallel: kernels/cuda/moe/moe_compute_index.cu (155 lines)
//   Fused MoE token index computation replacing:
//     torch::bincount + 2 × torch::argsort + torch::cumsum + CPU sync
//
//   Phase 1: histogram     — atomicAdd per-expert token counts
//   Phase 2: prefix_sum    — 1 block, CUB BlockScan exclusive scan
//   Phase 3: place_indices — atomicAdd on offsets, write bidirectional maps
//
// This kernel is called once per MoE layer (64 layers per forward).
// Input: expert_id tensor (num_tokens × topk flat), int32
// Output: src_to_dst, dst_to_src, expert_sizes
//
// BI-V100 SM70 adaptations:
//   - CUB BlockScan via cub/block/block_scan.cuh (CUDA 10.2 compatible)
//   - atomicAdd for int32 (SM70 native)
//   - kMoeIndexBlock = 256 threads per block
//   - Single-block prefix_sum (num_experts ≤ 256 guaranteed for Qwen3.5)

#include <cuda_runtime.h>
#include <stdint.h>

// CUB for BlockScan (exclusive prefix sum in shared memory)
#include <cub/block/block_scan.cuh>

// =========================================================================
// Compile-time constants
// =========================================================================
static constexpr int32_t kMoeIndexBlock = 256;

// =========================================================================
// Phase 1: Histogram — count tokens per expert
// =========================================================================
// Each thread processes one element of the flat expert_id array.
// atomicAdd to expert_sizes[eid] to build the histogram.
// Grid: ceil(N / 256) blocks

__global__ void moe_histogram_kernel(
    const int32_t* __restrict__ expert_id,    // (N,) flat expert assignments
    int32_t*       __restrict__ expert_sizes,  // (num_experts,) output counts
    int64_t        num_elements,
    int32_t        num_experts
) {
    int64_t tid = (int64_t)blockIdx.x * kMoeIndexBlock + threadIdx.x;
    if (tid < num_elements) {
        int32_t eid = expert_id[tid];
        if (eid >= 0 && eid < num_experts) {
            atomicAdd(&expert_sizes[eid], 1);
        }
    }
}

// =========================================================================
// Phase 2: Exclusive prefix sum — compute expert offsets
// =========================================================================
// Single block, one thread per expert (num_experts ≤ 256).
// Uses CUB BlockScan for an efficient exclusive prefix sum.
// Input:  expert_sizes  (per-expert token counts from Phase 1)
// Output: expert_offsets (exclusive scan — start position per expert)
//
// The exclusive scan means expert_offsets[e] = sum(expert_sizes[0..e-1]).
// This gives the starting position in the sorted token array for expert e.

__global__ void moe_prefix_sum_kernel(
    const int32_t* __restrict__ expert_sizes,    // (num_experts,) counts
    int32_t*       __restrict__ expert_offsets,   // (num_experts,) output offsets
    int32_t        num_experts
) {
    using BlockScan = cub::BlockScan<int32_t, kMoeIndexBlock>;
    __shared__ typename BlockScan::TempStorage s_scan;

    // Each thread loads one expert's count (0 if out of range)
    int32_t val = (threadIdx.x < num_experts) ? expert_sizes[threadIdx.x] : 0;
    int32_t offset;

    // Exclusive sum: offset[i] = sum(val[0..i-1])
    BlockScan(s_scan).ExclusiveSum(val, offset);
    __syncthreads();

    if (threadIdx.x < num_experts) {
        expert_offsets[threadIdx.x] = offset;
    }
}

// =========================================================================
// Phase 3: Place indices — build bidirectional permutation maps
// =========================================================================
// For each token assignment (flat_idx, expert_id):
//   pos = atomicAdd(&expert_offsets[eid], 1)  — claim next slot
//   dst_to_src[pos] = flat_idx                — sorted→original mapping
//   src_to_dst[flat_idx] = pos                — original→sorted mapping
//
// After this kernel:
//   dst_to_src contains token indices grouped by expert
//   src_to_dst[i] tells where token i ended up in the sorted order
//
// Note: expert_offsets is consumed destructively (atomicAdd increments it).
//       The caller must keep expert_sizes separately.

__global__ void moe_place_indices_kernel(
    const int32_t* __restrict__ expert_id,      // (N,) flat expert assignments
    int32_t*       __restrict__ expert_offsets,  // (num_experts,) — destructive
    int32_t*       __restrict__ dst_to_src,      // (N,) output: sorted→original
    int32_t*       __restrict__ src_to_dst,      // (N,) output: original→sorted
    int64_t        num_elements,
    int32_t        num_experts
) {
    int64_t flat_idx = (int64_t)blockIdx.x * kMoeIndexBlock + threadIdx.x;
    if (flat_idx >= num_elements) return;

    int32_t eid = expert_id[flat_idx];
    if (eid < 0 || eid >= num_experts) return;

    // Claim the next position in this expert's segment
    int32_t pos = atomicAdd(&expert_offsets[eid], 1);

    // Write bidirectional mapping
    dst_to_src[pos] = (int32_t)flat_idx;
    src_to_dst[flat_idx] = pos;
}

// =========================================================================
// Host-side orchestrator — launches all 3 phases
// =========================================================================
// Matches upstream xllm::kernel::cuda::moe_compute_index signature.
//
// Usage from ixformer::infer::moe_compute_token_index_api:
//   Phase 1: histogram → expert_sizes
//   Phase 2: prefix_sum → expert_offsets (scratch, used by Phase 3)
//   Phase 3: place_indices → dst_to_src, src_to_dst
//
// All 3 phases run on the same CUDA stream with implicit synchronization
// (each kernel completes before the next starts within the stream).

extern "C" int ex_moe_compute_index(
    const int32_t* expert_id,      // (N,) device pointer
    int32_t*       src_to_dst,     // (N,) device pointer, output
    int32_t*       dst_to_src,     // (N,) device pointer, output
    int32_t*       expert_sizes,   // (num_experts,) device pointer, output
    int64_t        num_elements,   // N = num_tokens × topk
    int32_t        num_experts,    // E (64 for Qwen3.5)
    cudaStream_t   stream
) {
    if (num_experts > kMoeIndexBlock) return -1;  // Exceeds single-block scan

    int64_t grid = (num_elements + kMoeIndexBlock - 1) / kMoeIndexBlock;

    // Zero expert_sizes before histogram
    cudaMemsetAsync(expert_sizes, 0, num_experts * sizeof(int32_t), stream);

    // Allocate scratch for expert_offsets
    int32_t* expert_offsets;
    cudaMalloc(&expert_offsets, num_experts * sizeof(int32_t));

    // Phase 1: histogram
    moe_histogram_kernel<<<grid, kMoeIndexBlock, 0, stream>>>(
        expert_id, expert_sizes, num_elements, num_experts);

    // Phase 2: prefix sum (single block)
    moe_prefix_sum_kernel<<<1, kMoeIndexBlock, 0, stream>>>(
        expert_sizes, expert_offsets, num_experts);

    // Phase 3: place indices (destructive on expert_offsets)
    moe_place_indices_kernel<<<grid, kMoeIndexBlock, 0, stream>>>(
        expert_id, expert_offsets, dst_to_src, src_to_dst,
        num_elements, num_experts);

    cudaFree(expert_offsets);

    return 0;
}
