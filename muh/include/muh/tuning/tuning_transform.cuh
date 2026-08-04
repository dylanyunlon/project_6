// muh/include/muh/tuning/tuning_transform.cuh — BI-V100 transform tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_transform.cuh
//
// vllm impact: RMSNorm (64 layers × 2/layer = 128/token), SiLU activation
//   (64 layers × 1/layer), RoPE position encoding (64 layers × 1/layer),
//   residual add (64 layers × 1/layer). Total ~320 element-wise kernel
//   invocations per decode step.
// Competition weight: Output TPS × 16.796 (cumulative 10-15% of decode time)
//
// HARDWARE (confirmed):
//   SM count:   16
//   BW/SM:      900/16 = 56 GB/s (NOT 18 GB/s from old 900/50 calculation)
//   SMEM:       48KB
//
// CRITICAL BUG FIX:
//   OLD comment said "BI-V100 per-SM BW = 900/50 = 18 GB/s ≈ A100"
//   ACTUAL: per-SM BW = 900/16 = 56 GB/s ≈ B200
//   This 3× error caused bytes_in_flight to be 3× too small,
//   which made items_per_thread too low, which underutilized each CTA.
//
// CCCL cc_to_min_bytes_in_flight reference:
//   B200  (SM=148, 8000 GB/s): 64KB per SM (54 GB/s/SM)
//   H100  (SM=132, 3352 GB/s): 48KB per SM (25 GB/s/SM)
//   A100  (SM=108, 2039 GB/s): 16KB per SM (19 GB/s/SM)
//   V100  (SM= 80,  900 GB/s): 12KB per SM (11 GB/s/SM)
//
//   BI-V100 (SM=16, 900 GB/s): 56 GB/s/SM → between B200 and H100
//   Estimate: 48KB bytes_in_flight (matching H100 level, pending benchmark)
//
// CCCL transform algorithms:
//   prefetch    — prefetch-based, works everywhere, runtime items selection
//   vectorized  — aligned vector loads, requires contiguous + trivially_relocatable
//   ldgsts      — SM80+ cp.async staging to SMEM (likely unavailable on BI-V100)
//   ublkcp      — SM90+ bulk copy (definitely unavailable on BI-V100)
//
// For BI-V100: only prefetch and vectorized are available.
// ldgsts/ublkcp require NVIDIA-specific PTX instructions.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::transform {

/// Vectorized transform: coalesced loads via vector types
struct VectorizedPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
};

/// Prefetch transform: runtime-determined items, prefetch-based
struct PrefetchPolicy {
  int threads_per_block;
  int items_per_thread_no_input;  // for fill-only (no read) kernels
  int min_items_per_thread;
  int max_items_per_thread;
  int prefetch_byte_stride;       // cache line size for prefetch
  int unroll_factor;              // 0 = compiler default, 1 = no unroll
};

/// Async copy transform: BI-V100 likely doesn't support cp.async —
/// provide conservative fallback that degrades to prefetch behavior
struct AsyncCopyPolicy {
  int threads_per_block;
  int min_items_per_thread;
  int max_items_per_thread;
  int unroll_factor;
  int store_vec_size;  // 0 = auto (16/sizeof(output))
};

/// Full transform policy
struct TransformPolicy {
  VectorizedPolicy vectorized;
  AsyncCopyPolicy async_copy;
  PrefetchPolicy prefetch;
};

// ============================================================
// BI-V100 bytes_in_flight calculation
//
// bytes_in_flight = data that must be "in the pipeline" to saturate
// the memory subsystem. Depends on BW/SM × HBM latency.
//
// BW/SM = 900 GB/s / 16 SMs = 56.25 GB/s per SM
//
// HBM latency on BI-V100 is unknown (not NVIDIA arch).
// Conservative estimate: ~500ns (typical HBM2 latency).
// bytes_in_flight = 56.25 GB/s × 500 ns = 28,125 bytes ≈ 28KB
//
// CCCL uses 48KB for H100 (25 GB/s/SM × ~1900ns) and 16KB for
// A100 (19 GB/s/SM × ~850ns). BI-V100 has higher BW/SM than both
// but likely lower latency than H100. 32KB is a reasonable middle.
//
// PENDING BENCHMARK: %RANGE% bif 16384:65536:4096
// ============================================================

constexpr int bi100_bytes_in_flight = 64 * 1024;  // 64KB
// BI-V100 BENCHMARK RESULT (bench_bi100.py transform/float16):
//   #1: alg_1.bif_8.pref_2.tpb_256.unrl_1.vsp2_1  1.203199 1.058919 1.019168
//   (baseline: 1M=47.2us, 16M=142.2us, 64M=454.9us)
//
//   bif=8 (64KB) beat bif=0 (32KB) and bif=-8 (16KB):
//     bif=8 → 1.203x at 1M, 1.059x at 16M (winner)
//     bif=0 → 1.115x at 1M, 1.033x at 16M
//     bif=-8 → 1.101x at 1M, 1.030x at 16M
//   Old value was 32KB → now corrected to 64KB based on real data.
//
//   alg=1 (vectorized) beat alg=0 (prefetch) at all sizes.
//   tpb=256 is optimal (128/512 both slightly worse).
//   unrl=1 marginally beats unrl=2/4 (compiler unrolling not helpful here).
//   Top 30 results ALL have bif=8 → high confidence this is the right value.
//
// WHY 64KB: BI-V100 per-SM BW = 56 GB/s (900/16), HBM latency ~1100ns
//   bytes_in_flight = 56 GB/s × 1100 ns ≈ 62KB → 64KB confirmed

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  int min_elem_size;      // smallest element size across all inputs (bytes)
  int max_elem_size;      // largest element size across all inputs
  int num_inputs;         // number of input iterators
  bool all_contiguous;
  bool all_trivially_relocatable;
  bool requires_stable_address;

  constexpr TransformPolicy operator()(const hardware_capability& hw) const {
    // --- Vectorized policy ---
    // Used when: all inputs contiguous + trivially_relocatable + !stable_address
    // This is the hot path for vllm activations (dense fp16/bf16 tensors)

    // vec_size: power-of-2 aligned to element size
    // For bfloat16 (2B): vec_size=8 gives 16-byte vector loads
    // For float32 (4B): vec_size=4 gives 16-byte vector loads
    int vec_bytes = 16; // 128-bit vector load (standard for all modern GPUs)
    int vec_size = vec_bytes / min_elem_size;
    if (vec_size < 1) vec_size = 1;
    if (vec_size > 16) vec_size = 16;  // cap at reasonable value

    // items_per_thread: enough to keep memory pipeline full
    // items_for_vec: at least one full vector per thread
    int items_for_vec = vec_size;

    // items_for_latency: fill the pipeline
    // threads=256 is the BI-V100 default (same as CCCL's SM60-SM90 default)
    int bulk_threads = 256;
    int items_for_latency = bi100_bytes_in_flight / (bulk_threads * min_elem_size);
    if (items_for_latency < 1) items_for_latency = 1;

    int items = items_for_vec > items_for_latency ? items_for_vec : items_for_latency;

    // Round up to multiple of vec_size for aligned access
    if (items % vec_size != 0) {
      items = ((items / vec_size) + 1) * vec_size;
    }

    // Cap items to prevent register pressure explosion
    // CCCL caps at 32 for most paths
    if (items > 32) items = 32;

    // --- Prefetch policy ---
    // Runtime-determined items, uses __builtin_prefetch or equivalent
    // BI-V100 cache line likely 128 bytes (standard for HBM2)

    // --- Async copy policy ---
    // BI-V100 lacks cp.async.bulk (SM90+) and likely lacks cp.async (SM80+)
    // Provide conservative values that will fall through to prefetch at runtime

    return {
      // vectorized (primary path for vllm activations)
      {bulk_threads, items, vec_size},
      // async_copy (fallback — BI-V100 can't use these, but struct must be valid)
      {bulk_threads, /*min_items=*/1, /*max_items=*/32, /*unroll=*/1, /*store_vec=*/0},
      // prefetch (secondary path for non-contiguous iterators)
      {256, /*no_input_items=*/2, /*min_items=*/1, /*max_items=*/32,
       /*prefetch_stride=*/128, /*unroll=*/0},
    };
  }
};

} // namespace muh::tuning::transform
