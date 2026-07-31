// muh/include/muh/tuning/tuning_segmented_reduce.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_reduce.cuh
// CCCL: delegates to reduce::policy_selector, adds warp-level reduce policies
//
// vllm relevance: per-head attention score aggregation
// SMEM risk: inherits from reduce (already validated)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"
#include "muh/tuning/tuning_reduce.cuh"

namespace muh::tuning::segmented_reduce {

struct SegmentedReduceWarpReducePolicy {
  int threads_per_block;
  int threads_per_warp;
  int items_per_thread;
  int vec_size;
  CacheLoadModifier load_modifier;
};

struct SegmentedReducePolicy {
  reduce::ReducePassPolicy large_segment;
  SegmentedReduceWarpReducePolicy medium_segment;
  SegmentedReduceWarpReducePolicy small_segment;
};

struct policy_selector {
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;

  constexpr SegmentedReducePolicy operator()(const hardware_capability& hw) const {
    // Delegate to reduce for the large-segment policy
    auto rp = reduce::policy_selector{accum_t, operation_t, offset_size, accum_size}(hw).multi_tile;

    return {
      rp,
      // medium: full warp (32 threads) per segment
      {rp.threads_per_block, 32, rp.items_per_thread, rp.vec_size, rp.load_modifier},
      // small: single thread per segment
      {rp.threads_per_block, 1, rp.items_per_thread, rp.vec_size, rp.load_modifier},
    };
  }
};

} // namespace muh::tuning::segmented_reduce
