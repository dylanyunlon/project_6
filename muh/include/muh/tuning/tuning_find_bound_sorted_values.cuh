// muh/include/muh/tuning/tuning_find_bound_sorted_values.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_find_bound_sorted_values.cuh
// CCCL: threads=256, items=8, binary search per thread
//
// vllm relevance: block_table index lookup in paged attention
// SMEM risk: minimal (binary search, no tile)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::find_bound {

struct FindBoundPolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier haystack_load_modifier;
  CacheLoadModifier needles_load_modifier;
};

struct policy_selector {
  int haystack_type_size;
  int needle_type_size;

  constexpr FindBoundPolicy operator()(const hardware_capability& /*hw*/) const {
    // CCCL: fixed policy for all architectures
    return {256, 8, LOAD_LDG, LOAD_LDG};
  }
};

} // namespace muh::tuning::find_bound
