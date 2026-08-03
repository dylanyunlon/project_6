// muh/include/muh/tuning/tuning_find_bound_sorted_values.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_find_bound_sorted_values.cuh
// CCCL source: 106 lines. Three CC tiers:
//   SM80+: {512, N4B(15, combined), LOAD_DEFAULT}
//   SM60+: {256, N4B(15, combined), LOAD_DEFAULT}
//   SM50:  {256, N4B(15, combined), LOAD_LDG}
//
// vllm relevance: binary search in sorted token ID arrays (vocabulary lookup,
// sorted sampling indices), lower/upper bound operations on sorted KV cache offsets.
//
// No SMEM usage (binary search is register-only), BI-V100 safe.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::find_bound_sorted_values {

struct FindBoundSortedValuesPolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier load_modifier;
};

constexpr int nominal_4b_items(int nominal, int combined_size) {
  int result = nominal * 4 / combined_size;
  return result > 0 ? result : 1;
}

// CCCL policy_selector: three tiers by CC
// BI-V100 uses SM80+ tier: {512, N4B(15, combined), LOAD_DEFAULT}

struct policy_selector {
  int range_type_size;
  int values_type_size;

  constexpr FindBoundSortedValuesPolicy operator()(const hardware_capability& /*hw*/) const {
    int combined = range_type_size + values_type_size;
    int items = nominal_4b_items(15, combined);
    // SM80+ policy
    return FindBoundSortedValuesPolicy{512, items, LOAD_DEFAULT};
  }
};

} // namespace muh::tuning::find_bound_sorted_values
