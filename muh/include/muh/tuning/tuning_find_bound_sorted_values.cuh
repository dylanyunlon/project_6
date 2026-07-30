// muh/include/muh/tuning/tuning_find_bound_sorted_values.cuh — BI-V100 find_bound_sorted_values tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_find_bound_sorted_values.cuh
// vllm impact: binary search (minor)
// Competition weight: minimal

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::find_bound_sorted_values {

struct FindBoundSortedValuesPolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier load_modifier;
};

struct bi100_default {
  static constexpr int threads = 512;
  static constexpr int items = 15;
  static constexpr int load_mod = LOAD_DEFAULT;
};

struct policy_selector {
  int range_size;
  int values_size;

  constexpr FindBoundSortedValuesPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, LOAD_DEFAULT};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, LOAD_DEFAULT};
  }
};

} // namespace muh::tuning::find_bound_sorted_values
