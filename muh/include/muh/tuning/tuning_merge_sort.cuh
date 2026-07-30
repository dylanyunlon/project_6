// muh/include/muh/tuning/tuning_merge_sort.cuh — BI-V100 merge_sort tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_merge_sort.cuh
// vllm impact: sorting in scheduler/sampler
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::merge_sort {

struct MergeSortPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 17;
  static constexpr int load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr int load_mod = LOAD_DEFAULT;
};

struct policy_selector {
  int key_size;

  constexpr MergeSortPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT};
  }
};

} // namespace muh::tuning::merge_sort
