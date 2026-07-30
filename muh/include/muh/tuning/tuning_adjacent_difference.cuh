// muh/include/muh/tuning/tuning_adjacent_difference.cuh — BI-V100 adjacent_difference tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_adjacent_difference.cuh
// vllm impact: difference calculation (minor)
// Competition weight: minimal

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::adjacent_difference {

struct AdjacentDifferencePolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
};

struct bi100_default {
  static constexpr int threads = 128;
  static constexpr int items = 7;
  static constexpr int load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr int load_mod = LOAD_LDG;
};

struct policy_selector {
  int value_size;

  constexpr AdjacentDifferencePolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_LDG};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_LDG};
  }
};

} // namespace muh::tuning::adjacent_difference
