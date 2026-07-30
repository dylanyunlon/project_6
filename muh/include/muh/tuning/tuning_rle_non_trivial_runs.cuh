// muh/include/muh/tuning/tuning_rle_non_trivial_runs.cuh — BI-V100 rle_non_trivial_runs tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_rle_non_trivial_runs.cuh
// vllm impact: non-trivial run detection
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_non_trivial_runs {

struct RleNonTrivialRunsPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  LookbackDelayPolicy lookback_delay;
};

struct bi100_default {
  static constexpr int threads = 192;
  static constexpr int items = 20;
  static constexpr int load_algo = BLOCK_LOAD_DIRECT;
};

struct policy_selector {
  int key_size;

  constexpr RleNonTrivialRunsPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::rle_non_trivial_runs
