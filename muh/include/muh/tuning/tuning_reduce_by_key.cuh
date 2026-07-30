// muh/include/muh/tuning/tuning_reduce_by_key.cuh — BI-V100 reduce_by_key tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce_by_key.cuh
// vllm impact: grouped reduction in multi-head attention (reduce per head)
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::reduce_by_key {

struct ReduceByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  LookbackDelayPolicy lookback_delay;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 13;
  static constexpr int load_algo = BLOCK_LOAD_DIRECT;
  static constexpr int load_mod = LOAD_LDG;
};

struct policy_selector {
  int key_size;
  int accum_size;
  int offset_size;

  constexpr ReduceByKeyPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT, LOAD_LDG, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT, LOAD_LDG, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::reduce_by_key
