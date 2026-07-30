// muh/include/muh/tuning/tuning_select_if.cuh — BI-V100 select_if tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_select_if.cuh
// vllm impact: token filtering in speculative decoding
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::select_if {

struct SelectIfPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  LookbackDelayPolicy lookback_delay;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 18;
  static constexpr int load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
};

struct policy_selector {
  int input_size;

  constexpr SelectIfPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::select_if
