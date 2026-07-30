// muh/include/muh/tuning/tuning_batched_topk.cuh — BI-V100 batched_topk tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_batched_topk.cuh
// vllm impact: batched top-k across sequences
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::batched_topk {

struct BatchedTopkPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 16;
  static constexpr int load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
};

struct policy_selector {
  int key_size;

  constexpr BatchedTopkPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE};
  }
};

} // namespace muh::tuning::batched_topk
