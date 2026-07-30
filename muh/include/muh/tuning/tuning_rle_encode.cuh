// muh/include/muh/tuning/tuning_rle_encode.cuh — BI-V100 rle_encode tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_rle_encode.cuh
// vllm impact: run-length encoding in sparse attention
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_encode {

struct RleEncodePolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 14;
  static constexpr int load_algo = BLOCK_LOAD_DIRECT;
};

struct policy_selector {
  int key_size;

  constexpr RleEncodePolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT};
  }
};

} // namespace muh::tuning::rle_encode
