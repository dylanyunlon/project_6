// muh/include/muh/tuning/tuning_three_way_partition.cuh — BI-V100 three_way_partition tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_three_way_partition.cuh
// vllm impact: three-way split in scheduler
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::three_way_partition {

struct ThreeWayPartitionPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 12;
  static constexpr int load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
};

struct policy_selector {
  int input_size;

  constexpr ThreeWayPartitionPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE};
  }
};

} // namespace muh::tuning::three_way_partition
