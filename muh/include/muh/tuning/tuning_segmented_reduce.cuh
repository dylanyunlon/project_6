// muh/include/muh/tuning/tuning_segmented_reduce.cuh — BI-V100 segmented_reduce tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_reduce.cuh
// vllm impact: per-segment reduce in batched attention
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_reduce {

struct SegmentedReducePolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 16;
  static constexpr int vec_size = 4;
};

struct policy_selector {
  int accum_size;
  int offset_size;

  constexpr SegmentedReducePolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, bi100_default::vec_size};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, bi100_default::vec_size};
  }
};

} // namespace muh::tuning::segmented_reduce
