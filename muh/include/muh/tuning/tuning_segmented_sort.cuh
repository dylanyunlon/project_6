// muh/include/muh/tuning/tuning_segmented_sort.cuh — BI-V100 segmented_sort tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_sort.cuh
// vllm impact: per-segment sorting
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_sort {

struct SegmentedSortPolicy {
  int threads_per_block;
  int items_per_thread;
  int radix_bits;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 11;
  static constexpr int radix_bits = 6;
};

struct policy_selector {
  int key_size;
  int value_size;

  constexpr SegmentedSortPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, bi100_default::radix_bits};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, bi100_default::radix_bits};
  }
};

} // namespace muh::tuning::segmented_sort
