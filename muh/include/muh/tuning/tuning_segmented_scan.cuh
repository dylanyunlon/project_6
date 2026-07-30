// muh/include/muh/tuning/tuning_segmented_scan.cuh — BI-V100 segmented_scan tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_scan.cuh
// vllm impact: per-segment prefix scan
// Competition weight: Input TPS × 2.799

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_scan {

struct SegmentedScanPolicy {
  int threads_per_block;
  int items_per_thread;
};

struct bi100_default {
  static constexpr int threads = 128;
  static constexpr int items = 9;
};

struct policy_selector {
  int accum_size;

  constexpr SegmentedScanPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items};
  }
};

} // namespace muh::tuning::segmented_scan
