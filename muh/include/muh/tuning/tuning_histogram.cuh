// muh/include/muh/tuning/tuning_histogram.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_histogram.cuh
// CCCL SM100: 4 specializations by num_channels × num_active_channels
//   Some use threads=1024 → SMEM overflow on BI-V100 for histogram bins > 48KB
//
// vllm relevance: token frequency histogram for repetition_penalty
// SMEM risk: HIGH. Histogram SMEM = num_bins * sizeof(counter_t), NOT threads*items.
//   256 bins * 4B = 1024B (safe). But 65536 bins → 256KB (overflow).
//   BI-V100 limit: 49152 / 4 = 12288 bins max.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::histogram {

struct HistogramPolicy {
  int threads_per_block;
  int items_per_thread;
  int privatized_bins_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  bool is_work_stealing;
};

struct policy_selector {
  int num_channels;
  int num_active_channels;
  int max_bins;

  constexpr HistogramPolicy operator()(const hardware_capability& hw) const {
    // SMEM for privatized histogram: privatized_bins * sizeof(int) * threads/warp_size
    // Must fit in 48KB
    int max_privatized = hw.max_shared_memory_per_block / (4 * 8); // 8 = threads/warp conservative
    int priv_bins = max_bins < max_privatized ? max_bins : max_privatized;
    if (priv_bins < 1) priv_bins = 1;

    if (num_channels == 1) {
      return {384, 12, priv_bins, BLOCK_LOAD_DIRECT, LOAD_LDG, false};
    }
    // multi-channel: fewer threads to leave SMEM for bins
    return {256, 8, priv_bins, BLOCK_LOAD_DIRECT, LOAD_LDG, false};
  }
};

} // namespace muh::tuning::histogram
