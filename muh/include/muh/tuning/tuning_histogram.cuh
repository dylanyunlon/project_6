// muh/include/muh/tuning/tuning_histogram.cuh — BI-V100 histogram tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_histogram.cuh
// vllm impact: token frequency counting in sampling
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::histogram {

struct HistogramPolicy {
  int threads_per_block;
  int pixels_per_thread;
  int vec_size;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
};

struct bi100_default {
  static constexpr int threads = 768;
  static constexpr int items = 12;
  static constexpr int vec_size = 4;
  static constexpr int load_algo = BLOCK_LOAD_DIRECT;
  static constexpr int load_mod = LOAD_LDG;
};

struct policy_selector {
  int sample_size;
  int num_channels;

  constexpr HistogramPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, bi100_default::vec_size, BLOCK_LOAD_DIRECT, LOAD_LDG};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, bi100_default::vec_size, BLOCK_LOAD_DIRECT, LOAD_LDG};
  }
};

} // namespace muh::tuning::histogram
