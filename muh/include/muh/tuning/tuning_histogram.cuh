// muh/include/muh/tuning/tuning_histogram.cuh — BI-V100
// Full port from CCCL: SM100 (2 entries) + SM90 (2 entries) + default
// Dispatch on: (num_channels, num_active_channels, counter_size, sample_size, is_even)
// Histogram SMEM: num_bins * counter_size per privatized copy
#pragma once
#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::histogram {

enum class HistogramMemoryPreference { SMEM, GMEM };

struct HistogramPolicy {
  int threads_per_block;
  int items_per_thread;
  int privatized_smem_bins;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  bool rle_compress;
  HistogramMemoryPreference memory_preference;
  bool work_stealing;
  int max_smem_bins; // 0 = unlimited
};

struct policy_selector {
  bool sample_is_primitive;
  int sample_size;       // sizeof(SampleT)
  int counter_size;      // sizeof(CounterT), typically 4
  int sample_size_bytes;
  int num_channels;
  int num_active_channels;
  bool is_even;

  constexpr int t_scale(int nominal_items) const {
    int sample_scale = (sample_size_bytes + 3) / 4;
    int result = nominal_items / num_active_channels / sample_scale;
    return result > 0 ? result : 1;
  }

  constexpr HistogramPolicy operator()(const hardware_capability& hw) const {
    // === SM100 tunings (BI-V100 adapted) ===
    if (num_channels == 1 && num_active_channels == 1 &&
        counter_size == 4 && sample_is_primitive && sample_size == 1) {
      if (is_even) {
        // ipt_12.tpb_928.rle_0.ws_0.mem_1.ld_2.laid_0.vec_2
        // BI-V100: tpb=928 may exceed 16 SM occupancy, but keep for throughput
        // SMEM: 928*12*1 + 2048*4 = 19264 → safe
        return {928, 12, 1<<2, BLOCK_LOAD_DIRECT, LOAD_CA,
                false, HistogramMemoryPreference::SMEM, false, 2048};
      } else {
        // ipt_12.tpb_448.rle_0.ws_0.mem_1.ld_1.laid_0.vec_2
        return {448, 12, 1<<2, BLOCK_LOAD_DIRECT, LOAD_LDG,
                false, HistogramMemoryPreference::SMEM, false, 2048};
      }
    }

    // === SM90 tunings ===
    if (num_channels == 1 && num_active_channels == 1 &&
        counter_size == 4 && sample_is_primitive) {
      if (sample_size == 1) {
        return {768, 12, 1<<2, BLOCK_LOAD_DIRECT, LOAD_LDG,
                false, HistogramMemoryPreference::SMEM, false, 2048};
      }
      if (sample_size == 2) {
        return {960, 10, 1<<2, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                true, HistogramMemoryPreference::SMEM, false, 2048};
      }
    }

    // === Default (SM50+) ===
    return {384, t_scale(16), 4, BLOCK_LOAD_DIRECT, LOAD_LDG,
            true, HistogramMemoryPreference::SMEM, false, 0};
  }
};

} // namespace muh::tuning::histogram
