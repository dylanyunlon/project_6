// muh/include/muh/tuning/tuning_scan_by_key.cuh — BI-V100 scan_by_key tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_scan_by_key.cuh
// vllm impact: attention mask prefix scan per sequence
// Competition weight: Input TPS × 2.799

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::scan_by_key {

struct ScanByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  BlockStoreAlgorithm store_algorithm;
  LookbackDelayPolicy lookback_delay;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 15;
  static constexpr int load_algo = BLOCK_LOAD_WARP_TRANSPOSE;
  static constexpr int store_algo = BLOCK_STORE_WARP_TRANSPOSE;
};

struct policy_selector {
  int key_size;
  int accum_size;

  constexpr ScanByKeyPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_WARP_TRANSPOSE, BLOCK_STORE_WARP_TRANSPOSE, {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::scan_by_key
