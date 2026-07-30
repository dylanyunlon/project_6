// muh/include/muh/tuning/tuning_unique_by_key.cuh — BI-V100 unique_by_key tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_unique_by_key.cuh
// vllm impact: deduplication in beam search
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::unique_by_key {

struct UniqueByKeyPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
};

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr int items = 12;
  static constexpr int load_algo = BLOCK_LOAD_DIRECT;
  static constexpr int load_mod = LOAD_DEFAULT;
};

struct policy_selector {
  int key_size;
  int value_size;

  constexpr UniqueByKeyPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT, LOAD_DEFAULT};
    }
    // Fallback
    return {bi100_default::threads, bi100_default::items, BLOCK_LOAD_DIRECT, LOAD_DEFAULT};
  }
};

} // namespace muh::tuning::unique_by_key
