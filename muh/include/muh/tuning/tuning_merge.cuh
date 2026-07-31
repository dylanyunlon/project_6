// muh/include/muh/tuning/tuning_merge.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_merge.cuh
// CCCL: threads=256/512, items=nominal_4B scaled, depends on bulk copy support
//
// vllm relevance: beam search candidate merging
// SMEM risk: tile = threads * items * (key_size + value_size), can overflow for large types

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::merge {

struct MergePolicy {
  int threads_per_block;
  int items_per_thread;
  CacheLoadModifier load_modifier;
  BlockStoreAlgorithm store_algorithm;
  bool use_bulk_copy;
};

// CCCL's nominal_4B_items_to_items
constexpr int nominal_4B_items(int nominal, int type_size) {
  int items = nominal * 4 / type_size;
  return items < 1 ? 1 : items;
}

struct policy_selector {
  int key_size;
  int value_size;  // 0 for keys-only
  bool can_bulk_copy;

  constexpr MergePolicy operator()(const hardware_capability& hw) const {
    int tune_size = key_size + value_size;

    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100) && can_bulk_copy) {
      // SM100-like: 512 threads, bulk copy
      int items = nominal_4B_items(11, tune_size);
      // SMEM check: 512 * items * tune_size <= 49152
      while (512 * items * tune_size > hw.max_shared_memory_per_block && items > 1)
        items--;
      return {512, items, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, true};
    }

    // Default: 256 threads, no bulk
    int items = nominal_4B_items(15, tune_size);
    while (256 * items * tune_size > hw.max_shared_memory_per_block && items > 1)
      items--;
    return {256, items, LOAD_DEFAULT, BLOCK_STORE_WARP_TRANSPOSE, false};
  }
};

} // namespace muh::tuning::merge
