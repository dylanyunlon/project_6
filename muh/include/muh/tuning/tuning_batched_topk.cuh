// muh/include/muh/tuning/tuning_batched_topk.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_batched_topk.cuh
// CCCL: extends topk with batch-level parallelism
//
// vllm relevance: multi-sequence parallel decode top-k sampling
// SMEM risk: histogram SMEM = (1 << bits) * sizeof(int) * max_batches
//   bits=11: 2048*4*batches — even at batches=1, keys_tile + 8192 can overflow
//   bits=8: 256*4*batches — much safer
//   Decision: use bits=8 (same as radix_sort) for all key sizes on BI-V100.
//
// SMEM layout: keys_tile (union with values_tile) + histogram per batch
//   total = threads * items * max(key_size, value_size) + (1<<bits) * 4 * batches
//   Must be ≤ 49152

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"
#include "muh/tuning/tuning_topk.cuh"

namespace muh::tuning::batched_topk {

struct BatchedTopkPolicy {
  topk::TopkPolicy per_batch;
  int max_batches_per_block;
};

struct policy_selector {
  int key_size;
  int max_k;

  constexpr BatchedTopkPolicy operator()(const hardware_capability& hw) const {
    auto base = topk::policy_selector{key_size}(hw);

    // Force bits=8 for BI-V100 (same decision as radix_sort)
    // bits=11 → 2048 buckets → histogram SMEM explodes with batching
    int bits = 8;
    int buckets = 1 << bits;  // 256
    int hist_smem_per_batch = buckets * 4;  // 1024 bytes per batch

    // keys_tile for base policy
    int keys_tile = base.threads_per_block * base.items_per_thread * key_size;

    // Max batches: (SMEM - keys_tile) / hist_per_batch
    int remaining_smem = hw.max_shared_memory_per_block - keys_tile;
    int max_batches = remaining_smem > 0 ? remaining_smem / hist_smem_per_batch : 1;
    if (max_batches < 1) max_batches = 1;
    if (max_batches > 32) max_batches = 32;  // cap for occupancy

    // Verify total SMEM
    int total_smem = keys_tile + hist_smem_per_batch * max_batches;
    while (total_smem > hw.max_shared_memory_per_block && max_batches > 1) {
      max_batches--;
      total_smem = keys_tile + hist_smem_per_batch * max_batches;
    }

    // Override base bits_per_pass to 8
    topk::TopkPolicy adjusted_base = base;
    adjusted_base.bits_per_pass = bits;

    return {adjusted_base, max_batches};
  }
};

} // namespace muh::tuning::batched_topk
