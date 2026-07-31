// muh/include/muh/tuning/tuning_batched_topk.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_batched_topk.cuh
// CCCL: extends topk with batch-level parallelism
//
// vllm relevance: multi-sequence parallel decode top-k sampling
// SMEM risk: same as topk (handled by bits_per_pass)

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
    
    // Batches per block: limited by SMEM
    // Each batch needs: bits_per_pass buckets * sizeof(int) for histogram
    int buckets = 1 << base.bits_per_pass;
    int hist_smem = buckets * 4;  // sizeof(int)
    int max_batches = hw.max_shared_memory_per_block / hist_smem;
    if (max_batches < 1) max_batches = 1;
    if (max_batches > 32) max_batches = 32;  // cap for occupancy

    return {base, max_batches};
  }
};

} // namespace muh::tuning::batched_topk
