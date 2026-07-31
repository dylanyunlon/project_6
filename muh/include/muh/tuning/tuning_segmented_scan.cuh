// muh/include/muh/tuning/tuning_segmented_scan.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_segmented_scan.cuh
// CCCL: threads=128, items=9 (nominal), scale_mem_bound for SMEM
//
// vllm relevance: per-sequence softmax denominator accumulation
// SMEM risk: scale_mem_bound handles it. tuple<AccumT, bool> → effective size = accum_size + 1 (padded to alignment)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::segmented_scan {

struct SegmentedScanPolicy {
  int threads_per_block;
  int items_per_thread;
  int max_segments_per_block;
  BlockLoadAlgorithm load_algorithm;
  BlockStoreAlgorithm store_algorithm;
  BlockScanAlgorithm scan_algorithm;
};

struct policy_selector {
  int accum_size;
  int accum_align;

  constexpr SegmentedScanPolicy operator()(const hardware_capability& /*hw*/) const {
    constexpr int nominal_threads = 128;
    constexpr int nominal_items = 9;
    constexpr int max_segments = 512;

    // Multi-segment agent uses tuple<AccumT, bool>: size = round_up(accum_size + 1, accum_align)
    int tuple_size = accum_size + accum_align;  // conservative estimate
    auto [items, threads] = scale_mem_bound(nominal_threads, nominal_items, tuple_size);

    return {threads, items, max_segments,
            BLOCK_LOAD_WARP_TRANSPOSE,
            BLOCK_STORE_WARP_TRANSPOSE,
            BLOCK_SCAN_WARP_SCANS};
  }
};

} // namespace muh::tuning::segmented_scan
