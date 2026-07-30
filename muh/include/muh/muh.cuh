// muh/include/muh/muh.cuh — Top-level muh header
//
// Complete BI-V100 tuning dispatch for all 26 CCCL algorithms.
// Include this single header to get all tuning policies.

#pragma once

// Hardware descriptor
#include "muh/hardware.cuh"

// Shared types (compatible with CCCL)
#include "muh/tuning/common.cuh"

// P0: Highest priority for competition (Output TPS × 16.796)
#include "muh/tuning/tuning_reduce.cuh"
#include "muh/tuning/tuning_topk.cuh"
#include "muh/tuning/tuning_scan.cuh"

// P1: High priority
#include "muh/tuning/tuning_transform.cuh"
#include "muh/tuning/tuning_transform_tile.cuh"
#include "muh/tuning/tuning_batch_memcpy.cuh"
#include "muh/tuning/tuning_radix_sort.cuh"
#include "muh/tuning/tuning_reduce_by_key.cuh"
#include "muh/tuning/tuning_scan_by_key.cuh"
#include "muh/tuning/tuning_select_if.cuh"
#include "muh/tuning/tuning_histogram.cuh"
#include "muh/tuning/tuning_merge.cuh"
#include "muh/tuning/tuning_merge_sort.cuh"
#include "muh/tuning/tuning_unique_by_key.cuh"
#include "muh/tuning/tuning_batched_topk.cuh"

// P2: Segmented/specialized
#include "muh/tuning/tuning_for.cuh"
#include "muh/tuning/tuning_segmented_reduce.cuh"
#include "muh/tuning/tuning_segmented_scan.cuh"
#include "muh/tuning/tuning_segmented_sort.cuh"
#include "muh/tuning/tuning_segmented_radix_sort.cuh"
#include "muh/tuning/tuning_three_way_partition.cuh"
#include "muh/tuning/tuning_rle_encode.cuh"
#include "muh/tuning/tuning_rle_non_trivial_runs.cuh"

// P3: Utility
#include "muh/tuning/tuning_adjacent_difference.cuh"
#include "muh/tuning/tuning_find.cuh"
#include "muh/tuning/tuning_find_bound_sorted_values.cuh"

namespace muh {
constexpr int MUH_VERSION_MAJOR = 0;
constexpr int MUH_VERSION_MINOR = 2;
constexpr int MUH_VERSION_PATCH = 0;
constexpr int MUH_ALGORITHM_COUNT = 26;

struct scoring {
  static constexpr double output_weight = 16.796;
  static constexpr double input_weight  = 2.799;
  static constexpr double cache_weight  = 0.56;
  static constexpr double baseline_threshold = 8000.0;
  static constexpr double advanced_uplift    = 0.30;
  static constexpr double special_uplift     = 0.50;
};
} // namespace muh
