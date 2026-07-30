// muh/include/muh/tuning/tuning_transform.cuh — BI-V100 transform tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_transform.cuh
//
// vllm impact: Activation functions (SiLU, GELU), RMSNorm, residual add
// Competition weight: Output TPS × 16.796

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::transform {

/// Bulk transform policy (for contiguous input/output)
struct BulkPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;          // vectorization width
};

/// No-input policy (for fill/memset operations)
struct FillPolicy {
  int threads_per_block;
  int items_per_thread_no_input;
};

/// Full transform policy
struct TransformPolicy {
  BulkPolicy bulk;
  FillPolicy fill;
};

// ============================================================
// BI-V100 tuning values
//
// CCCL reference from tuning_transform.cuh:
//   Default bulk: threads=256, items=auto, vec_size=auto
//   The transform kernel is bandwidth-bound for large elementwise ops.
//   Key insight: vec_size should match the hardware's natural vector width.
//   For NVIDIA: vec_size is typically 4 (128-bit loads).
//   For BI-V100: TBD, likely also 4 or 8.
//
//   items_per_thread is computed as:
//     items_for_vec = ceil(vector_bytes / min_elem_size)
//     items_for_latency = (latency * bandwidth) / (threads * elem_size)
//     items = max(items_for_vec, items_for_latency)
// ============================================================

struct bi100_bulk_default {
  static constexpr int threads  = 256;
  static constexpr int items    = 8;    // conservative starting point
  static constexpr int vec_size = 4;    // 128-bit vector loads
};

struct bi100_fill_default {
  static constexpr int threads = 256;
  static constexpr int items   = 2;
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  int min_elem_size;    // minimum element size across all inputs
  int max_elem_size;    // maximum element size
  int num_inputs;       // number of input iterators

  constexpr TransformPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      // For BI-V100: start with CCCL defaults, tune via benchmark
      int vec_size = bi100_bulk_default::vec_size;

      // Compute items_per_thread based on vectorization
      int items_for_vec = (vec_size * 4) / min_elem_size; // 4 = sizeof(int)
      if (items_for_vec < 1) items_for_vec = 1;

      // Ensure items is a multiple of vec_size
      int items = items_for_vec;
      if (items % vec_size != 0) {
        items = ((items / vec_size) + 1) * vec_size;
      }

      return {
        {bi100_bulk_default::threads, items, vec_size},
        {bi100_fill_default::threads, bi100_fill_default::items}
      };
    }

    // Fallback
    return {
      {bi100_bulk_default::threads, bi100_bulk_default::items, bi100_bulk_default::vec_size},
      {bi100_fill_default::threads, bi100_fill_default::items}
    };
  }
};

} // namespace muh::tuning::transform
