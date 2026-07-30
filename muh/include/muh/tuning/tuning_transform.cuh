// muh/include/muh/tuning/tuning_transform.cuh — BI-V100 transform tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_transform.cuh
//
// vllm impact: Activation functions (SiLU, GELU), RMSNorm, residual add
// Competition weight: Output TPS × 16.796
//
// CCCL structure: three transform policy types selected by iterator properties:
//   1. TransformVectorizedPolicy — contiguous + trivially_relocatable inputs
//   2. TransformAsyncCopyPolicy — SM90+ bulk copy (cp.async.bulk)
//   3. TransformPrefetchPolicy — fallback when stable_address needed
//
// For vllm: activations are contiguous dense fp16/bf16 tensors.
// → primarily hits the vectorized path.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::transform {

/// Vectorized transform: coalesced loads via vector types
struct VectorizedPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
};

/// Async copy transform: SM90+ bulk shared memory copy
struct AsyncCopyPolicy {
  int threads_per_block;
  int min_items_per_thread;
  int store_vec_size;
};

/// Prefetch transform: prefetch-based fallback
struct PrefetchPolicy {
  int threads_per_block;
};

/// Fill policy (no input, e.g. memset)
struct FillPolicy {
  int threads_per_block;
  int items_per_thread;
};

/// Full transform policy — selected at dispatch time based on iterator properties
struct TransformPolicy {
  VectorizedPolicy vectorized;
  AsyncCopyPolicy async_copy;
  PrefetchPolicy prefetch;
  FillPolicy fill;
};

// ============================================================
// BI-V100 tuning
//
// CCCL SM90+ vectorized path:
//   threads = 256 (SM90) or 128 (SM100)
//   items = computed from: max(items_for_vec, items_for_latency)
//     items_for_vec = ceil(vec_bytes / min_elem_size)
//     items_for_latency = (min_bytes_in_flight) / (threads * elem_size)
//   vec_size = auto (power-of-2 aligned to hardware vector width)
//
// CCCL SM90+ async_copy path:
//   threads = 256 (SM90) or 128 (SM100)
//   min_items_per_thread = computed from SMEM capacity
//   store_vec_size = auto_ublkcp_store_vec_size(output.value_type_size)
//
// CCCL prefetch:
//   threads = 256
//
// For BI-V100: start with SM100 values (128 threads for bulk/async,
// 256 for prefetch). vec_size = 4 (128-bit loads, standard for most GPUs).
// ============================================================

struct policy_selector {
  int min_elem_size;
  int max_elem_size;
  int num_inputs;
  bool all_contiguous;
  bool all_trivially_relocatable;
  bool requires_stable_address;

  constexpr TransformPolicy operator()(const hardware_capability& hw) const {
    // Compute vectorization params
    int vec_size = 4; // 128-bit, matches CCCL default

    // items_for_vec: how many items fit in one vector load
    int items_for_vec = (vec_size * 4) / min_elem_size; // 4 = sizeof(int)
    if (items_for_vec < 1) items_for_vec = 1;

    // items_for_latency: enough items to hide memory latency
    // CCCL cc_to_min_bytes_in_flight: B200=64KB, H100=48KB, A100=16KB, V100=12KB
    // BI-V100 per-SM BW = 900/50 = 18 GB/s ≈ A100 (2000/108 = 18.5 GB/s)
    // → Use 16KB (A100-level), not 48-64KB
    int bytes_in_flight = 16 * 1024;
    int items_for_latency = bytes_in_flight / (256 * min_elem_size);
    if (items_for_latency < 1) items_for_latency = 1;

    int bulk_items = items_for_vec > items_for_latency ? items_for_vec : items_for_latency;

    // Ensure items is a multiple of vec_size for aligned access
    if (bulk_items % vec_size != 0) {
      bulk_items = ((bulk_items / vec_size) + 1) * vec_size;
    }

    // BI-V100: 128 threads for bulk (SM100-like), 256 for prefetch
    int bulk_threads = hw.at_least(hardware_capability::vendor_t::iluvatar, 100) ? 128 : 256;

    return {
      // vectorized
      {bulk_threads, bulk_items, vec_size},
      // async_copy (BI-V100 may not support cp.async.bulk — conservative)
      {bulk_threads, 4, vec_size},
      // prefetch
      {256},
      // fill
      {256, 2},
    };
  }
};

} // namespace muh::tuning::transform
