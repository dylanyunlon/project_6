// muh/include/muh/tuning/tuning_reduce.cuh — BI-V100 reduce tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_reduce.cuh
// Pattern: policy_selector functor, dispatches on muh::hardware_capability
//
// vllm impact: Attention score reduction in multi-head attention
// Competition weight: Output TPS × 16.796 (highest priority)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::reduce {

/// Policy for a single reduction pass (mirrors cub::ReducePassPolicy)
struct ReducePassPolicy {
  int threads_per_block;
  int items_per_thread;
  int vec_size;
  BlockReduceAlgorithm reduce_algorithm;
  CacheLoadModifier load_modifier;
};

/// Full reduction policy (mirrors cub::ReducePolicy)
struct ReducePolicy {
  ReducePassPolicy multi_tile;
  ReducePassPolicy single_tile;
};

// ============================================================
// BI-V100 tuning values
// Status: PENDING BENCHMARK
//
// These are initialized from CCCL SM90/SM100 reference values.
// Must be replaced with actual BI-V100 benchmark results.
//
// CCCL SM100 reference (for comparison):
//   float32, offset_4: ipt=16, tpb=512, ipv=2 → 1.061x speedup
//   float64, offset_4: ipt=16, tpb=640, ipv=1 → 1.018x speedup
//   int64, offset_4:   ipt=15, tpb=512, ipv=2 → 1.020x speedup
//   int64, offset_8:   ipt=15, tpb=512, ipv=1 → 1.019x speedup
// ============================================================

/// BI-V100 tuning for sum(float32), 4-byte offset
struct bi100_float32_plus_o4 {
  // Benchmark annotation format: ipt_N.tpb_M.ipv_K <geo> <min> <avg> <max>
  // BI-V100: TBD — using SM100 reference as starting point
  static constexpr int items              = 16;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

/// BI-V100 tuning for sum(float64), 4-byte offset
struct bi100_float64_plus_o4 {
  // BI-V100: TBD
  static constexpr int items              = 16;
  static constexpr int threads            = 640;
  static constexpr int items_per_vec_load = 1;
};

/// BI-V100 tuning for sum(int64), 4-byte offset
struct bi100_int64_plus_o4 {
  // BI-V100: TBD
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

/// BI-V100 tuning for sum(int64), 8-byte offset
struct bi100_int64_plus_o8 {
  // BI-V100: TBD
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 1;
};

/// BI-V100 default fallback
struct bi100_default {
  static constexpr int items              = 16;
  static constexpr int threads            = 256;
  static constexpr int items_per_vec_load = 4;
};

// ============================================================
// policy_selector — the core dispatch functor
// Follows CCCL's exact pattern:
//   1. Accept hardware description
//   2. Match type/op/size to a tuning struct
//   3. Apply mem-bound scaling
//   4. Return ReducePolicy
// ============================================================

struct policy_selector {
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;

  /// Dispatch for BI-V100
  constexpr ReducePolicy operator()(const hardware_capability& hw) const {
    // Only tuned for sum currently (matching CCCL's approach)
    if (operation_t == op_kind_t::plus &&
        hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {

      if (accum_t == type_t::float32 && offset_size == 4 && accum_size == 4) {
        auto [t, i] = scale_mem_bound(
            bi100_float32_plus_o4::threads,
            bi100_float32_plus_o4::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_float32_plus_o4::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }

      if (accum_t == type_t::float64 && offset_size == 4 && accum_size == 8) {
        auto [t, i] = scale_mem_bound(
            bi100_float64_plus_o4::threads,
            bi100_float64_plus_o4::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_float64_plus_o4::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }

      if (offset_size == 4 && accum_size == 8) {
        auto [t, i] = scale_mem_bound(
            bi100_int64_plus_o4::threads,
            bi100_int64_plus_o4::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_int64_plus_o4::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }

      if (offset_size == 8 && accum_size == 8) {
        auto [t, i] = scale_mem_bound(
            bi100_int64_plus_o8::threads,
            bi100_int64_plus_o8::items, accum_size);
        ReducePassPolicy rp{t, i, bi100_int64_plus_o8::items_per_vec_load,
                            BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
        return {rp, rp};
      }
    }

    // Fallback: conservative policy
    auto [t, i] = scale_mem_bound(
        bi100_default::threads, bi100_default::items, accum_size);
    ReducePassPolicy rp{t, i, bi100_default::items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
    return {rp, rp};
  }
};

} // namespace muh::tuning::reduce
