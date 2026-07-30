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

/// Determinism modes (mirrors cuda::execution::determinism::__determinism_t)
enum class determinism_t {
  run_to_run,      // default: WARP_REDUCTIONS + LOAD_LDG
  gpu_to_gpu,      // deterministic: RAKING + LOAD_DEFAULT
  not_guaranteed,  // nondeterministic: WARP_REDUCTIONS_NONDETERMINISTIC
};

// ============================================================
// BI-V100 tuning values — initialized from CCCL SM100 benchmarks
// Status: PENDING BI-V100 BENCHMARK (values will change)
//
// CCCL SM100 benchmark results (from tuning_reduce.cuh):
//   float32+o4: ipt=16, tpb=512, ipv=2  → 1.061x geo, 1.167x max
//   float64+o4: ipt=16, tpb=640, ipv=1  → 1.018x geo, 1.057x max
//   int64+o4:   ipt=15, tpb=512, ipv=2  → 1.020x geo, 1.058x max
//   int64+o8:   ipt=15, tpb=512, ipv=1  → 1.019x geo, 1.057x max
//
// CCCL SM90 deterministic benchmark results:
//   float32:    ipt=13, tpb=224          → 1.107x geo, 1.317x max
//   float64:    ipt=11, tpb=128          → 1.232x geo, 1.582x max
// ============================================================

// --- Non-deterministic (default) tunings ---

struct bi100_float32_plus_o4 {
  // BI-V100: TBD — SM100 ref: ipt_16.tpb_512.ipv_2 1.061 1.000 1.065 1.167
  static constexpr int items              = 16;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

struct bi100_float64_plus_o4 {
  // BI-V100: TBD — SM100 ref: ipt_16.tpb_640.ipv_1 1.018 1.000 1.016 1.057
  static constexpr int items              = 16;
  static constexpr int threads            = 640;
  static constexpr int items_per_vec_load = 1;
};

struct bi100_int64_plus_o4 {
  // BI-V100: TBD — SM100 ref: ipt_15.tpb_512.ipv_2 1.020 1.000 1.018 1.058
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 2;
};

struct bi100_int64_plus_o8 {
  // BI-V100: TBD — SM100 ref: ipt_15.tpb_512.ipv_1 1.019 1.000 1.017 1.057
  static constexpr int items              = 15;
  static constexpr int threads            = 512;
  static constexpr int items_per_vec_load = 1;
};

// --- Deterministic tunings (BLOCK_REDUCE_RAKING) ---
// CCCL uses these when determinism == gpu_to_gpu
// vec_size is forced to 1 for deterministic reduction

struct bi100_det_float32 {
  // BI-V100: TBD — SM90 ref: ipt_13.tpb_224 1.107 1.010 1.097 1.317
  static constexpr int items   = 13;
  static constexpr int threads = 224;
};

struct bi100_det_float64 {
  // BI-V100: TBD — SM86 ref: ipt_11.tpb_128 1.232 1.002 1.245 1.582
  static constexpr int items   = 11;
  static constexpr int threads = 128;
};

/// Fallback for types without specific tuning
struct bi100_default {
  static constexpr int items              = 16;
  static constexpr int threads            = 256;
  static constexpr int items_per_vec_load = 4;
};

// ============================================================
// policy_selector
//
// Dispatch logic mirrors CCCL's exactly:
//   1. if determinism == gpu_to_gpu → get_deterministic_tuning()
//   2. else → get_two_phase_tuning()
//   3. if determinism == not_guaranteed → override reduce_algorithm
// ============================================================

struct policy_selector {
  type_t accum_t;
  op_kind_t operation_t;
  int offset_size;
  int accum_size;
  determinism_t determinism = determinism_t::run_to_run;

  /// Deterministic reduction: BLOCK_REDUCE_RAKING, vec_size=1, LOAD_DEFAULT
  /// Matches CCCL get_deterministic_tuning()
  constexpr ReducePolicy get_deterministic(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      if (accum_t == type_t::float32) {
        auto [t, i] = scale_mem_bound(bi100_det_float32::threads,
                                       bi100_det_float32::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
      if (accum_t == type_t::float64) {
        auto [t, i] = scale_mem_bound(bi100_det_float64::threads,
                                       bi100_det_float64::items, accum_size);
        ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
        return {rp, rp};
      }
    }

    // Fallback deterministic
    auto [t, i] = scale_mem_bound(256, 16, accum_size);
    ReducePassPolicy rp{t, i, 1, BLOCK_REDUCE_RAKING, LOAD_DEFAULT};
    return {rp, rp};
  }

  /// Standard two-phase reduction: BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG
  /// Matches CCCL get_two_phase_tuning()
  constexpr ReducePolicy get_two_phase(const hardware_capability& hw) const {
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

    // Fallback: SM60-equivalent conservative policy
    auto [t, i] = scale_mem_bound(
        bi100_default::threads, bi100_default::items, accum_size);
    ReducePassPolicy rp{t, i, bi100_default::items_per_vec_load,
                        BLOCK_REDUCE_WARP_REDUCTIONS, LOAD_LDG};
    return {rp, rp};
  }

  /// Main dispatch — mirrors CCCL's operator()(compute_capability)
  constexpr ReducePolicy operator()(const hardware_capability& hw) const {
    if (determinism == determinism_t::gpu_to_gpu) {
      return get_deterministic(hw);
    }

    auto policy = get_two_phase(hw);
    if (determinism == determinism_t::not_guaranteed) {
      policy.multi_tile.reduce_algorithm = BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC;
    }
    return policy;
  }
};

} // namespace muh::tuning::reduce
