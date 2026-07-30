// muh/include/muh/muh.cuh — Top-level muh header
//
// Provides the complete tuning dispatch for Iluvatar BI-V100.
// Include this single header to get all tuning policies.
//
// Usage:
//   #include <muh/muh.cuh>
//
//   auto hw = muh::target_hw;  // BI-V100 by default
//   auto reduce_policy = muh::tuning::reduce::policy_selector{
//       .accum_t = muh::tuning::type_t::float32,
//       .operation_t = muh::tuning::op_kind_t::plus,
//       .offset_size = 4,
//       .accum_size = 4,
//   }(hw);
//
//   auto scan_policy = muh::tuning::scan::policy_selector{
//       .input_value_size = 4,
//       .accum_size = 4,
//       .offset_size = 4,
//       .input_type = muh::tuning::type_t::float32,
//       .accum_type = muh::tuning::type_t::float32,
//       .operation_t = muh::tuning::op_kind_t::plus,
//       .is_primitive_accum = true,
//   }(hw);

#pragma once

// Hardware descriptor
#include "muh/hardware.cuh"

// Shared types (compatible with CCCL)
#include "muh/tuning/common.cuh"

// Per-algorithm tuning (P0 = highest priority for competition)
#include "muh/tuning/tuning_reduce.cuh"      // P0: attention reduction
#include "muh/tuning/tuning_topk.cuh"        // P0: sampling top-k/top-p
#include "muh/tuning/tuning_scan.cuh"        // P0: prefix scan in paged attention

// P1
#include "muh/tuning/tuning_transform.cuh"   // P1: activation kernels
#include "muh/tuning/tuning_batch_memcpy.cuh" // P1: KV cache management

// P2
#include "muh/tuning/tuning_for.cuh"         // P2: RoPE position encoding

namespace muh {

/// Version info
constexpr int MUH_VERSION_MAJOR = 0;
constexpr int MUH_VERSION_MINOR = 1;
constexpr int MUH_VERSION_PATCH = 0;

/// Competition scoring formula
/// Token吞吐加权值 = Output TPS × 16.796 + Input TPS × 2.799 + Cache TPS × 0.56
struct scoring {
  static constexpr double output_weight = 16.796;
  static constexpr double input_weight  = 2.799;
  static constexpr double cache_weight  = 0.56;
  static constexpr double baseline_threshold = 8000.0;  // minimum to pass
  static constexpr double advanced_uplift    = 0.30;    // 30% for advanced prize
  static constexpr double special_uplift     = 0.50;    // 50% for special prize
};

} // namespace muh
