// muh/include/muh/tuning/common.cuh — Shared tuning types
//
// Compatible with cub/device/dispatch/tuning/common.cuh.
// Re-exports CCCL's enum types so muh tuning headers can reference them
// without pulling in the full CUB dependency tree.
// When compiling against actual CCCL, prefer their originals.

#pragma once

#include "muh/hardware.cuh"

// If CCCL is available, use their definitions
#if __has_include(<cub/config.cuh>)
#include <cub/block/block_load.cuh>
#include <cub/block/block_store.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/tuning/common.cuh>

namespace muh::tuning {
  using cub::BlockLoadAlgorithm;
  using cub::BlockStoreAlgorithm;
  using cub::BlockReduceAlgorithm;
  using cub::BlockScanAlgorithm;
  using cub::CacheLoadModifier;
  using cub::LookbackDelayAlgorithm;
  using cub::LookbackDelayPolicy;
  using cub::detail::type_t;
  using cub::detail::op_kind_t;
  // Re-export all enum values for convenience
  using cub::BLOCK_LOAD_DIRECT;
  using cub::BLOCK_LOAD_VECTORIZE;
  using cub::BLOCK_LOAD_WARP_TRANSPOSE;
  using cub::BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED;
  using cub::BLOCK_STORE_DIRECT;
  using cub::BLOCK_STORE_WARP_TRANSPOSE;
  using cub::BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED;
  using cub::BLOCK_REDUCE_RAKING;
  using cub::BLOCK_REDUCE_WARP_REDUCTIONS;
  using cub::BLOCK_SCAN_RAKING;
  using cub::BLOCK_SCAN_WARP_SCANS;
  using cub::LOAD_DEFAULT;
  using cub::LOAD_CA;
  using cub::LOAD_CG;
  using cub::LOAD_CS;
  using cub::LOAD_LDG;
}

#else
// Standalone definitions when CCCL is not available (for analysis/codegen)

namespace muh::tuning {

enum BlockLoadAlgorithm {
  BLOCK_LOAD_DIRECT,
  BLOCK_LOAD_VECTORIZE,
  BLOCK_LOAD_TRANSPOSE,
  BLOCK_LOAD_WARP_TRANSPOSE,
  BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED,
  BLOCK_LOAD_STRIPED,
};

enum BlockStoreAlgorithm {
  BLOCK_STORE_DIRECT,
  BLOCK_STORE_WARP_TRANSPOSE,
  BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED,
  BLOCK_STORE_STRIPED,
};

enum BlockReduceAlgorithm {
  BLOCK_REDUCE_RAKING,
  BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY,
  BLOCK_REDUCE_WARP_REDUCTIONS,
  BLOCK_REDUCE_WARP_REDUCTIONS_NONDETERMINISTIC,
};

enum BlockScanAlgorithm {
  BLOCK_SCAN_RAKING,
  BLOCK_SCAN_RAKING_MEMOIZE,
  BLOCK_SCAN_WARP_SCANS,
};

enum CacheLoadModifier {
  LOAD_DEFAULT,
  LOAD_CA,
  LOAD_CG,
  LOAD_CS,
  LOAD_CV,
  LOAD_LDG,
};

enum class LookbackDelayAlgorithm {
  no_delay,
  fixed_delay,
  exponential_backoff,
  exponential_backoff_jitter,
  exponential_backoff_jitter_window,
  exponential_backon_jitter_window,
  exponential_backon_jitter,
  exponential_backon,
};

struct LookbackDelayPolicy {
  LookbackDelayAlgorithm kind;
  unsigned int delay;
  unsigned int l2_write_latency;
};

enum class type_t {
  boolean, int8, int16, int32, int64, int128,
  uint8, uint16, uint32, uint64, uint128,
  float32, float64, other
};

enum class op_kind_t { plus, min, max, other };

} // namespace muh::tuning
#endif // __has_include(<cub/config.cuh>)

namespace muh::tuning {

/// Scaling result — matches CCCL's cub::detail::scaling_result field order:
///   { items_per_thread, threads_per_block }
/// Callers destructure as: auto [items, threads] = scale_mem_bound(...);
struct scaling_result {
  int items_per_thread;
  int threads_per_block;
};

/// Memory-bound scaling for non-4-byte types.
///
/// Mirrors cub::detail::scale_mem_bound() from cub/util_arch.cuh lines 153-161.
/// Returns {items_per_thread, threads_per_block} — items-first, matching CCCL.
///
/// Three operations:
///   1. Scale items inversely with type size (4B nominal → 8B halves, 1B doubles)
///   2. Clamp items to [1, nominal*2] (allows small types to increase items)
///   3. Cap threads by SMEM: min(nominal, round_up(48KB / (type_size * items), 32))
///
/// CCCL source reference:
///   items  = clamp(nominal_items * 4 / target_size, 1, nominal_items * 2)
///   threads = min(nominal_threads, round_up(max_smem / (target_size * items), 32))
///
/// The previous muh version had three bugs:
///   a) Return order was {threads, items} — should be {items, threads}
///   b) Upper clamp was nominal*1 — should be nominal*2
///   c) No SMEM cap on threads — CCCL caps threads to prevent SMEM overflow
constexpr scaling_result scale_mem_bound(
    int nominal_4B_threads, int nominal_4B_items, int target_type_size) {
  constexpr int max_smem = 48 * 1024; // 49152 bytes

  // Step 1+2: scale items, clamp to [1, nominal*2]
  int items = nominal_4B_items * 4 / target_type_size;
  if (items < 1) items = 1;
  if (items > nominal_4B_items * 2) items = nominal_4B_items * 2;

  // Step 3: cap threads by SMEM
  // round_up(x, 32) = ((x + 31) / 32) * 32
  int smem_per_thread = target_type_size * items;
  int max_threads_by_smem;
  if (smem_per_thread > 0) {
    int raw = max_smem / smem_per_thread;
    max_threads_by_smem = ((raw + 31) / 32) * 32;
  } else {
    max_threads_by_smem = nominal_4B_threads;
  }
  int threads = nominal_4B_threads < max_threads_by_smem
                  ? nominal_4B_threads : max_threads_by_smem;

  return {items, threads}; // items-first, matching CCCL scaling_result
}

} // namespace muh::tuning
