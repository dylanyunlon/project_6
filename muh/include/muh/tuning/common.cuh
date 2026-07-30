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

/// Memory-bound scaling: given nominal params for 4-byte types,
/// scale items_per_thread inversely with actual type size
/// to keep shared memory footprint constant.
/// Directly mirrors cub::detail::MemBoundScaling.
struct scaled_params {
  int threads_per_block;
  int items_per_thread;
};

constexpr scaled_params scale_mem_bound(
    int nominal_threads, int nominal_4b_items, int type_size) {
  int items = (nominal_4b_items * 4) / type_size;
  if (items < 1) items = 1;
  if (items > nominal_4b_items) items = nominal_4b_items;
  return {nominal_threads, items};
}

} // namespace muh::tuning
