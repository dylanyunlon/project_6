// muh/include/muh/tuning/tuning_batch_memcpy.cuh — BI-V100 batch memcpy tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_batch_memcpy.cuh
//
// vllm impact: KV cache block copy between GPU memory regions
// Competition weight: Cache TPS × 0.56
//
// CCCL structure: two-tier (SmallBuffer handled by single block,
// LargeBuffer by multi-block collaboration). Thresholds:
//   warp_level: 128 bytes
//   block_level: 8 KiB
// muh must replicate this structure, not flatten it.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::batch_memcpy {

/// Small buffer policy: single thread block handles many small buffers
struct SmallBufferPolicy {
  int threads_per_block;
  int buffers_per_thread;
  int bytes_per_thread;
  bool prefer_pow2_bits;
  int block_level_tile_size;
  int warp_level_threshold;
  int block_level_threshold;
  LookbackDelayPolicy buffer_lookback_delay;
  LookbackDelayPolicy block_lookback_delay;
};

/// Large buffer policy: multiple blocks collaborate on one large buffer
struct LargeBufferPolicy {
  int threads_per_block;
  int bytes_per_thread;
};

/// Full batch memcpy policy
struct BatchMemcpyPolicy {
  SmallBufferPolicy small_buffer;
  LargeBufferPolicy large_buffer;
};

// ============================================================
// BI-V100 tuning values — from CCCL SM70+ defaults
//
// CCCL policy_selector (all architectures):
//   small: 128 threads, 4 bufs/thread, 8 bytes/thread
//          prefer_pow2_bits = (cc < 7.0)
//          warp_threshold = 128, block_threshold = 8192
//          delays = default_delay_constructor_policy(true)
//   large: 256 threads, 32 bytes/thread
//
// For BI-V100: start with CCCL defaults.
// The delay policy is arch-sensitive (CCCL uses
// default_delay_constructor_policy which picks fixed_delay for
// primitive types). We use fixed_delay as starting point.
// ============================================================

struct policy_selector {

  constexpr BatchMemcpyPolicy operator()(const hardware_capability& hw) const {
    // BI-V100: assume >= SM70 equivalent (no prefer_pow2_bits)
    bool prefer_pow2 = false;

    LargeBufferPolicy large{256, 32};

    SmallBufferPolicy small{
      /* threads_per_block = */    128,
      /* buffers_per_thread = */   4,
      /* bytes_per_thread = */     8,
      /* prefer_pow2_bits = */     prefer_pow2,
      /* block_level_tile_size = */ large.threads_per_block * large.bytes_per_thread,
      /* warp_level_threshold = */  128,
      /* block_level_threshold = */ 8 * 1024,
      /* buffer_lookback_delay = */ {LookbackDelayAlgorithm::fixed_delay, 350, 450},
      /* block_lookback_delay = */  {LookbackDelayAlgorithm::fixed_delay, 350, 450},
    };

    return {small, large};
  }
};

} // namespace muh::tuning::batch_memcpy
