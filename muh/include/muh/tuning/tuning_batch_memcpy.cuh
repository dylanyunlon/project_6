// muh/include/muh/tuning/tuning_batch_memcpy.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_batch_memcpy.cuh
// CCCL source: 227 lines. DeviceMemcpy (batched copy) tuning for KV cache block copy.
//
// CCCL has a two-tier policy:
//   small_buffer: {128 threads, 4 buffers/thread, 8 bytes/thread, prefer_pow2_bits,
//                  block_tile=8192, warp_threshold=128, block_threshold=8192,
//                  buffer_delay=default, block_delay=default}
//   large_buffer: {256 threads, 32 bytes/thread}
//
// vllm relevance: KV cache block copy in paged attention — when sequences are
// forked (beam search) or compacted, batches of small/medium KV cache blocks
// need to be copied efficiently. This is the [muh] batch_memcpy kernel.
//
// SMEM: small buffer kernel uses shared memory for:
//   - buffer metadata: buffers_per_tile * (src_ptr + dst_ptr + size) = tile * 24B
//   - byte staging: threads * bytes_per_thread = 128 * 8 = 1024B
//   - prefix scan: small (scan of buffer counts)
// Total ≈ 4KB for default config → well within 48KB.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::batch_memcpy {

// ============================================================================
// Policy structs — matching CCCL exactly
// ============================================================================

struct BatchedCopySmallBufferPolicy {
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

struct BatchedCopyLargeBufferPolicy {
  int threads_per_block;
  int bytes_per_thread;
};

struct BatchedCopyLookbackPolicy {
  BatchedCopySmallBufferPolicy small_buffer;
  BatchedCopyLargeBufferPolicy large_buffer;
};

enum class BatchedCopyAlgorithm { lookback };

struct BatchedCopyPolicy {
  BatchedCopyAlgorithm algorithm;
  BatchedCopyLookbackPolicy lookback;
};

// ============================================================================
// policy_selector — matches CCCL exactly
//
// CCCL uses the same policy for all CC, only prefer_pow2_bits differs:
//   - SM < 7.0: prefer_pow2_bits = true
//   - SM >= 7.0: prefer_pow2_bits = false
//
// BI-V100: equivalent to SM70+ → prefer_pow2_bits = false
// ============================================================================

struct policy_selector {
  constexpr BatchedCopyPolicy operator()(const hardware_capability& /*hw*/) const {
    auto large = BatchedCopyLargeBufferPolicy{256, 32};

    auto small = BatchedCopySmallBufferPolicy{
      128,   // threads_per_block
      4,     // buffers_per_thread
      8,     // bytes_per_thread
      false, // prefer_pow2_bits (SM70+ = false)
      large.threads_per_block * large.bytes_per_thread, // block_level_tile_size = 8192
      128,   // warp_level_threshold
      8 * 1024, // block_level_threshold = 8192
      default_lookback_delay(4),  // buffer offset delay (BufferOffsetT = int32)
      default_lookback_delay(4),  // block offset delay
    };

    return BatchedCopyPolicy{
      BatchedCopyAlgorithm::lookback,
      BatchedCopyLookbackPolicy{small, large}
    };
  }
};

} // namespace muh::tuning::batch_memcpy
