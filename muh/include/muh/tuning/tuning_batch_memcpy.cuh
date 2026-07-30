// muh/include/muh/tuning/tuning_batch_memcpy.cuh — BI-V100 batch memcpy tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_batch_memcpy.cuh
//
// vllm impact: KV cache block copy between GPU memory regions
// Competition weight: Cache TPS × 0.56

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::batch_memcpy {

/// Batch memcpy policy
struct BatchMemcpyPolicy {
  int threads_per_block;
  LookbackDelayPolicy buffer_lookback_delay;
  LookbackDelayPolicy block_lookback_delay;
};

// ============================================================
// BI-V100 tuning values
//
// CCCL reference from tuning_batch_memcpy.cuh:
//   Default: threads=256
//   Buffer lookback delay and block lookback delay are separate —
//   they control the two decoupled lookback scans that coordinate
//   the batch copy across thread blocks.
//
//   For vllm: KV cache copies are typically large contiguous blocks
//   (head_dim × num_layers × sizeof(half)), so high throughput matters
//   more than latency.
// ============================================================

struct bi100_default {
  static constexpr int threads = 256;
  static constexpr LookbackDelayPolicy buffer_delay = {
    LookbackDelayAlgorithm::fixed_delay, 350, 450};
  static constexpr LookbackDelayPolicy block_delay = {
    LookbackDelayAlgorithm::fixed_delay, 350, 450};
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  constexpr BatchMemcpyPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      return {bi100_default::threads,
              bi100_default::buffer_delay,
              bi100_default::block_delay};
    }

    // Fallback
    return {256,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450},
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }
};

} // namespace muh::tuning::batch_memcpy
