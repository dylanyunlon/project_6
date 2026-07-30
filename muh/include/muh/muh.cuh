// muh/include/muh/muh.cuh — Top-level muh header
//
// Only includes tuning headers for algorithms that are on vllm's hot path.
// Not every CCCL algorithm needs a muh tuning header — only the ones
// that actually execute during Qwen3.6 inference on BI-V100.
//
// vllm hot path analysis (by competition scoring weight):
//   Output TPS × 16.796 (83%): attention, sampling, activations, layernorm, RoPE
//   Input TPS × 2.799 (14%):   paged attention prefix scan
//   Cache TPS × 0.56 (3%):     KV cache block copy

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

// --- Algorithms that appear on vllm hot path ---

#include "muh/tuning/tuning_reduce.cuh"       // attention score reduction
#include "muh/tuning/tuning_topk.cuh"         // top-k/top-p sampling
#include "muh/tuning/tuning_scan.cuh"         // prefix scan in paged attention
#include "muh/tuning/tuning_transform.cuh"    // SiLU, GELU, RMSNorm
#include "muh/tuning/tuning_batch_memcpy.cuh" // KV cache block copy
#include "muh/tuning/tuning_for.cuh"          // RoPE position encoding

// That's it. 6 algorithms, not 26.
// histogram, rle_encode, merge_sort, select_if, etc. are CCCL algorithms
// that vllm does not call on the inference hot path.

namespace muh {
constexpr int MUH_VERSION_MAJOR = 0;
constexpr int MUH_VERSION_MINOR = 3;
constexpr int MUH_VERSION_PATCH = 0;
constexpr int MUH_ALGORITHM_COUNT = 6; // only the ones that matter

struct scoring {
  static constexpr double output_weight = 16.796;
  static constexpr double input_weight  = 2.799;
  static constexpr double cache_weight  = 0.56;
  static constexpr double baseline_threshold = 8000.0;
  static constexpr double advanced_uplift    = 0.30;
  static constexpr double special_uplift     = 0.50;
};
} // namespace muh
