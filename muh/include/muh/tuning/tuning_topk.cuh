// muh/include/muh/tuning/tuning_topk.cuh — BI-V100 top-k tuning
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_topk.cuh
//
// vllm impact: Top-k / top-p sampling in decode stage
// Competition weight: Output TPS × 16.796 (highest priority, tied with reduce)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::topk {

/// Top-k policy (mirrors cub's topk_policy)
struct TopkPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  BlockScanAlgorithm scan_algorithm;
  int bits_per_pass;
};

// ============================================================
// BI-V100 tuning values
//
// CCCL reference from tuning_topk.cuh policy_selector:
//   The CCCL topk uses radix-based selection with configurable bits_per_pass.
//   Key insight: bits_per_pass trades off #passes vs per-pass work.
//   For small key_size (1-2B): 4-6 bits optimal
//   For large key_size (4-8B): 8-11 bits optimal
//
//   Default: threads=512, items=nominal_4b*4/key_size, bits=calc_bits_per_pass(key_size)
//   calc_bits_per_pass: key_size<=2 → 8, key_size<=4 → 9, key_size<=8 → 10, else → 11
// ============================================================

/// BI-V100 topk for 2-byte keys (float16/bfloat16 — most relevant for LLM logits)
struct bi100_topk_2B {
  // LLM decode: logits are typically fp16/bf16, so key_size=2
  // CCCL default for 2B: bits_per_pass=8, items=4*4/2=8
  static constexpr int threads        = 512;
  static constexpr int items           = 8;    // nominal_4b_items=4, scaled: 4*4/2=8
  static constexpr int bits_per_pass   = 8;
  static constexpr BlockLoadAlgorithm load_algo = BLOCK_LOAD_DIRECT;
  static constexpr BlockScanAlgorithm scan_algo = BLOCK_SCAN_WARP_SCANS;
};

/// BI-V100 topk for 4-byte keys (float32)
struct bi100_topk_4B {
  static constexpr int threads        = 512;
  static constexpr int items           = 4;    // nominal_4b_items=4, scaled: 4*4/4=4
  static constexpr int bits_per_pass   = 9;
  static constexpr BlockLoadAlgorithm load_algo = BLOCK_LOAD_DIRECT;
  static constexpr BlockScanAlgorithm scan_algo = BLOCK_SCAN_WARP_SCANS;
};

/// BI-V100 default fallback
struct bi100_topk_default {
  static constexpr int threads        = 256;
  static constexpr int items           = 4;
  static constexpr int bits_per_pass   = 8;
  static constexpr BlockLoadAlgorithm load_algo = BLOCK_LOAD_DIRECT;
  static constexpr BlockScanAlgorithm scan_algo = BLOCK_SCAN_WARP_SCANS;
};

// ============================================================
// policy_selector
// ============================================================

struct policy_selector {
  int key_size;

  static constexpr int calc_bits_per_pass(int ks) {
    if (ks <= 2) return 8;
    if (ks <= 4) return 9;
    if (ks <= 8) return 10;
    return 11;
  }

  constexpr TopkPolicy operator()(const hardware_capability& hw) const {
    if (hw.at_least(hardware_capability::vendor_t::iluvatar, 100)) {
      switch (key_size) {
        case 1:
        case 2:
          return {bi100_topk_2B::threads, bi100_topk_2B::items,
                  bi100_topk_2B::load_algo, bi100_topk_2B::scan_algo,
                  bi100_topk_2B::bits_per_pass};
        case 4:
          return {bi100_topk_4B::threads, bi100_topk_4B::items,
                  bi100_topk_4B::load_algo, bi100_topk_4B::scan_algo,
                  bi100_topk_4B::bits_per_pass};
      }
    }

    // Fallback
    int items = (4 * 4) / key_size;
    if (items < 1) items = 1;
    if (items > 4) items = 4;
    return {bi100_topk_default::threads, items,
            bi100_topk_default::load_algo, bi100_topk_default::scan_algo,
            calc_bits_per_pass(key_size)};
  }
};

} // namespace muh::tuning::topk
