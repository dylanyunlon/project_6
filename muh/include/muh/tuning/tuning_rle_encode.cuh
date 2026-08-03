// muh/include/muh/tuning/tuning_rle_encode.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_rle_encode.cuh
// CCCL source: 626 lines. Complete SM80/SM90/SM100 tuning tables ported.
//
// vllm relevance: attention mask sparse representation (RLE compression)
// Long context (100K tokens) causal mask has huge runs of 1s → RLE saves memory.
//
// SMEM: RLE uses agent_rle (BlockLoad + BlockScan), similar to reduce_by_key.
// tile = threads * items * key_size. BI-V100 limit 48KB.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_encode {

struct RleLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy lookback_delay;
};

enum class RleAlgorithm { lookback };

struct RleEncodePolicy {
  RleAlgorithm algorithm;
  RleLookbackPolicy lookback;
};

// ============================================================================
// SM80 tuning table — from CCCL lines 135-181
// ============================================================================
// key_size → {threads, items, load_algo, delay}
// length_size assumed 4 (int32), primitive types
//
// key=1B: {256, 14, DIRECT, no_delay(640)}
// key=2B: {256, 13, DIRECT, no_delay(900)}
// key=4B: {256, 13, DIRECT, no_delay(1080)}
// key=8B: {224,  9, WARP_TRANSPOSE, no_delay(1075)}
// key=16B:{128,  7, WARP_TRANSPOSE, no_delay(630)}

// ============================================================================
// SM90 tuning table — from CCCL lines 196-243
// ============================================================================
// key=1B: {256, 13, DIRECT, no_delay(620)}
// key=2B: {128, 22, DIRECT, no_delay(775)}
// key=4B: {192, 14, WARP_TRANSPOSE, fixed_delay(284, 480)}
// key=8B: {128, 19, WARP_TRANSPOSE, no_delay(515)}
// key=16B:{128, 11, WARP_TRANSPOSE, fixed_delay(428, 930)}

// ============================================================================
// SM100 tuning table — from CCCL lines 257-298, with benchmark annotations
// ============================================================================
// key=1B: {256, 14, DIRECT, LOAD_CA, exponential_backon(468, 300)}
//   ipt_14.tpb_256.trp_0.ld_1.ns_468.dcid_7.l2w_300 1.202228 1.126160 1.197973 1.307692
// key=2B: {224, 14, DIRECT, LOAD_DEFAULT, exponential_backon(376, 420)}
//   ipt_14.tpb_224.trp_0.ld_0.ns_376.dcid_7.l2w_420 1.123754 1.002404 1.113839 1.274882
// key=4B: {256, 14, DIRECT, LOAD_CA, exponential_backon(956, 70)}
//   ipt_14.tpb_256.trp_0.ld_1.ns_956.dcid_7.l2w_70 1.134395 1.071951 1.137008 1.169419
// key=8B: {224, 9, WARP_TRANSPOSE, LOAD_DEFAULT, exponential_backoff(188, 765)}
//   ipt_9.tpb_224.trp_1.ld_0.ns_188.dcid_2.l2w_765 1.100140 1.020069 1.116462 1.345506

struct policy_selector {
  int length_size;
  int key_size;
  type_t key_type;

  constexpr auto make_default_policy(CacheLoadModifier load_mod) const -> RleLookbackPolicy {
    int combined = length_size + key_size;
    int max_input = length_size > key_size ? length_size : key_size;
    int items = (max_input <= 8)
      ? 6
      : clamp(ceil_div(6 * 8, combined), 1, 6);
    return {128, items, BLOCK_LOAD_DIRECT, load_mod, BLOCK_SCAN_WARP_SCANS,
            default_lookback_delay(length_size)};
  }

  constexpr auto get_lookback_policy(const hardware_capability& hw) const -> RleLookbackPolicy {
    // BI-V100 SMEM check helper
    auto smem_safe = [&](int threads, int items) -> bool {
      int tile = threads * items * key_size;
      return tile <= (hw.max_shared_memory_per_block - 4096); // 4KB headroom
    };

    auto cap_items = [&](int threads, int items) -> int {
      while (!smem_safe(threads, items) && items > 1) items--;
      return items;
    };

    if (length_size == 4) {
      // ---- SM100 tuning with BI-V100 SMEM cap + delay scaling ----
      // delay_ns *= 0.5, l2w *= 0.6 for BI-V100 (16 SMs, 6MB L2 vs SM100 148 SMs, 50MB L2)
      if (key_size == 1) {
        int items = cap_items(256, 14);
        return {256, items, BLOCK_LOAD_DIRECT, LOAD_CA, BLOCK_SCAN_WARP_SCANS,
                {DelayAlgorithm::exponential_backon, 234, 180}};
      }
      if (key_size == 2) {
        int items = cap_items(224, 14);
        return {224, items, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                {DelayAlgorithm::exponential_backon, 188, 252}};
      }
      if (key_size == 4) {
        int items = cap_items(256, 14);
        return {256, items, BLOCK_LOAD_DIRECT, LOAD_CA, BLOCK_SCAN_WARP_SCANS,
                {DelayAlgorithm::exponential_backon, 478, 42}};
      }
      if (key_size == 8) {
        int items = cap_items(224, 9);
        return {224, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                {DelayAlgorithm::exponential_backoff, 94, 459}};
      }
      if (key_size == 16) {
        // SM90 fallback (SM100 not tuned for 16B keys)
        int items = cap_items(128, 11);
        return {128, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                {DelayAlgorithm::fixed_delay, 214, 558}};
      }
    }

    return make_default_policy(LOAD_DEFAULT);
  }

  constexpr RleEncodePolicy operator()(const hardware_capability& hw) const {
    return {RleAlgorithm::lookback, get_lookback_policy(hw)};
  }
};

} // namespace muh::tuning::rle_encode
