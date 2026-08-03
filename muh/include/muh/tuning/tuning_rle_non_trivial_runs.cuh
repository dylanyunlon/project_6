// muh/include/muh/tuning/tuning_rle_non_trivial_runs.cuh — BI-V100
//
// Mirrors: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_rle_non_trivial_runs.cuh
// CCCL source: 691 lines. Complete SM80/SM90/SM100 tuning tables ported.
//
// vllm relevance: identifies non-trivial segments (length>1) in attention masks.
// Works with rle_encode for sparse attention pattern detection.

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_non_trivial_runs {

struct RleNonTrivialRunsLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  bool store_with_time_slicing;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy lookback_delay;
};

enum class RleNonTrivialRunsAlgorithm { lookback };

struct RleNonTrivialRunsPolicy {
  RleNonTrivialRunsAlgorithm algorithm;
  RleNonTrivialRunsLookbackPolicy lookback;
};

// ============================================================================
// SM80 tuning (CCCL lines 140-190)
// key=1B: {192, 20, DIRECT, no_delay(630)}
// key=2B: {192, 20, WARP_TRANSPOSE, no_delay(1015)}
// key=4B: {224, 15, WARP_TRANSPOSE, no_delay(915)}
// key=8B: {256, 13, WARP_TRANSPOSE, no_delay(1065)}
// key=16B:{192, 13, WARP_TRANSPOSE, no_delay(1050)}
//
// SM90 tuning (CCCL lines 200-252)
// key=1B: {256, 18, DIRECT, no_delay(385)}
// key=2B: {224, 20, DIRECT, no_delay(675)}
// key=4B: {256, 18, DIRECT, no_delay(695)}
// key=8B: {224, 14, WARP_TRANSPOSE, no_delay(840)}
// key=16B:{288,  9, WARP_TRANSPOSE, fixed_delay(484, 1150)}
//
// SM100 tuning (CCCL lines 260-330) with benchmark annotations:
// key=1B: {224, 20, WARP_TRANSPOSE, LOAD_CA, exponential_backoff(64, 315)}
//   ipt_20.tpb_224.trp_1.ts_0.ld_1.ns_64.dcid_2.l2w_315 1.119878 1.003690 1.130067 1.338983
// key=2B: {224, 20, WARP_TRANSPOSE, LOAD_DEFAULT, exponential_backon(116, 340)}
//   ipt_20.tpb_224.trp_1.ts_0.ld_0.ns_116.dcid_7.l2w_340 1.146528 1.072769 1.152390 1.333333
// key=4B: {224, 13, DIRECT, LOAD_DEFAULT, exponential_backoff(252, 470)}
//   ipt_13.tpb_224.trp_0.ts_0.ld_0.ns_252.dcid_2.l2w_470 1.113202 1.003690 1.133114 1.349296
// key=8B: {256, 15, WARP_TRANSPOSE, LOAD_DEFAULT, exponential_backoff(28, 520)}
//   ipt_15.tpb_256.trp_1.ts_0.ld_0.ns_28.dcid_2.l2w_520 1.114944 1.033189 1.122360 1.252083
// key=8B(double): falls back to SM90 {224, 14, WARP_TRANSPOSE}
// ============================================================================

struct policy_selector {
  int length_size;
  int key_size;
  type_t key_type;

  constexpr auto make_default_policy(CacheLoadModifier load_mod) const
    -> RleNonTrivialRunsLookbackPolicy
  {
    int items = 15 * 4 / key_size;
    if (items < 1) items = 1;
    if (items > 15) items = 15;
    return {96, items, BLOCK_LOAD_DIRECT, load_mod, true, BLOCK_SCAN_WARP_SCANS,
            default_lookback_delay(key_size)};
  }

  constexpr auto get_lookback_policy(const hardware_capability& hw) const
    -> RleNonTrivialRunsLookbackPolicy
  {
    auto smem_safe = [&](int threads, int items) -> bool {
      return threads * items * key_size <= (hw.max_shared_memory_per_block - 4096);
    };
    auto cap = [&](int threads, int items) -> int {
      while (!smem_safe(threads, items) && items > 1) items--;
      return items;
    };

    if (length_size == 4) {
      // SM100 tuning with BI-V100 delay scaling: ns*0.5, l2w*0.6
      if (key_size == 1) {
        int items = cap(224, 20);
        return {224, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_CA, false,
                BLOCK_SCAN_WARP_SCANS, {DelayAlgorithm::exponential_backoff, 32, 189}};
      }
      if (key_size == 2) {
        int items = cap(224, 20);
        return {224, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false,
                BLOCK_SCAN_WARP_SCANS, {DelayAlgorithm::exponential_backon, 58, 204}};
      }
      if (key_size == 4) {
        int items = cap(224, 13);
        return {224, items, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, false,
                BLOCK_SCAN_WARP_SCANS, {DelayAlgorithm::exponential_backoff, 126, 282}};
      }
      if (key_size == 8 && key_type != type_t::float64) {
        int items = cap(256, 15);
        return {256, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false,
                BLOCK_SCAN_WARP_SCANS, {DelayAlgorithm::exponential_backoff, 14, 312}};
      }
      if (key_size == 8) { // double: SM90 fallback
        int items = cap(224, 14);
        return {224, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false,
                BLOCK_SCAN_WARP_SCANS, {DelayAlgorithm::no_delay, 0, 504}};
      }
      if (key_size == 16) { // SM90 fallback
        int items = cap(288, 9);
        return {288, items, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, false,
                BLOCK_SCAN_WARP_SCANS, {DelayAlgorithm::fixed_delay, 242, 690}};
      }
    }

    return make_default_policy(LOAD_DEFAULT);
  }

  constexpr RleNonTrivialRunsPolicy operator()(const hardware_capability& hw) const {
    return {RleNonTrivialRunsAlgorithm::lookback, get_lookback_policy(hw)};
  }
};

} // namespace muh::tuning::rle_non_trivial_runs
