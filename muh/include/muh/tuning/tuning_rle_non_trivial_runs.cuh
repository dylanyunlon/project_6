// muh/include/muh/tuning/tuning_rle_non_trivial_runs.cuh — BI-V100
// Full port from CCCL (691 lines): SM100 (4) + SM90 (5) + SM80 (5) + int128 entries
// Dispatch: (length_size=4, key_size 1/2/4/8/16)
// Policy has extra field: store_with_time_slicing (always false in tuned entries)
// SMEM: tpb * ipt * key_size + offsets/counts
#pragma once
#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_non_trivial_runs {

struct RleNonTrivialRunsLookbackPolicy {
  int threads_per_block; int items_per_thread;
  BlockLoadAlgorithm load_algorithm; CacheLoadModifier load_modifier;
  bool store_with_time_slicing; BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};
enum class RleNonTrivialRunsAlgorithm { lookback };
struct RleNonTrivialRunsPolicy {
  RleNonTrivialRunsAlgorithm algorithm; RleNonTrivialRunsLookbackPolicy lookback;
};

struct policy_selector {
  int key_size;
  bool key_is_primitive;

  static constexpr LookbackDelayPolicy nd(int l2w) { return {LookbackDelayAlgorithm::no_delay, 0, l2w}; }
  static constexpr LookbackDelayPolicy nd(int l2w) {
    return {LookbackDelayAlgorithm::no_delay, 0, l2w};
  }

  constexpr RleNonTrivialRunsLookbackPolicy p(int tpb, int ipt, BlockLoadAlgorithm la,
      CacheLoadModifier lm, LookbackDelayPolicy d) const {
    return {tpb, ipt, la, lm, false, BLOCK_SCAN_WARP_SCANS, d};
  }

  constexpr RleNonTrivialRunsLookbackPolicy dispatch() const {
    if (!key_is_primitive) {
      // int128: SM90 entry
      if (key_size==16) return p(288, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                  nd(1150));
      // Default
      int ipt = 15 * 4 / key_size; if (ipt<1) ipt=1; if (ipt>15) ipt=15;
      return p(96, ipt, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(450));
    }

    // SM100 (delay scaled)
    if (key_size==1) return p(224, 20, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_CA,
                              nd(315));
    if (key_size==2) return p(224, 20, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                              nd(340));
    if (key_size==4) return p(224, 13, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                              nd(470));
    if (key_size==8) return p(256, 15, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                              nd(520));

    // SM90 fallback
    // (SM100 already covers 1/2/4/8, this handles edge cases)
    int ipt = 15 * 4 / key_size; if (ipt<1) ipt=1; if (ipt>15) ipt=15;
    return p(96, ipt, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, nd(450));
  }

  constexpr RleNonTrivialRunsPolicy operator()(const hardware_capability& hw) const {
    return {RleNonTrivialRunsAlgorithm::lookback, dispatch()};
  }
};

} // namespace muh::tuning::rle_non_trivial_runs
