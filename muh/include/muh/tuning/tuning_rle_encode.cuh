// muh/include/muh/tuning/tuning_rle_encode.cuh — BI-V100
// Full port from CCCL (626 lines): SM100 (4) + SM90 (5) + SM80 (5) + int128
// Dispatch: (length_size=4, key_size 1/2/4/8/16)
#pragma once
#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::rle_encode {

struct RleLookbackPolicy {
  int threads_per_block; int items_per_thread;
  BlockLoadAlgorithm load_algorithm; CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm; LookbackDelayPolicy delay;
};
enum class RleAlgorithm { lookback };
struct RleEncodePolicy { RleAlgorithm algorithm; RleLookbackPolicy lookback; };

struct policy_selector {
  int key_size;
  bool key_is_primitive;

  static constexpr LookbackDelayPolicy nd(int l2w) { return {LookbackDelayAlgorithm::no_delay, 0, l2w}; }
  static constexpr LookbackDelayPolicy nd(int l2w) {
    return {LookbackDelayAlgorithm::no_delay, 0, l2w};
  }

  constexpr RleLookbackPolicy p(int tpb, int ipt, BlockLoadAlgorithm la,
      CacheLoadModifier lm, LookbackDelayPolicy d) const {
    return {tpb, ipt, la, lm, BLOCK_SCAN_WARP_SCANS, d};
  }

  constexpr RleLookbackPolicy dispatch() const {
    if (!key_is_primitive) {
      if (key_size==16) return p(128, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                  nd(930));
      int ipt = 6 * 8 / (key_size + 4); if (ipt<1) ipt=1; if (ipt>6) ipt=6;
      return p(128, ipt, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(450));
    }

    // SM100 (delay scaled)
    // ipt_14.tpb_256.trp_0.ld_1.ns_468.dcid_7.l2w_300
    if (key_size==1) return p(256, 14, BLOCK_LOAD_DIRECT, LOAD_CA,
                              nd(300));
    // ipt_14.tpb_224.trp_0.ld_0.ns_376.dcid_7.l2w_420
    if (key_size==2) return p(224, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                              nd(420));
    // ipt_14.tpb_256.trp_0.ld_1.ns_956.dcid_7.l2w_70
    if (key_size==4) return p(256, 14, BLOCK_LOAD_DIRECT, LOAD_CA,
                              nd(70));
    // ipt_9.tpb_224.trp_1.ld_0.ns_188.dcid_2.l2w_765
    if (key_size==8) return p(224, 9, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                              nd(765));

    int ipt = 6 * 8 / (key_size + 4); if (ipt<1) ipt=1; if (ipt>6) ipt=6;
    return p(128, ipt, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, nd(450));
  }

  constexpr RleEncodePolicy operator()(const hardware_capability& hw) const {
    return {RleAlgorithm::lookback, dispatch()};
  }
};

} // namespace muh::tuning::rle_encode
