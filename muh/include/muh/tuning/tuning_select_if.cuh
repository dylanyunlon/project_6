// muh/include/muh/tuning/tuning_select_if.cuh — BI-V100
//
// Full port from: cccl_upstream/cub/cub/device/dispatch/tuning/tuning_select_if.cuh
// CCCL has SM80 (20 specializations) + SM90 (20) + SM100 (42 + may_alias + distinct_partitions)
// = 82 active benchmark-tuned entries.
//
// Strategy: BI-V100 starts from SM90 tunings (closest architecture match),
// applies SMEM cap (48KB) and SM-count compensation (16 SMs → larger tiles).
// SM100 tunings used where they don't overflow, with delay scaled (ns*0.5, l2w*0.6).
//
// Hardware constraints:
//   max_shared_memory_per_block = 49152 (48KB)
//   sm_count = 16
//   warp_size = 32
//   memory_bandwidth = 900 GB/s
//
// SMEM model for select_if:
//   select agent needs: input tile + selection flags + scan temp
//   Conservative: threads * items * input_size + scan overhead (~1KB)
//   BLOCK_LOAD_WARP_TRANSPOSE adds: threads * items * input_size (staging buffer)
//   With flags: + threads * items * 1 (bool flag per element)
//
// Delay scaling rationale:
//   SM100 L2 = 50MB, BI-V100 L2 = 6MB → 8.3x smaller
//   SM100 BW = 3.35 TB/s, BI-V100 BW = 900 GB/s → 3.7x slower
//   Empirical: ns * 0.5, l2w * 0.6 (conservative, pending benchmark)

#pragma once

#include "muh/hardware.cuh"
#include "muh/tuning/common.cuh"

namespace muh::tuning::select_if {

// ============================================================================
// Policy types — mirrors CCCL exactly
// ============================================================================

struct SelectLookbackPolicy {
  int threads_per_block;
  int items_per_thread;
  BlockLoadAlgorithm load_algorithm;
  CacheLoadModifier load_modifier;
  BlockScanAlgorithm scan_algorithm;
  LookbackDelayPolicy delay;
};

enum class SelectAlgorithm { lookback };

struct SelectPolicy {
  SelectAlgorithm algorithm;
  SelectLookbackPolicy lookback;
};

// ============================================================================
// policy_selector — full CCCL parity dispatch
// ============================================================================

struct policy_selector {
  int input_size;        // sizeof(InputT)
  int flag_size;         // 0 if no flags, sizeof(FlagT) otherwise
  int offset_size;       // sizeof(OffsetT), typically 4 or 8
  bool input_is_primitive;
  bool may_alias;        // SelectImpl::SelectPotentiallyInPlace
  bool distinct_partitions; // for partition API

  // Derived booleans
  constexpr bool has_flags() const { return flag_size > 0; }
  constexpr bool keep_rejects() const { return false; } // set externally via SelectImpl

  // SMEM safety check: returns true if tile fits in 48KB
  constexpr bool smem_safe(int threads, int items, int elem_sz, bool flagged,
                           bool warp_transpose) const {
    int tile = threads * items * elem_sz;
    if (warp_transpose) tile *= 2; // staging buffer
    if (flagged) tile += threads * items; // flag array
    tile += 1024; // scan temp overhead
    return tile <= 49152;
  }

  // Scale SM100 nominal_4b_items to actual items for this input size
  // Mirrors CCCL: Nominal4BItemsToItems
  constexpr int scale_items(int nominal_4b, int elem_sz) const {
    if (elem_sz <= 4) return nominal_4b;
    // For larger types, scale down proportionally
    int scaled = nominal_4b * 4 / elem_sz;
    return scaled > 0 ? scaled : 1;
  }

  // Make a policy with SMEM safety check, falls back to reducing items
  constexpr SelectLookbackPolicy make_safe_policy(
      int threads, int nominal_4b_items,
      BlockLoadAlgorithm load_alg, CacheLoadModifier load_mod,
      LookbackDelayPolicy delay) const {
    int items = scale_items(nominal_4b_items, input_size);
    bool wt = (load_alg == BLOCK_LOAD_WARP_TRANSPOSE);
    bool fl = has_flags();

    // SMEM overflow protection
    while (!smem_safe(threads, items, input_size, fl, wt) && items > 1) {
      items--;
    }
    // If still overflows, reduce threads
    while (!smem_safe(threads, items, input_size, fl, wt) && threads > 32) {
      threads -= 32;
    }

    return {threads, items, load_alg, load_mod, BLOCK_SCAN_WARP_SCANS, delay};
  }

  // Scale SM100 delay for BI-V100
  static constexpr LookbackDelayPolicy scale_delay(
      LookbackDelayAlgorithm algo, int ns, int l2w) {
    return {algo, static_cast<int>(ns * 0.5), static_cast<int>(l2w * 0.6)};
  }

  // ============================================================================
  // SM80 tuning table — 20 entries from CCCL
  // BI-V100 uses these directly (similar SMEM budget, pre-async era)
  // SM80 had 108 SMs, BI-V100 has 16 → we keep SM80 items (already conservative)
  // ============================================================================
  constexpr SelectLookbackPolicy get_sm80_tuning() const {
    bool fl = has_flags();
    // CCCL SM80 only tuned for offset_size=4, primitive types

    if (!fl && !may_alias) {
      // select::if (no flags, no alias)
      switch (input_size) {
        case 1: return {992, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 395}};
        case 2: return {576, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 870}};
        case 4: return {256, 18, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 1130}};
        case 8: return {192, 10, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 832, 1165}};
      }
    }
    if (fl && !may_alias) {
      // select::flagged
      switch (input_size) {
        case 1: return {224, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 735}};
        case 2: return {256, 20, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 1155}};
        case 4: return {320, 10, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 124, 1115}};
        case 8: return {384, 6, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 1130}};
      }
    }

    // Default fallback (matches CCCL DefaultPolicy)
    int nominal_items = 10;
    int items = (nominal_items * 4 / input_size);
    if (items < 1) items = 1;
    if (items > nominal_items) items = nominal_items;
    CacheLoadModifier mod = may_alias ? LOAD_CA : LOAD_LDG;
    return {128, items, BLOCK_LOAD_DIRECT, mod, BLOCK_SCAN_WARP_SCANS,
            {LookbackDelayAlgorithm::fixed_delay, 350, 450}};
  }

  // ============================================================================
  // SM90 tuning table — 20 entries from CCCL
  // SM90 had 128+ SMs, BI-V100 has 16 → items kept as-is (SMEM safe)
  // ============================================================================
  constexpr SelectLookbackPolicy get_sm90_tuning() const {
    bool fl = has_flags();

    if (!fl && !may_alias) {
      // select::if
      switch (input_size) {
        case 1: return {256, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 580}};
        case 2: return {256, 22, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 320, 605}};
        case 4: return {384, 17, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 76, 1150}};
        case 8: return {384, 11, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 380, 1140}};
      }
    }
    if (fl && !may_alias) {
      // select::flagged
      switch (input_size) {
        case 1: return {448, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::no_delay, 0, 715}};
        case 2: return {448, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 504, 765}};
        case 4: return {384, 15, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 415, 1125}};
        case 8: return {384, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT, BLOCK_SCAN_WARP_SCANS,
                        {LookbackDelayAlgorithm::fixed_delay, 360, 1170}};
      }
    }

    // Fall through to SM80
    return get_sm80_tuning();
  }

  // ============================================================================
  // SM100 → BI-V100 adapted tuning — CCCL's benchmark-tuned values
  // with SMEM overflow protection and delay scaling
  //
  // Each entry has the original CCCL benchmark annotation preserved.
  // Threads capped at safe values; items scaled via nominal_4b_items.
  // ============================================================================

  // Returns nullopt-equivalent (items=0) if no SM100 tuning exists
  constexpr SelectLookbackPolicy get_sm100_adapted() const {
    bool fl = has_flags();
    constexpr SelectLookbackPolicy NO_MATCH = {0, 0, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                                BLOCK_SCAN_WARP_SCANS,
                                                {LookbackDelayAlgorithm::fixed_delay, 0, 0}};

    // ---- select::if (no flags, no keep_rejects) ----
    if (!fl && !may_alias && offset_size == 4) {
      if (input_size == 1) {
        // trp_0.ld_0.ipt_22.tpb_384.ns_0.dcid_2.l2w_915
        return make_safe_policy(384, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backoff, 0, 915));
      }
      if (input_size == 4) {
        // trp_1.ld_0.ipt_15.tpb_384.ns_1508.dcid_5.l2w_585
        return make_safe_policy(384, 15, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1508, 585));
      }
    }
    if (!fl && may_alias && offset_size == 4) {
      if (input_size == 1) {
        // trp_1.ld_0.ipt_20.tpb_448.ns_596.dcid_6.l2w_295
        return make_safe_policy(448, 20, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 596, 295));
      }
    }

    // ---- select::flagged ----
    if (fl && !may_alias && offset_size == 4) {
      if (input_size == 1) {
        // trp_0.ld_0.ipt_20.tpb_896.ns_84.dcid_7.l2w_480
        // NOTE: tpb=896 may exceed SM=16 occupancy, keep for throughput
        return make_safe_policy(896, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 84, 480));
      }
      if (input_size == 2) {
        // trp_0.ld_0.ipt_22.tpb_256.ns_1292.dcid_5.l2w_750
        return make_safe_policy(256, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1292, 750));
      }
      if (input_size == 4) {
        // trp_0.ld_0.ipt_14.tpb_512.ns_844.dcid_6.l2w_675
        return make_safe_policy(512, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 844, 675));
      }
      if (input_size == 8) {
        // trp_0.ld_1.ipt_22.tpb_320.ns_660.dcid_7.l2w_1030
        return make_safe_policy(320, 22, BLOCK_LOAD_DIRECT, LOAD_CA,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 660, 1030));
      }
    }
    if (fl && may_alias && offset_size == 4) {
      if (input_size == 1) {
        // trp_0.ld_0.ipt_20.tpb_1024.ns_360.dcid_6.l2w_380
        return make_safe_policy(1024, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 360, 380));
      }
      if (input_size == 2) {
        // trp_1.ld_0.ipt_20.tpb_448.ns_136.dcid_2.l2w_760
        return make_safe_policy(448, 20, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backoff, 136, 760));
      }
      if (input_size == 4) {
        // trp_1.ld_0.ipt_14.tpb_384.ns_524.dcid_7.l2w_635
        return make_safe_policy(384, 14, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 524, 635));
      }
      if (input_size == 8) {
        // trp_1.ld_1.ipt_21.tpb_384.ns_1316.dcid_5.l2w_990
        return make_safe_policy(384, 21, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_CA,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1316, 990));
      }
    }

    // ---- partition::if (distinct_partitions=yes) ----
    if (!fl && !may_alias && distinct_partitions) {
      if (offset_size == 4 && input_size == 1) {
        // trp_0.ld_0.ipt_15.tpb_608.ns_676.dcid_7.l2w_500
        return make_safe_policy(608, 15, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 676, 500));
      }
      if (offset_size == 4 && input_size == 2) {
        // trp_0.ld_0.ipt_22.tpb_320.ns_1756.dcid_6.l2w_615
        return make_safe_policy(320, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 1756, 615));
      }
      if (offset_size == 4 && input_size == 4) {
        // trp_1.ld_0.ipt_19.tpb_320.ns_716.dcid_5.l2w_570
        return make_safe_policy(320, 19, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 716, 570));
      }
      if (offset_size == 8 && input_size == 1) {
        // trp_0.ld_0.ipt_22.tpb_576.ns_368.dcid_7.l2w_680
        return make_safe_policy(576, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 368, 680));
      }
      if (offset_size == 8 && input_size == 2) {
        // trp_1.ld_0.ipt_20.tpb_608.ns_516.dcid_7.l2w_635
        return make_safe_policy(608, 20, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 516, 635));
      }
      if (offset_size == 8 && input_size == 4) {
        // trp_1.ld_0.ipt_18.tpb_608.ns_1712.dcid_5.l2w_825
        return make_safe_policy(608, 18, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1712, 825));
      }
    }

    // ---- partition::if (distinct_partitions=no) ----
    if (!fl && !may_alias && !distinct_partitions) {
      if (offset_size == 4 && input_size == 1) {
        // trp_0.ld_0.ipt_22.tpb_224.ns_68.dcid_2.l2w_990
        return make_safe_policy(224, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backoff, 68, 990));
      }
      if (offset_size == 4 && input_size == 2) {
        // trp_0.ld_0.ipt_22.tpb_320.ns_560.dcid_5.l2w_640
        return make_safe_policy(320, 22, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 560, 640));
      }
      if (offset_size == 4 && input_size == 4) {
        // trp_1.ld_0.ipt_19.tpb_608.ns_724.dcid_5.l2w_970
        return make_safe_policy(608, 19, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 724, 970));
      }
      if (offset_size == 8 && input_size == 1) {
        // trp_0.ld_0.ipt_20.tpb_608.ns_1016.dcid_6.l2w_545
        return make_safe_policy(608, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 1016, 545));
      }
      if (offset_size == 8 && input_size == 2) {
        // trp_1.ld_0.ipt_22.tpb_288.ns_124.dcid_2.l2w_690
        return make_safe_policy(288, 22, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backoff, 124, 690));
      }
      if (offset_size == 8 && input_size == 4) {
        // trp_1.ld_0.ipt_19.tpb_608.ns_1884.dcid_6.l2w_950
        return make_safe_policy(608, 19, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 1884, 950));
      }
      if (offset_size == 8 && input_size == 8) {
        // trp_1.ld_0.ipt_23.tpb_416.ns_0.dcid_2.l2w_1200
        return make_safe_policy(416, 23, BLOCK_LOAD_WARP_TRANSPOSE, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backoff, 0, 1200));
      }
    }

    // ---- partition::flagged (distinct_partitions=yes) ----
    if (fl && !may_alias && distinct_partitions) {
      if (offset_size == 4 && input_size == 1) {
        // trp_0.ld_0.ipt_20.tpb_448.ns_964.dcid_7.l2w_385
        return make_safe_policy(448, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 964, 385));
      }
      if (offset_size == 4 && input_size == 8) {
        // trp_0.ld_0.ipt_21.tpb_384.ns_300.dcid_7.l2w_580
        return make_safe_policy(384, 21, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 300, 580));
      }
      if (offset_size == 8 && input_size == 1) {
        // trp_0.ld_1.ipt_20.tpb_448.ns_240.dcid_6.l2w_845
        return make_safe_policy(448, 20, BLOCK_LOAD_DIRECT, LOAD_CA,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter, 240, 845));
      }
      if (offset_size == 8 && input_size == 2) {
        // trp_0.ld_0.ipt_14.tpb_320.ns_1428.dcid_7.l2w_830
        return make_safe_policy(320, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 1428, 830));
      }
      if (offset_size == 8 && input_size == 4) {
        // trp_0.ld_0.ipt_14.tpb_640.ns_1204.dcid_5.l2w_635
        return make_safe_policy(640, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1204, 635));
      }
      if (offset_size == 8 && input_size == 8) {
        // trp_0.ld_0.ipt_19.tpb_384.ns_1016.dcid_7.l2w_875
        return make_safe_policy(384, 19, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 1016, 875));
      }
    }

    // ---- partition::flagged (distinct_partitions=no) ----
    if (fl && !may_alias && !distinct_partitions) {
      if (offset_size == 4 && input_size == 1) {
        // trp_0.ld_0.ipt_24.tpb_256.ns_2024.dcid_5.l2w_835
        return make_safe_policy(256, 24, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 2024, 835));
      }
      if (offset_size == 4 && input_size == 4) {
        // trp_0.ld_0.ipt_11.tpb_448.ns_476.dcid_7.l2w_665
        return make_safe_policy(448, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 476, 665));
      }
      if (offset_size == 4 && input_size == 8) {
        // trp_0.ld_0.ipt_20.tpb_384.ns_1420.dcid_5.l2w_525
        return make_safe_policy(384, 20, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1420, 525));
      }
      if (offset_size == 8 && input_size == 1) {
        // trp_0.ld_0.ipt_12.tpb_256.ns_0.dcid_5.l2w_850
        return make_safe_policy(256, 12, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 0, 850));
      }
      if (offset_size == 8 && input_size == 2) {
        // trp_0.ld_0.ipt_12.tpb_256.ns_1552.dcid_7.l2w_730
        return make_safe_policy(256, 12, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon, 1552, 730));
      }
      if (offset_size == 8 && input_size == 4) {
        // trp_0.ld_0.ipt_14.tpb_352.ns_1444.dcid_5.l2w_655
        return make_safe_policy(352, 14, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backon_jitter_window, 1444, 655));
      }
      if (offset_size == 8 && input_size == 8) {
        // trp_0.ld_0.ipt_11.tpb_512.ns_536.dcid_2.l2w_845
        return make_safe_policy(512, 11, BLOCK_LOAD_DIRECT, LOAD_DEFAULT,
                                scale_delay(LookbackDelayAlgorithm::exponential_backoff, 536, 845));
      }
    }

    return NO_MATCH;
  }

  // ============================================================================
  // Main dispatch — mirrors CCCL's cc-based fallback chain
  // BI-V100 → try SM100 adapted → SM90 → SM80 → default
  // ============================================================================
  constexpr SelectPolicy operator()(const hardware_capability& hw) const {
    // Try SM100 adapted tunings first (best benchmark data)
    auto sm100 = get_sm100_adapted();
    if (sm100.items_per_thread > 0) {
      return {SelectAlgorithm::lookback, sm100};
    }

    // Fall back to SM90 tunings (good general-purpose values)
    if (input_is_primitive) {
      return {SelectAlgorithm::lookback, get_sm90_tuning()};
    }

    // Final fallback to SM80
    return {SelectAlgorithm::lookback, get_sm80_tuning()};
  }
};

} // namespace muh::tuning::select_if
