================================================================================
CCCL vs muh 精确比对审计报告
================================================================================

### 1. scale_mem_bound 函数 parity check
------------------------------------------------------------
  float32 (CCCL SM100 reduce)         CCCL=( 16i, 512t,tile= 32768B) muh=( 16i, 512t,tile= 32768B) ✓
  float64 (CCCL SM100 reduce)         CCCL=(  8i, 640t,tile= 40960B) muh=(  8i, 640t,tile= 40960B) ✓
  accum8 (CCCL SM100 reduce)          CCCL=(  7i, 512t,tile= 28672B) muh=(  7i, 512t,tile= 28672B) ✓
  scan 4B (CCCL SM100 scan)           CCCL=( 22i, 384t,tile= 33792B) muh=( 22i, 384t,tile= 33792B) ✓
  scan 8B (CCCL SM100 scan)           CCCL=( 11i, 416t,tile= 36608B) muh=( 11i, 416t,tile= 36608B) ✓
  det float32 SM90                    CCCL=( 13i, 224t,tile= 11648B) muh=( 13i, 224t,tile= 11648B) ✓
  det float64 SM86                    CCCL=(  5i, 128t,tile=  5120B) muh=(  5i, 128t,tile=  5120B) ✓
  1-byte type                         CCCL=( 32i, 256t,tile=  8192B) muh=( 32i, 256t,tile=  8192B) ✓
  2-byte type                         CCCL=( 32i, 256t,tile= 16384B) muh=( 32i, 256t,tile= 16384B) ✓
  16-byte type (int128)               CCCL=(  4i, 256t,tile= 16384B) muh=(  4i, 256t,tile= 16384B) ✓
  SMEM cap test (should trigger)      CCCL=(  8i, 768t,tile= 49152B) muh=(  8i, 768t,tile= 49152B) ✓
  → scale_mem_bound: FULL PARITY ✓

### 2. reduce tuning: CCCL SM100值 → BI-V100 scale_mem_bound适配后
------------------------------------------------------------
  CCCL benchmarked on SM100 → muh should use scale_mem_bound for BI-V100
  Key: reduce loads to REGISTERS not SMEM → SMEM cap rarely triggers

  float32_plus_o4           @4B: scaled=(16i,  512t) tile= 32768B (66.7%)
  float32_plus_o4           @8B: scaled=( 8i,  512t) tile= 32768B (66.7%)
  float64_plus_o4           @4B: scaled=(16i,  640t) tile= 40960B (83.3%)
  float64_plus_o4           @8B: scaled=( 8i,  640t) tile= 40960B (83.3%)
  accum8_plus_o4            @4B: scaled=(15i,  512t) tile= 30720B (62.5%)
  accum8_plus_o4            @8B: scaled=( 7i,  512t) tile= 28672B (58.3%)
  accum8_plus_o8            @4B: scaled=(15i,  512t) tile= 30720B (62.5%)
  accum8_plus_o8            @8B: scaled=( 7i,  512t) tile= 28672B (58.3%)
  det_float32_sm90          @4B: scaled=(13i,  224t) tile= 11648B (23.7%)
  det_float32_sm90          @8B: scaled=( 6i,  224t) tile= 10752B (21.9%)
  det_float32_sm86          @4B: scaled=( 6i,  224t) tile=  5376B (10.9%)
  det_float32_sm86          @8B: scaled=( 3i,  224t) tile=  5376B (10.9%)
  det_float64_sm86          @4B: scaled=(11i,  128t) tile=  5632B (11.5%)
  det_float64_sm86          @8B: scaled=( 5i,  128t) tile=  5120B (10.4%)
  default_fallback          @4B: scaled=(16i,  256t) tile= 16384B (33.3%)
  default_fallback          @8B: scaled=( 8i,  256t) tile= 16384B (33.3%)

### 3. muh bi100 reduce当前值 vs CCCL参考
------------------------------------------------------------
  muh改用了更大的items (24 vs SM100的16)来补偿16 SMs
  这是对的——reduce加载到寄存器,SMEM不是瓶颈

  ★ float32 plus (paged_attention score reduction — 83% weight):
    CCCL SM100:  items=16, threads=512, vec=2
    muh BI-V100: items=24, threads=512, vec=2
    理由: 16 SMs vs 148 SMs, 每个CTA需要处理更多数据
    tile对比: SM100=512*16*4=32768B | BI-V100=512*24*4=49152B (exactly 48KB)
    → items=24 用满了SMEM → 合理但有风险,如果BlockReduce实际占SMEM则溢出
    → 但注释说reduce不用BlockLoad(loads to registers) → 安全

### 4. scan tuning: CCCL SM100 → BI-V100 SMEM约束
------------------------------------------------------------
  Scan DOES use BlockLoad staging in SMEM → tile_bytes ≤ 49152 is HARD

  lookback_1B_o4       @1B: tpb= 512 ipt=18 tile=  9216B ✓
  lookback_1B_o4       @2B: tpb= 512 ipt=18 tile= 18432B ✓
  lookback_1B_o4       @4B: tpb= 512 ipt=18 tile= 36864B ✓
  lookback_1B_o4       @8B: tpb= 512 ipt=18 tile= 73728B ✗ OVERFLOW → max_items=12
  lookback_2B_o4       @1B: tpb= 512 ipt=13 tile=  6656B ✓
  lookback_2B_o4       @2B: tpb= 512 ipt=13 tile= 13312B ✓
  lookback_2B_o4       @4B: tpb= 512 ipt=13 tile= 26624B ✓
  lookback_2B_o4       @8B: tpb= 512 ipt=13 tile= 53248B ✗ OVERFLOW → max_items=12
  lookback_4B_o4       @1B: tpb= 384 ipt=22 tile=  8448B ✓
  lookback_4B_o4       @2B: tpb= 384 ipt=22 tile= 16896B ✓
  lookback_4B_o4       @4B: tpb= 384 ipt=22 tile= 33792B ✓
  lookback_4B_o4       @8B: tpb= 384 ipt=22 tile= 67584B ✗ OVERFLOW → max_items=16
  lookback_8B_o4       @1B: tpb= 416 ipt=23 tile=  9568B ✓
  lookback_8B_o4       @2B: tpb= 416 ipt=23 tile= 19136B ✓
  lookback_8B_o4       @4B: tpb= 416 ipt=23 tile= 38272B ✓
  lookback_8B_o4       @8B: tpb= 416 ipt=23 tile= 76544B ✗ OVERFLOW → max_items=14
  lookback_1B_o8       @1B: tpb= 384 ipt=14 tile=  5376B ✓
  lookback_1B_o8       @2B: tpb= 384 ipt=14 tile= 10752B ✓
  lookback_1B_o8       @4B: tpb= 384 ipt=14 tile= 21504B ✓
  lookback_1B_o8       @8B: tpb= 384 ipt=14 tile= 43008B ✓
  lookback_4B_o8       @1B: tpb= 416 ipt=19 tile=  7904B ✓
  lookback_4B_o8       @2B: tpb= 416 ipt=19 tile= 15808B ✓
  lookback_4B_o8       @4B: tpb= 416 ipt=19 tile= 31616B ✓
  lookback_4B_o8       @8B: tpb= 416 ipt=19 tile= 63232B ✗ OVERFLOW → max_items=14
  lookback_8B_o8       @1B: tpb= 320 ipt=22 tile=  7040B ✓
  lookback_8B_o8       @2B: tpb= 320 ipt=22 tile= 14080B ✓
  lookback_8B_o8       @4B: tpb= 320 ipt=22 tile= 28160B ✓
  lookback_8B_o8       @8B: tpb= 320 ipt=22 tile= 56320B ✗ OVERFLOW → max_items=19

  关键发现:
  - scan lookback_4B_o4: items=22, threads=384 → tile@4B=33792 ✓ tile@8B=67584 ✗
  - scan lookback_8B_o4: items=23, threads=416 → tile@8B=76544 ✗
  - 这些值在SM100上是安全的(228KB SMEM),但在BI-V100(48KB)上必须降级
  - muh已经做了降级(用scale_mem_bound),但需要验证降级后的值是否正确

### 5. CCCL benchmark format解析
------------------------------------------------------------
  NVIDIA的benchmark注释格式:
  ipt_<items>.tpb_<threads>.ns_<delay>.dcid_<algo>.l2w_<latency>.trp_<transpose>.ld_<load>
  后跟4个浮点数: 在[2^16, 2^20, 2^24, 2^28]四个problem size下的speedup

  dcid映射:
    0 = no_delay
    1 = fixed_delay
    2 = exp_backoff
    3 = exp_backoff_jitter
    4 = exp_backoff_jitter_window
    5 = exp_backon_jitter_window
    6 = exp_backon_jitter
    7 = exp_backon

### 6. 竞赛关键路径优先级
------------------------------------------------------------
  Token吞吐加权值 = Output_TPS × 16.796 + Input_TPS × 2.799 + Cache_TPS × 0.56
  → Output_TPS权重83%, Input_TPS权重14%, Cache_TPS权重3%

  decode热路径 (Output TPS):
    1. paged_attention score reduction → reduce (DONE: muh tuned)
    2. softmax denominator prefix-sum → scan (DONE: muh tuned)
    3. top-k/top-p sampling → topk/radix_sort (DONE: muh tuned)
    4. RMSNorm/SiLU/RoPE element-wise → transform (DONE: muh tuned)

  prefill热路径 (Input TPS):
    5. flash_attention → scan + reduce
    6. MoE expert routing → select_if + reduce_by_key

  cache热路径 (Cache TPS):
    7. KV cache block copy → batch_memcpy (DONE: muh tuned)

### 7. 待验证的关键问题
------------------------------------------------------------
  1. reduce items=24: 虽然loads to registers, 但实际BlockReduce<WARP_REDUCTIONS>的SMEM用量需要确认
  2. scan delay参数: 0.5x/0.6x缩放是启发式, 需要BI-V100实测L2 write latency
  3. LOAD_LDG vs LOAD_DEFAULT: topk bench显示BI-V100上LOAD_DEFAULT更快, reduce/scan可能同理
  4. SM count=16 → wave efficiency: 所有tuning都需要重新算occupancy
  5. transform bytes_in_flight: 从18GB/s改为56GB/s后items需要相应增大
