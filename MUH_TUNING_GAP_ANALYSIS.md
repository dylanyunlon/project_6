# muh Tuning Gap Analysis — CCCL vs BI-V100 适配
## 2026-08-07

### 方法论

直接读取 CCCL 源码（26 个 tuning_*.cuh），提取竞赛相关的 benchmark annotations，
对比 muh 已有的 BI-V100 struct 值。每个算法的优先级由竞赛评分公式决定：

```
Score = Output_TPS × 16.796 + Input_TPS × 2.799 + Cache_TPS × 0.56
```

Output TPS = 83%, Input TPS = 14%, Cache TPS = 3%

---

### P0: 直接影响竞赛评分的算法

#### 1. REDUCE (Output TPS 83%) — ★★★★★
- **竞赛路径**: paged_attention score reduction, float32, plus
- **CCCL SM100**: `ipt_16.tpb_512.ipv_2 → 1.061/1.000/1.065/1.167`
- **muh BI-V100**: `bi100_plus_float32_o4 {512, 24, 2}` — tile=12288 (1.5× SM100)
- **状态**: ✅ 完成 (62% 行覆盖)
- **待定**: SM=16 items 适配 (P0 BUG)、LOAD_LDG vs LOAD_DEFAULT benchmark

#### 2. SCAN (Output TPS 83%) — ★★★★☆
- **竞赛路径**: softmax denominator prefix sum, float32, plus
- **CCCL SM100**: `ipt_22.tpb_384.ns_1904.dcid_6.l2w_830 → 1.148/0.997/1.140/1.463`
- **muh BI-V100**: `bi100_lookback_4B_o4 {384, 22}` — 与 SM100 同 tile
- **状态**: ✅ 核心完成 (39% 行覆盖，lookback + SM90 fallback)
- **待定**: Lookback delay 参数需实测校准、8B structs 99% SMEM 需验证

#### 3. TRANSFORM (Input TPS 14% + all activations) — ★★★★☆
- **竞赛路径**: SiLU/GeLU/RMSNorm, bfloat16
- **CCCL**: bytes_in_flight 是核心参数, B200=64KB, H100=48KB
- **muh BI-V100**: bytes_in_flight=64KB (confirmed by babelstream bench)
- **状态**: ✅ 核心完成
- **待定**: Vectorized vs prefetch algorithm 选择需实测

---

### P1: 间接影响性能的算法

#### 4. TOPK (sampling, Output TPS) — ★★★☆☆
- **竞赛路径**: logit sampling, float32 keys
- **CCCL**: bits_per_pass, thread count, BLOCK_SCAN_WARP_SCANS
- **muh BI-V100**: 有 inline tuning (threads=512, bits_per_pass=11)
- **状态**: ✅ 基本完成
- **待定**: Onesweep vs multi-sweep 选择

#### 5. SELECT_IF (MoE routing) — ★★☆☆☆
- **竞赛路径**: expert selection, float32, not_flagged, no_rejects, offset_4
- **CCCL SM80**: `{threads=256, items=18, WARP_TRANSPOSE, no_delay=1130}`
- **muh BI-V100**: 零 bi100 structs, 用 get_sm100_adapted() inline 计算
- **状态**: ⚠️ 只需 1/77 个 specialization, 但完全缺失
- **待定**: 需添加 bi100_select_float32_nf_nr_o4 struct

#### 6. RADIX_SORT (topk helper) — ★★☆☆☆
- **竞赛路径**: float32 key sort for sampling
- **CCCL**: 2381 行, onesweep + histogram, SM100 有复杂分支
- **muh BI-V100**: 222 行 (9% 覆盖)
- **状态**: ⚠️ 需要 onesweep 路径
- **待定**: bits_per_pass 和 histogram SMEM

---

### P2: 理论覆盖但不直接影响评分

| 算法 | CCCL 行数 | muh 行数 | 覆盖率 | 竞赛影响 |
|------|----------|---------|-------|---------|
| reduce_by_key | 1735 | 217 | 13% | 低 |
| scan_by_key | 2008 | 161 | 8% | 低 |
| unique_by_key | 1510 | 179 | 12% | 低 |
| three_way_partition | 708 | 67 | 9% | 低 |
| segmented_reduce | 471 | 112 | 24% | 低 |
| 其余 14 个 | ~4000 | ~800 | ~20% | 无 |

---

### 关键差距总结

1. **gen_patch.py 管道断裂** — 产出零 patch。已被 gen_config.py 替代。
2. **muh headers 20% 完成** — 但竞赛相关的 5 个算法 (reduce/scan/transform/topk/select_if) 核心参数已就位。
3. **缺 benchmark 验证** — 所有 BI-V100 speedup 标 TBD，需要在 Phanthy Cloud 上跑。
4. **Python layer 是真正的注入点** — 已在 triton_flash_attention.py 添加 8 个 BI-V100 configs, prefix_prefill.py 修 BLOCK=64, _custom_ops.py 修 SMEM=48KB。gen_config.py 又发现 19 个新候选 configs。
