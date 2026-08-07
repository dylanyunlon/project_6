# CCCL ↔ muh Tuning Header Gap Report

> **Generated**: 2026-08-06 (auto-analyzed from source code)
> **Source of truth**: `cccl_upstream/cub/cub/device/dispatch/tuning/tuning_*.cuh`
> **muh headers**: `muh/include/muh/tuning/tuning_*.cuh`

## Executive summary

- **26 algorithms** have both CCCL original and muh BI-V100 tuning headers.
- muh covers **19% of CCCL lines** (3568 / 18094).
- CCCL contains **294 benchmark annotations** across all algorithms. muh has **1 benchmarked algorithm** (scan, partial).
- The **#1 gap** is not code coverage — it's the absence of BI-V100 benchmark data in `ipt_N.tpb_M speedup` format.

## Per-algorithm coverage

| Algorithm | CCCL lines | muh lines | Coverage | CCCL bench pts | muh bi100 structs | muh benchmarked? |
|-----------|-----------|----------|----------|---------------|-------------------|-----------------|
| reduce | 478 | 297 | 62% | 6 | 14 | ✗ |
| scan | 1525 | 591 | 38% | 16 | 22 | ✓ (partial) |
| topk | 121 | 113 | 93% | 0 | 0 | ✗ |
| radix_sort | 2381 | 222 | 9% | 70 | 0 | ✗ |
| select_if | 2729 | 459 | 16% | 82 | 0 | ✗ |
| scan_by_key | 2008 | 145 | 7% | 30 | 0 | ✗ |
| reduce_by_key | 1735 | 171 | 9% | 32 | 0 | ✗ |
| unique_by_key | 1539 | 166 | 10% | 29 | 0 | ✗ |
| three_way_partition | 788 | 99 | 12% | 13 | 0 | ✗ |
| rle_non_trivial_runs | 691 | 68 | 9% | 8 | 0 | ✗ |
| segmented_sort | 640 | 189 | 29% | 0 | 0 | ✗ |
| rle_encode | 626 | 63 | 10% | 4 | 0 | ✗ |
| transform | 549 | 185 | 33% | 0 | 0 | ✗ |
| histogram | 363 | 76 | 20% | 4 | 0 | ✗ |
| segmented_radix_sort | 311 | 48 | 15% | 0 | 0 | ✗ |
| batch_memcpy | 227 | 95 | 41% | 0 | 0 | ✗ |
| batched_topk | 186 | 66 | 35% | 0 | 0 | ✗ |
| merge_sort | 193 | 83 | 43% | 0 | 0 | ✗ |
| merge | 180 | 89 | 49% | 0 | 0 | ✗ |
| segmented_reduce | 189 | 51 | 26% | 0 | 0 | ✗ |
| segmented_scan | 158 | 45 | 28% | 0 | 0 | ✗ |
| adjacent_difference | 118 | 77 | 65% | 0 | 0 | ✗ |
| find | 90 | 39 | 43% | 0 | 0 | ✗ |
| find_bound_sorted_values | 106 | 47 | 44% | 0 | 0 | ✗ |
| transform_tile | 85 | 33 | 38% | 0 | 0 | ✗ |
| for | 78 | 51 | 65% | 0 | 1 | ✗ |
| **TOTAL** | **18094** | **3568** | **19%** | **294** | **37** | **1/26** |

## Reduce: CCCL SM100 → muh BI-V100 divergence analysis

### SM100 benchmark annotations in CCCL
```
ipt_15.tpb_512.ipv_2  1.020  1.000  1.018  1.058  (geo=1.024) — accum8, offset4
ipt_15.tpb_512.ipv_1  1.019  1.000  1.017  1.057  (geo=1.023) — accum8, offset8
ipt_16.tpb_512.ipv_2  1.061  1.000  1.065  1.167  (geo=1.072) — float32, offset4
ipt_16.tpb_640.ipv_1  1.018  1.000  1.016  1.057  (geo=1.022) — float64, offset4
ipt_13.tpb_224        1.107  1.010  1.097  1.317  (geo=1.127) — deterministic float32 (sm90)
ipt_6.tpb_224         1.034  1.000  1.032  1.091  (geo=1.039) — deterministic float32 (sm86)
```

### Key divergences

| Parameter | CCCL SM100 | muh BI-V100 | Rationale | Risk |
|-----------|-----------|------------|-----------|------|
| float32+plus items | 16 | 24 | Compensate for 16 vs 148 SMs | Unvalidated: may hurt L1 hit rate |
| float64+plus threads | 640 | 384 | Clean 12-warp config | May underutilize vs 20-warp original |
| float64+plus vec | 1 | 2 | 16B vectorized loads | Alignment risk with non-contiguous data |
| det float32 items | 13 | 32 | More work per CTA on 16 SMs | 2.5× register pressure increase |
| accum1/2/16 | absent | added | Extrapolated from scaling | Not in CCCL SM100, completely theoretical |

## Scan: lookback delay calibration gap

CCCL SM100 lookback delay parameters (from benchmark annotations):
- `delay_ns` range: 228 – 1904 ns
- `dcid` (delay constructor ID) range: 1 – 7
- `l2_write_latency` range: 520 – 965 ns

These are calibrated on SM100's 50MB L2 cache. BI-V100 has 6MB L2 → delay parameters need re-calibration. Current muh values use heuristic scaling (SM100 × 0.5 for ns, × 0.6 for l2w) without hardware validation.

## Priority action items (by Output TPS impact)

| # | Algorithm | CCCL bench pts needed | vllm hot path | Weight |
|---|-----------|----------------------|---------------|--------|
| 1 | reduce | 6 | paged_attention score reduction | 83% |
| 2 | scan | 16 (8 remaining) | softmax denominator | 83% |
| 3 | topk | 0 (format from radix_sort) | vocab=152064 sampling | 83% |
| 4 | radix_sort | 70 | logit sorting for top-k/top-p | 83% |
| 5 | select_if | 82 | top-p token filtering | 83% |
| 6 | transform | 0 (no CCCL benches) | RMSNorm/SiLU/RoPE | 10-15% |
| 7 | scan_by_key | 30 | per-sequence softmax | ~5% |
| 8 | reduce_by_key | 32 | per-sequence aggregation | ~3% |
| 9 | batch_memcpy | 0 | KV cache block copy | 3% |
