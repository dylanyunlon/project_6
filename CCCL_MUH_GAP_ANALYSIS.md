# CCCL ↔ muh 完整 Gap 分析

> 生成时间: 2026-08-06 | HEAD: 2a7ca10 | 26 算法全量扫描

## 核心数据

| 指标 | 值 | 说明 |
|------|------|------|
| CCCL 算法总数 | 26 | cub/device/dispatch/tuning/ 下所有 tuning_*.cuh |
| muh tuning headers | 26 | 1:1 文件对应 ✓ |
| CCCL 代码行 | 18,094 | 所有 tuning_*.cuh 总和 |
| muh 代码行 | 3,568 | 19.7% 覆盖率 |
| CCCL benchmark 注释 | 299 | `ipt_N.tpb_M ... speedup` 格式的数据点 |
| SM100 模板特化 | 157 | NVIDIA 为 SM100 跑出的最优配置数 |
| BI-V100 命名 struct | 37 | muh 中 `bi100_*` struct 数量 |
| 有 bi100 struct 的算法 | 3/26 | reduce(14个), scan(22个), for(1个) |
| 有 SMEM 保护的算法 | 16/26 | scale_mem_bound 或 while loop |

## 关键发现

### 1. 只有 reduce 和 scan 达到了"READY"状态

reduce 和 scan 是唯一两个同时具备 bi100 命名 struct + SMEM 保护 + 完整 policy_selector 的算法。但即便如此，这些 struct 的值全部是从 SM100 推导的**理论值**，没有一个在 BI-V100 上实测过。

### 2. 其余 24 个算法停留在"inline only"

"inline only" 意味着 muh header 里有 policy_selector，但它的值是硬编码在 if/else 分支里的，不是通过命名 struct 暴露的。gen_patch.py 提取不到这些值（它只认 `struct bi100_*` 模式）。

### 3. CCCL 有 299 个 benchmark 数据点，muh 有 0 个

CCCL 的 benchmark 注释格式完美定义了目标：
```
ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148442 0.997167 1.139902 1.462651
```
四个数字 = 四个 problem size 下的加速比。muh 需要在 BI-V100 上产出同样格式的 299 个数据点来填充所有空位。

### 4. 竞赛瓶颈不在代码量而在实测数据

- 代码架构已经搭好（26 个 header + policy_selector + gen_patch 管道）
- 缺的是 BI-V100 实测数据来替换理论值
- 没有实测数据，所有 bi100_* struct 的值都是猜的

## 26 算法状态矩阵

| 算法 | CCCL 行 | muh 行 | CCCL BM | SM100 特化 | bi100 struct | SMEM✓ | 状态 |
|------|---------|--------|---------|-----------|-------------|-------|------|
| reduce | 478 | 297 | 7 | 6 | 14 | ✓ | ✓ READY |
| scan | 1,525 | 591 | 18 | 12 | 22 | ✓ | ✓ READY |
| for | 78 | 51 | 0 | 0 | 1 | ✗ | ⚠ no SMEM |
| topk | 121 | 113 | 0 | 0 | 0 | ✗ | △ inline |
| transform | 549 | 185 | 0 | 0 | 0 | ✗ | △ inline |
| batch_memcpy | 227 | 95 | 0 | 0 | 0 | ✗ | △ inline |
| select_if | 2,729 | 459 | 84 | 52 | 0 | ✓ | △ inline |
| radix_sort | 2,381 | 222 | 70 | 0 | 0 | ✓ | △ inline |
| scan_by_key | 2,008 | 145 | 30 | 17 | 0 | ✓ | △ inline |
| reduce_by_key | 1,735 | 171 | 32 | 22 | 0 | ✓ | △ inline |
| unique_by_key | 1,539 | 166 | 29 | 21 | 0 | ✓ | △ inline |
| three_way_partition | 788 | 99 | 13 | 9 | 0 | ✓ | △ inline |
| rle_non_trivial_runs | 691 | 68 | 8 | 8 | 0 | ✗ | △ inline |
| segmented_sort | 640 | 189 | 0 | 0 | 0 | ✓ | △ inline |
| rle_encode | 626 | 63 | 4 | 7 | 0 | ✗ | △ inline |
| histogram | 363 | 76 | 4 | 3 | 0 | ✗ | △ inline |
| segmented_radix_sort | 311 | 48 | 0 | 0 | 0 | ✓ | △ inline |
| batch_memcpy | 227 | 95 | 0 | 0 | 0 | ✗ | △ inline |
| merge_sort | 193 | 83 | 0 | 0 | 0 | ✓ | △ inline |
| segmented_reduce | 189 | 51 | 0 | 0 | 0 | ✗ | △ inline |
| batched_topk | 186 | 66 | 0 | 0 | 0 | ✓ | △ inline |
| merge | 180 | 89 | 0 | 0 | 0 | ✓ | △ inline |
| segmented_scan | 158 | 45 | 0 | 0 | 0 | ✓ | △ inline |
| adjacent_difference | 118 | 77 | 0 | 0 | 0 | ✓ | △ inline |
| find_bound_sorted_values | 106 | 47 | 0 | 0 | 0 | ✗ | △ inline |
| find | 90 | 39 | 0 | 0 | 0 | ✓ | △ inline |
| transform_tile | 85 | 33 | 0 | 0 | 0 | ✗ | △ inline |

## gen_patch 管道状态

当前 gen_patch.py 跑出来的结果：

```
READ  reduce:    bi100_plus_float32_o4 → {items:24, threads:512, vec:2}
READ  scan:      bi100_sm90_float32    → {threads:128, items:24}
READ  topk:      __inline_topk__       → {threads:512, bits_per_pass:11}
READ  transform: __inline_transform__  → {bytes_in_flight:64}
READ  for:       bi100_default         → {threads:256, items:4}
SKIP  其余 21 个算法: no bi100_* structs
```

**0 个 patch 生成**——因为 VLLM_INJECTION_POINTS 映射表中的 key 与当前 struct 字段名不匹配。这是管道断裂点。

## CCCL benchmark 源码作为 muh 的输入规范

CCCL bench/reduce/base.cuh 定义了 benchmark 框架：
- 参数空间：`%RANGE% TUNE_ITEMS_PER_THREAD ipt 7:24:1` / `%RANGE% TUNE_THREADS_PER_BLOCK tpb 128:1024:32`
- 输出格式：`ipt_N.tpb_M.ipv_K speedup0 speedup1 speedup2 speedup3`
- 四个 problem size：`Elements{io}` = 2^16, 2^20, 2^24, 2^28

muh 的 bench_bi100.py 已经有 topk 的实测数据（最佳配置：ipt=4, tpb=512, ld=0），
但 reduce/scan/transform 还没跑。

## CCCL 已有的可直接利用的资产

| 资产类型 | 数量 | 路径 | 用途 |
|----------|------|------|------|
| CUB benchmarks | 80 .cu | cccl_upstream/cub/benchmarks/bench/ | 参数空间搜索框架 |
| CUB tests | 243 .cu | cccl_upstream/cub/test/ | 正确性验证 |
| CUB examples | 18 .cu | cccl_upstream/cub/examples/ | API 验证 |
| Thrust examples | 52 .cu | cccl_upstream/thrust/examples/ | 算法验证 |
| muh schemas | 27 .yaml | muh/schema/ | 参数空间定义 |

总计 420 个 .cu 文件可直接编译运行在 BI-V100 上产出数据。

## 下一步行动

优先级按竞赛权重排序：

1. **reduce 实测** (Output TPS × 16.796 = 83%): 用 bench/reduce/sum.cu 框架，在 BI-V100 上扫描 ipt∈[7,24] × tpb∈{128..1024:32} × ipv∈{1,2,4}
2. **scan 实测** (decode softmax): 用 bench/scan/exclusive/sum.cu 框架，额外标定 LookbackDelay
3. **topk 补全** (sampling): 已有部分数据，需要补 batch=4 和 bits_per_pass 对比
4. **gen_patch 闭环**: 修复 VLLM_INJECTION_POINTS 映射，让 gen_patch 真正产出可用 patch
5. **50+ 功能测试**: 在 patch 后的 vllm 上跑竞赛功能验证
