# project_6 真实状态 v2

更新时间: 2026-08-06, 基于完整代码阅读

## 核心事实

**enginex 没有 .cu 源码。gen_patch 的 C++ injection 全部失效。** 但这不是终点。

实际可优化的三条路径：

### 路径 1: Triton kernel 参数调优 (直接有效)

文件: `prefix_prefill.py` (895行), `paged_attn.py` (794行)
状态: 22 个 flash_attn 配置 + 9 个 prefill 配置已计算 SMEM，未上机实测
关键参数:
- prefill: BLOCK_M, BLOCK_N, NUM_WARPS (已有 SMEM 约束扫描)
- decode: _PARTITION_SIZE=512 (硬编码), V1/V2 切换阈值
- 竞赛权重: Output TPS×16.796(83%) + Input TPS×2.799(14%)

gen_patch.py 第 87-103 行已经指向了这些真正的 injection points:
```python
('prefill', 'BLOCK_M'): [('prefix_prefill.py', 'BLOCK')],
('flash_attn', 'BLOCK_M'): [('vllm/attention/ops/triton_flash_attention.py', 'BLOCK_M')],
('moe', 'BLOCK_SIZE_M'): [('vllm/model_executor/layers/fused_moe/fused_moe.py', 'BLOCK_SIZE_M')],
```

### 路径 2: 模型适配 (功能门控)

文件: `vllm_adapter/qwen3_5.py` (588行), `qwen3_6_scripts/` (25+ patches)
状态: MoE 256 experts top-8 注册完成，treat ALL layers as full attention
待验证: TP=4 加载, reasoning 分离, tool_call parsing
竞赛门控: 50+ 功能测试全通过 + 效果偏差 ≤±4%

### 路径 3: vllm Python 层配置优化 (低风险高收益)

文件: `computility-run.yaml`, `baseline.muh`
关键发现 from paged_attn.py:
- 第 99 行: `use_v1 = True` 硬编码禁用了 V2 — 对 100K token 序列这是性能杀手
- `_PARTITION_SIZE = 512` 硬编码 — 应该根据 SM count=16 动态调整
- `max_num_seqs: 1` — 限制了批处理并行度
- `--enable-prefix-caching` — 已开启，但 cache copy kernel 未优化

## CCCL 资产的真实价值

CCCL 的价值不在于 C++ 注入（已证实失效），而在于：

1. **参数空间知识**: 27 个 tuning_*.cuh 告诉我们 NVIDIA 在 3 代 GPU 上搜索了哪些参数维度
   - reduce: ipt×tpb×ipv = 1044 个组合
   - scan: ipt×tpb×ns×dcid×l2w×trp×ld = ~26B 个（剪枝后可管理）
   - 这些维度完全适用于 Triton kernel 的等价参数

2. **benchmark 数据**: 199 条标注告诉我们在不同 problem size 下的加速比分布
   - 小数据量(<16M): 大多数优化无效（speedup≈1.0）
   - 大数据量(>256M): 加速比显著（最高 1.58x）
   - 这意味着 decode（小 batch）和 prefill（大 batch）需要不同策略

3. **约束模型**: scale_mem_bound, SMEM 公式, occupancy 计算
   - BI-V100: 16 SM, 48KB SMEM, 900GB/s BW
   - per-SM BW = 56 GB/s ≈ B200 水平
   - bytes_in_flight = 64KB (bench_bi100.py 已验证)

4. **算法映射**: muh_kernel_map.py 的 VLLM_KERNEL_MAP 精确映射了每个 vllm kernel 对应的 CCCL 算法
   - paged_attention → reduce (summary_statistics.cu Welford pattern)
   - softmax → scan
   - sampling → topk + radix_sort
   - normalization → transform + reduce

## bench_bi100.py 的实际作用

bench_bi100.py (713行) 是真正的工具 — 它用 PyTorch CUDA 操作模拟 CCCL benchmark:
- 不需要编译 C++，不需要 nvbench
- 直接在 BI-V100 上跑 torch.sum/torch.cumsum/torch.topk
- 输出 CCCL 格式: `ipt_N.tpb_M.ipv_K speedup0 speedup1 speedup2 speedup3`
- 搜索空间定义完整: reduce 1044 组合, scan 剪枝后可管理, topk/transform 都有

**但它需要 BI-V100 硬件才能跑。** 在 Phanthy Cloud 上部署就能开始标定。

## 代码覆盖率 (muh vs CCCL)

| 算法 | muh 行 | CCCL 行 | 比率 | 竞赛价值 |
|------|--------|---------|------|---------|
| reduce | 297 | 478 | 62% | 最高 — Output TPS 83% |
| topk | 113 | 121 | 93% | 高 — 每次 decode |
| scan | 370 | 1525 | 24% | 高 — softmax |
| transform | 185 | 549 | 34% | 中 — RMSNorm/SiLU |
| select_if | 459 | 2729 | 17% | 中 — token filter |
| radix_sort | 222 | 2381 | 9% | 中 — full sort |
| scan_by_key | 145 | 2008 | 7% | 中 — per-seq scan |
| reduce_by_key | 171 | 1735 | 10% | 中 — score aggregation |
| unique_by_key | 166 | 1539 | 11% | 低 — KV dedup |
| 其余 18 个 | 33-189 | 78-788 | varies | 低 |

muh 总计 3618 行 / CCCL 17000+ 行 = 21% 平均覆盖率。
reduce 和 topk 覆盖率最高（62%、93%），正好是竞赛权重最大的两个算法。

## 下一步具体行动

1. **在 Phanthy Cloud 上跑 bench_bi100.py** — 产出 BI-V100 真实 benchmark 数据
2. **把 benchmark 结果回填到 Triton kernel 参数** — prefix_prefill.py 的 BLOCK/NUM_WARPS
3. **修复 paged_attn.py 的 V2 禁用** — 对长序列性能至关重要
4. **功能测试回归** — 确保 qwen3_5.py 适配通过 50+ 用例
