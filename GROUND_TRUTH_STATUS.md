# project_6 真实状态报告

生成时间: 2026-08-05, commit 96f6465

## 一句话总结

**enginex 没有 .cu 源码，gen_patch 的 C++ injection 管道全部失效。** 实际可用的优化路径只有 Python/Triton 层面的参数调优。muh 的 27 个 C++ tuning headers 是正确的架构设计，但在竞赛引擎上无处注入。

---

## 1. 竞赛引擎的致命事实

```
gen_patch.py 第 47 行:
  WARNING: ALL csrc/*.cu targets are DEAD — files do not exist.
  enginex-vllm-bi100-qwen36 ships: Python + precompiled .so + Triton.
  No .cu source files. gen_patch patches have zero effect.
```

enginex 交付物 = Python 文件 + 预编译 .so + Triton kernels。  
不提供 C 源码 → 无法修改 CUDA kernel → C++ tuning header 无法注入到 vllm 的编译产物里。

**真正的优化路径:**
- Triton kernels (prefix_prefill.py, paged_attn.py): 可以改 BLOCK、NUM_WARPS 等 JIT 参数
- Python 配置层 (computility-run.yaml): max_model_len、gpu_memory_utilization 等
- 模型适配 (qwen3_5.py): MoE routing、attention 实现

## 2. 已有的 benchmark 数据 (真实的)

| 算法域 | 已跑配置数 | 来源 |
|--------|-----------|------|
| flash_attn | 22 configs | bi100_configs.json, SMEM 约束扫描 |
| prefill (Triton) | 9 configs | bi100_configs.json, BLOCK×NUM_WARPS |
| MoE | 5 configs | bi100_configs.json, BLOCK_SIZE_M |
| reduce/scan/topk CUB | 0 | bench_bi100.py 已写但需要 BI-V100 硬件才能跑 |

## 3. muh C++ headers vs CCCL 覆盖率

| 算法 | muh 行数 | CCCL 行数 | 覆盖率 | 竞赛优先级 |
|------|---------|---------|--------|-----------|
| reduce | 297 | 478 | 62% | **P0** — Output TPS 83% 权重 |
| scan | 352 | 1525 | 23% | **P0** — softmax 累积 |
| topk | 113 | 121 | 93% | **P0** — sampling 路径 |
| transform | 185 | 549 | 33% | P1 — RMSNorm/SiLU |
| select_if | 459 | 2729 | 16% | P1 — token filtering |
| radix_sort | 222 | 2381 | 9% | P1 — full sort path |
| scan_by_key | 145 | 2008 | 7% | P1 — per-seq softmax |
| reduce_by_key | 171 | 1735 | 9% | P1 — score aggregation |
| unique_by_key | 166 | 1539 | 10% | P1 — KV cache dedup |
| 其余 18 个 | 33-189 | 78-788 | 10-65% | P2 |

总计: muh 3618 行 vs CCCL 17000+ 行 = 平均 21% 覆盖率

## 4. CCCL 资产完整性

cccl_upstream/ 34MB, 3432 files — 是精选提取, 不是 full clone。

**已有 (竞赛必需的全有):**
- 27/27 tuning headers ✓
- 32/32 dispatch implementations ✓ 
- 25/25 agent kernels ✓
- 60/60 Thrust examples ✓
- 243 CUB tests ✓
- 78 CUB benchmark .cu files ✓
- 230 Thrust tests ✓
- 48 Thrust benchmark algorithms ✓

**不需要 full clone。** 缺的 ~21000 文件是 CI/CD、cudax、Python bindings、docs。

## 5. 真正的行动路径

### 短期 (功能测试通过)
竞赛门控: 50+ 功能测试全通过 + 效果偏差 ≤ ±4%

关键文件:
- `computility-run.yaml` — 控制 vllm 启动参数
- `qwen3_6_scripts/qwen3_5.py` (588行) — MoE 模型适配
- `prefix_prefill.py` — Triton prefill kernel, 可调 BLOCK/NUM_WARPS
- `paged_attn.py` — Triton decode kernel

### 中期 (性能优化)
目标: Token 吞吐加权值 ≥ 8000

```
加权值 = Output_TPS × 16.796 + Input_TPS × 2.799 + Cache_TPS × 0.56
```

**Output TPS (83%):** decode kernel → paged_attn.py Triton 参数优化  
**Input TPS (14%):** prefill kernel → prefix_prefill.py Triton 参数优化  
**Cache TPS (3%):** prefix caching 配置

### 长期 (如果能编译 C++)
如果能获取 EngineX 的 C 编译环境:
- muh C++ headers 可以直接注入
- bench_bi100.py 的 CUB parameter sweep 可以在 BI-V100 上跑
- 这条路 ROI 最高但依赖竞赛方提供编译链

## 6. 代码架构

```
project_6/
├── computility-run.yaml          ← 竞赛提交配置 (直接影响评测)
├── baseline.muh                  ← muh 格式的 vllm 配置
├── Dockerfile                    ← 竞赛镜像构建
├── cccl_upstream/                ← CCCL 精选 (34MB, 3432 files)
│   ├── cub/                      ← CUB: dispatch/tuning/agent/test/bench
│   ├── thrust/                   ← Thrust: examples/testing/benchmarks
│   └── libcudacxx/               ← CUDA 标准库
├── muh/                          ← kernel tuning 框架 (544KB)
│   ├── include/muh/tuning/       ← 27 个 BI-V100 tuning headers
│   ├── bench_bi100.py            ← CUB parameter sweep runner
│   ├── gen_patch.py              ← vllm patch 生成 (C++ 注入点已死)
│   ├── gen_yaml.py               ← computility-run.yaml 生成
│   └── parse.py                  ← .muh 配置解析器
├── muh_kernel_map.py             ← CCCL 算法 → vllm kernel 映射
├── muh_dispatch.py               ← 运行时 policy 分派
├── vllm/                         ← vllm 引擎源码 (11MB Python)
├── vllm_adapter/                 ← Qwen3.5 模型适配 + 部署脚本
├── qwen3_6_scripts/              ← Qwen3.6 patch 集合 (576KB, 25+ patches)
├── prefix_prefill.py             ← Triton prefill kernel (可调优)
├── paged_attn.py                 ← Triton decode kernel (可调优)
├── attention.py                  ← Attention 实现
└── enginex-vllm-bi100-qwen36-main.zip ← 竞赛基础引擎 (97MB)
```
