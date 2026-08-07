# muh 管道现实检查 — 2026-08-07

## 核心发现

### 1. gen_patch.py 输出为零

```
$ python3 muh/gen_patch.py --dry-run
READ reduce: bi100_plus_float32_o4 → {items: 24, threads: 512, vec: 2}
READ scan: bi100_sm90_float32 → {threads: 128, items: 24}
...
No patches generated.
```

原因: `VLLM_INJECTION_POINTS` 的 key `('reduce', 'partition_size')` 和 struct 提取出的 field `items`/`threads`/`vec` 不匹配。gen_patch 的"读"和"写"两端从未对齐。

### 2. 注入目标是 Python 不是 C++

enginex-vllm-bi100 **没有 `.cu` 源码**。所有 CUDA kernel 是预编译的 ixformer `.so`。

实际可调的全部是 Python 层:

| 文件 | 可调参数 | 竞赛影响 |
|------|---------|---------|
| `paged_attn.py` | `_PARTITION_SIZE=512`, V1/V2 dispatch logic | Output TPS (83%) |
| `prefix_prefill.py` | `BLOCK=64`, `BLOCK_N=64`, `NUM_WARPS=4` | Input TPS (14%) |
| `vllm/attention/ops/triton_flash_attention.py` | 17 个 autotune configs | Prefill throughput |
| `vllm/_custom_ops.py` | `return 49152` (SMEM fix) | 所有 Triton kernels |
| `computility-run.yaml` | `--max-num-seqs`, `--gpu-memory-utilization` | 调度效率 |

gen_patch.py 中的 `csrc/*.cu` 注入点全部是 dead code (注释已标注)。

### 3. muh C++ headers 的实际价值

muh 的 26 个 tuning headers 和 `scale_mem_bound` 实现是正确的理论分析工具。它们的价值不在于直接注入 vllm，而在于:

- 推导 SMEM 约束 (Triton `BLOCK_M × head_dim × elem_size` 上限)
- 推导 occupancy 模型 (BI-V100 16 SMs 的 wave efficiency)
- 推导 bytes_in_flight (56 GB/s per-SM → 64KB prefetch window → `num_stages=2`)
- 为 CCCL benchmark 验证提供 ground truth

这些推导已经手工应用到了 Python 代码中:
- `triton_flash_attention.py` 的 8 个 BI-V100 configs 引用了 CCCL babelstream/scan 分析
- `prefix_prefill.py` 的 BLOCK_N=64 推导基于 48KB SMEM 约束
- `_custom_ops.py` 的 49152 来自 hardware.cuh

### 4. 管道闭环的正确路径

```
CCCL tuning analysis    Python layer injection   Triton autotune
(理论推导)               (参数修改)                (运行时选择)
        │                       │                       │
        ▼                       ▼                       ▼
muh headers           paged_attn.py             triton.Config([...])
common.cuh            prefix_prefill.py         autotune picks best
hardware.cuh          _custom_ops.py            at runtime
        │                       │                       │
        └───────────────────────┴───────────────────────┘
                                │
                        竞赛评测得分
```

不是: `muh headers → gen_patch → #define injection → recompile`
而是: `muh analysis → Python config → Triton autotune → runtime perf`

## 下一步

1. 删除 gen_patch.py 中所有 dead `csrc/*.cu` 注入点
2. 重写 gen_patch 为 `gen_config.py`: 从 muh headers 推导 → 直接输出 Python patch
3. 用 CCCL benchmarks 验证: reduce/sum.cu, scan/exclusive/sum.cu, topk/keys.cu
4. 扩展 triton_flash_attention.py autotune 搜索空间 (当前 17 configs, 可加到 30+)
