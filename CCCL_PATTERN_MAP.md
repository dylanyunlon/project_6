# CCCL → vllm Kernel Pattern Mapping
## BI-V100 Competition Reference

### Pattern 1: Multi-field Reduction (paged_attention)

**CCCL source**: `thrust/examples/bounding_box.cu`, `summary_statistics.cu`
**vllm kernel**: `paged_attn.py` → ixformer paged_attention_v1/v2

```
CCCL:  transform_reduce(begin, end, unary_op, init, binary_op)
vllm:  for each KV block: score = Q·K, max_score = reduce_max, exp_sum = reduce_sum
```

**Tuning surface**:
- `_PARTITION_SIZE`: controls how many KV tokens per CTA in V2 mode
- V1/V2 dispatch threshold: `total_tiles vs 2 × sm_count`
- BI-V100: 16 SMs → V2 beneficial when seq_len > 1024 (2 waves of 16 CTAs × 512 partition)

**CCCL parameter**: `ReducePassPolicy{threads=512, items=24, vec=2, WARP_REDUCTIONS, LDG}`

### Pattern 2: Prefix Scan + Transform (softmax)

**CCCL source**: `thrust/examples/simple_moving_average.cu`, `cub/benchmarks/bench/scan/exclusive/sum.cu`
**vllm kernel**: `prefix_prefill.py` context_attention_fwd_kernel

```
CCCL:  inclusive_scan(begin, end, output, plus<float>)
vllm:  for each BLOCK_N chunk: qk = Q·K, m_new = max(m_old, max(qk)),
       l_new = l_old * exp(m_old - m_new) + sum(exp(qk - m_new))
```

**Tuning surface**:
- `BLOCK_M`: Q tile rows (32 or 64 for BI-V100)
- `BLOCK_N`: K/V sweep width (32 or 64)
- `NUM_WARPS`: 4 (16 SMs don't benefit from 8 warps per CTA)
- `num_stages`: 1 (no cp.async) or 2 (software pipeline)

**CCCL parameter**: `ScanLookbackPolicy{threads=384, items=22, WARP_TRANSPOSE, DEFAULT, WARP_SCANS, {backon_jitter_window, 952, 415}}`

### Pattern 3: Transform (activation functions)

**CCCL source**: `cub/benchmarks/bench/transform/babelstream.cu`
**vllm kernel**: Triton SiLU, GeLU, RMSNorm kernels (via `_custom_ops.py`)

```
CCCL:  transform(begin, end, output, silu_op)    // x * sigmoid(x)
vllm:  @triton.jit def silu_kernel(x): tl.sigmoid(x) * x
```

**Tuning surface**:
- `bytes_in_flight`: 64KB on BI-V100 (56 GB/s per-SM × 1100ns latency)
- Triton `num_stages=2` maps to BIF=64KB (2× prefetch window)
- `SMEM = 49152` (fixed by _custom_ops.py)

**CCCL parameter**: `TransformPrefetchPolicy{threads=256, bif=64KB, prefetch_stride=128}`

### Pattern 4: TopK (sampling)

**CCCL source**: `cub/benchmarks/bench/topk/keys.cu`
**vllm kernel**: sampling_kernels (precompiled .so)

```
CCCL:  DeviceTopk::TopK(keys, k, output)
vllm:  ixformer topk_sampling → radix_sort + select partial
```

**Tuning surface** (via .so, limited):
- `bits_per_pass`: 11 for float32 (32 bits / 3 passes)
- Thread count: 512 (baked into .so)

### Pattern 5: Triton Flash Attention (all patterns combined)

**CCCL source**: All of the above + `cub/agent/agent_scan.cuh` union SMEM model
**vllm kernel**: `triton_flash_attention.py`

```
Q_resident × K_streaming × softmax_online → Output
= transform_reduce (Q·K) + scan (softmax) + transform (V matmul)
```

**Tuning surface**: 17 existing + 19 new autotune configs from gen_config.py
**Key configs for BI-V100**:
```python
# Best for long context (seq_len > 4096):
Config(BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2)  # 40KB SMEM, 1 CTA/SM

# Best for short context (seq_len < 1024):
Config(BLOCK_M=32, BLOCK_N=32, num_warps=2, num_stages=2)  # 32KB SMEM, 2 CTAs/SM
```

---

### CCCL Asset Utilization Summary

| CCCL Asset | Files | Used for BI-V100 | Competition Impact |
|-----------|-------|-------------------|-------------------|
| Tuning headers (26) | 18094 lines | 3568 lines (20%) | P0: reduce/scan/transform |
| CUB benchmarks (80) | reduce/scan/topk/transform | benchmark framework | P0: parameter search |
| Thrust examples (52) | summary_stats/bounding_box/norm | pattern mapping | P1: architecture understanding |
| CUB tests (243) | correctness verification | 0% (need BI-V100) | P2: correctness |
| libcudacxx (1463) | type traits, atomics | implicit (via CUB) | Infra |

**Total usable CCCL assets**: 5205 files in cccl_upstream
**Competition-critical subset**: ~30 files (5 tuning headers + 10 benchmarks + 15 examples)
