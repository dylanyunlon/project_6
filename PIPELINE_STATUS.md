# muh Pipeline Status — Ground Truth

**Last verified**: 2026-08-07T01:45:31Z by automated analysis

## Architecture Summary

```
CCCL policy_selector(compute_capability) → ReducePolicy{threads, items, vec, algo, load_mod}
     ↕ mirrors
muh  policy_selector(hardware_capability) → same struct types, BI-V100 values
     ↕ gen_patch.py extracts bi100_* values
vllm patch_ops.sh → full-file Python replacements with tuning values baked in
```

## Injection Reality

### What gen_patch.py THINKS (csrc/*.cu — DEAD)
```
tuning_reduce.cuh → csrc/attention/attention_kernels.cu NUM_THREADS   ← NO .cu SOURCE
tuning_scan.cuh   → csrc/attention/paged_attention_v1.cu SCAN_BLOCK_SIZE ← NO .cu SOURCE
tuning_topk.cuh   → csrc/sampling/sampling_kernels.cu SAMPLING_BLOCK_SIZE ← NO .cu SOURCE
```

### What ACTUALLY happens (Python runtime — ALIVE)
```
_custom_ops.py    → SMEM 49152 (was 32768)     ← DEPLOYED ✓
paged_attn.py     → _PARTITION_SIZE=512         ← DEPLOYED ✓ (V2 partition, NOT CTA tile)
xformers.py       → _Q_CHUNK=256, sdpa_fallback ← DEPLOYED ✓
sampler.py        → torch.topk fast path        ← DEPLOYED ✓
prefix_prefill.py → Triton BLOCK_M/N/warps      ← DEPLOYED ✓ (but Triton not available)
computility-run.yaml → vllm server args         ← DEPLOYED ✓
```

### The Gap
muh C++ headers define precise per-type-per-op tuning values (14 reduce structs, 22 scan structs).
But the vllm engine on BI-V100 runs ixformer .so (precompiled, not tunable) + Python fallbacks.
The C++ headers' values cannot be injected into the precompiled .so.
They CAN inform:
1. Python fallback implementations (paged_attn.py, xformers.py) — tile sizes, chunk sizes
2. Triton JIT configs — if Triton were available (it's not on BI-V100 base image)
3. Future EngineX releases that expose tuning knobs

## Asset Inventory

| Asset | Count | Status |
|-------|-------|--------|
| CCCL tuning headers (upstream) | 27 | Complete |
| muh BI-V100 headers | 27 | Complete (14 reduce + 22 scan + others) |
| muh schema YAMLs | 27 | Complete |
| CUB benchmarks | 91 | Synced to NVIDIA/cccl main |
| CUB tests | 243 | Complete |
| CUB examples | 18 | Complete |
| Thrust examples | 60 | Complete |
| Deployed patches | 15 files | Via patch_ops.sh full replacement |
| bench_bi100.py search spaces | 5 algos | Defined, needs BI-V100 hardware to run |

## Tool Chain Status

| Tool | Input | Output | Status |
|------|-------|--------|--------|
| parse.py | baseline.muh | JSON config | ✓ Working |
| gen_patch.py | tuning_*.cuh | Patch report | ⚠ Reports structs but generates 0 patches (injection mapping mismatch) |
| gen_yaml.py | baseline.muh | computility-run.yaml | ✓ Working |
| bench_bi100.py | algo+dtype | CCCL-format speedup data | Needs BI-V100 hardware |
| patch_ops.sh | qwen3_6_scripts/ | Docker vllm patches | ✓ Working |
| muh_dispatch.py | hw+dtype+head_dim | AttentionConfig | ✓ Working (needs torch) |
| scale_mem_bound | (threads, items, type_size) | (items, threads) | ✓ CCCL parity verified |

## Critical Numbers

| Metric | Competition Threshold | Current Status |
|--------|----------------------|----------------|
| Functional tests | 50+ pass | 13 items In Progress (all FEA) |
| Effect deviation | ≤ ±4% | Untested (needs hardware) |
| Token throughput weighted | ≥ 8000 | Untested |
| Output TPS weight | 83% (×16.796) | Reduce/scan/topk optimization focus |
| SMEM limit | 49152 bytes | All 36 scan+reduce structs verified ✓ |
| SM count | 16 (confirmed) | All headers updated |

## Next Actions (Ranked by Competition Impact)

1. **Run bench_bi100.py on BI-V100** → get real speedup data for reduce/scan/topk
2. **Backfill speedup data to muh headers** → replace TBD/theoretical values
3. **Optimize Python fallback tile sizes** → paged_attn.py, xformers.py Q_CHUNK
4. **Tune computility-run.yaml** → max-num-seqs, max-batched-tokens, gpu-mem-util
5. **Enable prefix caching benchmark** → cached_tokens > 0 for repeat prompts
