# CCCL Asset → Competition Value Mapping

## Executive Summary

project_6 now contains **4,295 CCCL files** (42MB) — a strategic subset of NVIDIA's CCCL (135MB full).
We have **100%** of the competition-critical assets and **0%** of the irrelevant CI/Python/docs bloat.

## Asset Inventory

### Tier 1: Direct Competition Impact (ALL PRESENT ✓)

| CCCL Asset | Files | PRD Items | Competition Path |
|-----------|-------|-----------|-----------------|
| 26 tuning_*.cuh (SM80/90/100 benchmarks) | 26 | [muh] 语言规范, all标定items | The benchmark data we're adapting to BI-V100 |
| 27 muh tuning_*.cuh (BI-V100 adapted) | 29 | [EPIC] 27/27 CCCL parity | Our kernel tuning injection layer |
| 32 dispatch_*.cuh (algorithm impl) | 32 | gen_patch injection points | Where muh values get injected |
| 153 CUB benchmarks (.cu) | 153 | [muh-bench] reduce/scan/topk/transform | The actual benchmark binaries |
| 60 Thrust examples (.cu) | 60 | [CCCL-verify] all 22 items | Correctness verification suite |
| 217 CUB Catch2 tests (.cu) | 217 | [CCCL-test] all 8 items | Regression test matrix |
| 18 CUB examples (.cu) | 18 | [CCCL-verify] device_reduce/scan/topk | API-level verification |

### Tier 2: Build & Test Infrastructure (NOW PRESENT ✓)

| CCCL Asset | Files | Purpose |
|-----------|-------|---------|
| c2h/ (test helpers) | 27 | Catch2 test generators, validators, runner |
| nvbench_helper/ | 10 | Benchmark harness utilities for CUB benches |
| cmake/ | 29 | CMake presets, build helpers, target definitions |
| CMakePresets.json | 1 | Standardized build configurations |
| AGENTS.md / CLAUDE.md | 1 | NVIDIA's own AI agent instructions for CCCL |

### Tier 3: Extended Library (NOW PRESENT ✓)

| CCCL Asset | Files | Purpose |
|-----------|-------|---------|
| cudax/ | 794 | Experimental CUDA extensions (memory resources, launch, async) |
| libcudacxx/ | 1463 | CUDA C++ Standard Library headers |

### NOT Included (by design)

| CCCL Asset | Why Excluded |
|-----------|-------------|
| .github/, ci/ (65 files) | GitHub Actions workflows — irrelevant |
| python/ | Python bindings — we use C++ directly |
| docs/ (25 files) | Markdown docs — we have the source code |
| .git history | ~100MB of git objects — no value |

## Competition Critical Path

```
竞赛门槛: Token吞吐加权值 ≥ 8000
  = Output TPS × 16.796 (83%) + Input TPS × 2.799 (14%) + Cache TPS × 0.56 (3%)

CCCL → muh → vllm injection chain:
  cccl_upstream/cub/.../tuning_reduce.cuh  (SM100 benchmark data: ipt_16.tpb_512 speedup=1.148)
    → muh/include/muh/tuning/tuning_reduce.cuh  (BI-V100 adapted: SMEM ≤ 48KB)
      → muh/gen_patch.py  (extract bi100_* structs → unified diff)
        → vllm csrc/attention/paged_attention_v2.cu  (NUM_THREADS=512, VEC_SIZE=2)
          → Docker build → Phanthy Cloud 4×BI-V100 → 竞赛评测
```

## CCCL Examples → PRD Items Cross-Reference

| Thrust Example | PRD [CCCL-verify] Item | vllm Kernel Path |
|---------------|----------------------|-----------------|
| summary_statistics.cu | summary_statistics (P1) | benchmark 统计分析 |
| sort.cu | sort (P0) | top-k sampling radix sort |
| scan_by_key.cu | scan_by_key (P0) | softmax denominator |
| stream_compaction.cu | stream_compaction (P0) | token filtering |
| histogram.cu | histogram (P1) | repetition_penalty |
| norm.cu | norm (P0) | RMSNorm 精度基准 |
| saxpy.cu | saxpy (P0) | SiLU/RoPE/bias_add |
| run_length_encoding.cu | run_length_encoding (P1) | attention mask 压缩 |
| sum.cu + sum_rows.cu | sum+sum_rows (P0) | attention score reduction |
| dot_products_with_zip.cu | dot_products (P1) | multi-head attention score |
| sparse_vector.cu | sparse_vector (P1) | sparse attention |
| weld_vertices.cu | weld_vertices (P1) | KV cache deduplication |
| max_abs_diff.cu | max_abs_diff (P2) | 效果测试精度对比 |
| monte_carlo.cu | monte_carlo (P2) | temperature sampling |

## File Count Summary

| Component | Before | After | Delta |
|-----------|--------|-------|-------|
| cccl_upstream/ total | 3,432 | 4,295 | +863 |
| + c2h (test helpers) | 0 | 27 | +27 |
| + nvbench_helper | 0 | 10 | +10 |
| + cmake (build system) | 0 | 29 | +29 |
| + cudax (experimental) | 0 | 794 | +794 |
| + metadata files | 0 | 3 | +3 |
