# EngineX vllm Injection Point Map

> **Source**: `enginex-vllm-bi100-qwen36-main.zip` (101MB, 1444 files)
> **Generated**: 2026-08-02 from full source analysis

---

## 关键发现

### 1. 不是 C++ CUDA 文件注入 — 是 Python 层

EngineX vllm 的 CUDA kernels 全部预编译在 `ixformer.functions` (ixf_F) 中，打包在基础镜像里。
`_custom_ops.py` 是 Python 薄封装层，调用 `ixf_F.vllm_single_query_cached_kv_attention()` 等。

**没有 .cu 文件可以直接 patch。** muh 的 gen_patch.py 需要改为 patch Python 文件，不是 C++ 文件。

### 2. paged_attention_v2 未实现

```python
def paged_attention_v2(...) -> None:
    raise NotImplementedError()
```

且 `use_v1 = True` 硬编码覆盖了启发式逻辑。所有 decode 都走 v1。

### 3. 实际可调参数 (THE TUNING SURFACE)

| 参数 | 文件 | 当前值 | 作用 | 优先级 |
|------|------|--------|------|--------|
| `_PARTITION_SIZE` | `vllm/attention/ops/paged_attn.py:13` | 512 | PagedAttention partition (v2 用) | 低 (v2 disabled) |
| `use_v1` | `paged_attn.py:128` | `True` (hardcoded) | 强制 v1 | **P0** — 解锁 v2 可能提升长序列 |
| `BLOCK` | `prefix_prefill.py:712` | 128 (cc≥80) / 64 | Triton prefill tile size | **P0** — 直接影响 Input TPS |
| `NUM_WARPS` | `prefix_prefill.py:713` | 8 | Triton warp count | **P0** |
| `BLOCK_SIZE_M/N/K` | `fused_moe.py:342-344` | 64/64/32 | MoE kernel tile | **P0** — Qwen3.6 是 MoE |
| `get_max_shared_memory` | `_custom_ops.py:892` | `32 * 1024` | SMEM 上限声明 | **P0** — 可能错误限制性能 |
| Triton flash attention configs | `triton_flash_attention.py:214-303` | 8 个 triton.Config | Triton autotune 搜索空间 | P1 |

### 4. SMEM 32KB vs 48KB 冲突

`_custom_ops.py:892` 返回 `32 * 1024` (32KB)。
但 `hardware.cuh` 和 muh 假设 49152 (48KB)。
如果 BI-V100 实际 SMEM 是 32KB，则 muh 所有 tuning 的 SMEM 约束都需要从 48KB 降到 32KB。

### 5. ixf_F kernel 列表 (不可改，只能调参)

| Python 封装 | ixf_F 调用 | 说明 |
|-------------|-----------|------|
| `paged_attention_v1` | `ixf_F.vllm_single_query_cached_kv_attention` | decode 核心 |
| `silu_and_mul` | `ixf_F.silu_and_mul` | SwiGLU 激活 |
| `rms_norm` | `ixf_F.rms_norm` | LayerNorm |
| `fused_add_rms_norm` | `ixf_F.fused_add_rms_norm` | 融合残差+norm |
| `rotary_embedding` | `ixf_F.vllm_rotary_embedding_neox` | RoPE 位置编码 |
| `reshape_and_cache` | `ixf_F.vllm_cache_ops_reshape_and_cache` | KV cache 写入 |
| `copy_blocks` | `ixf_F.copy_blocks` | prefix cache block 复制 |
| `moe_align_block_size` | `ixf_F.vllm_moe_align_block_size` | MoE token 排列 |
| `invoke_fused_moe_kernel` | `ixf_F.vllm_invoke_fused_moe_kernel` | MoE GEMM |
| `topk_softmax` | `ixf_F.vllm_moe_topk_softmax` | MoE routing |
| `cutlass_scaled_mm` | `ixf_F.w8a8` | INT8 矩阵乘 |

### 6. Triton kernels (可直接修改)

这些是 Python Triton JIT 编译的 kernel，可以直接改源码：

- `prefix_prefill.py` — 3 个 `_fwd_kernel` 变体 (context attention)
- `triton_flash_attention.py` — Triton flash attention (8 个 autotune configs)
- `fused_moe.py` — MoE GEMM kernel (Triton, 自定义 config)

---

## muh 策略修正

### 旧策略 (假设 C++ injection)
```
CCCL tuning_*.cuh → muh bi100_* → gen_patch.py → C++ #define 注入 → 编译 .so
```

### 新策略 (实际 Python injection)
```
层1: Python 参数调优
  paged_attn.py: _PARTITION_SIZE, use_v1
  prefix_prefill.py: BLOCK, NUM_WARPS
  fused_moe.py: BLOCK_SIZE_M/N/K
  _custom_ops.py: get_max_shared_memory (32KB→实测值)

层2: Triton kernel 优化
  prefix_prefill.py: 3 个 _fwd_kernel — tile size, loop structure
  triton_flash_attention.py: autotune config 添加 BI-V100 特化
  fused_moe.py: MoE GEMM kernel tune

层3: CCCL/muh 知识迁移
  用 CCCL 的 tuning 方法论指导 Triton kernel 参数选择
  不是直接注入 C++ 值，而是把 CCCL 的 policy_selector 逻辑
  翻译成 Triton constexpr 参数
```
