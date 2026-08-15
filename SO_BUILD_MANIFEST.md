# 动态链接库完整清单与调用链

## 1. 已有预编译 .so（22 个）→ 调用链状态

### A. 已接入模型调用链（15 个）

| .so | 来源 | 模型中的环境变量 | 状态 |
|-----|------|-----------------|------|
| corex_gdn_causal_conv | 自研 CUDA | `BI100_GDN_COREX_CAUSAL_CONV` (default=True) | ✅ 代码引用 4 处 |
| corex_gdn_gated_norm | 自研 CUDA | `BI100_GDN_COREX_GATED_NORM` (default=True) | ✅ 代码引用 4 处 |
| corex_gdn_beta_decay | 自研 CUDA | `BI100_GDN_COREX_BETA_DECAY` (default=True) | ✅ 代码引用 4 处 |
| corex_gdn_qk_map | 自研 CUDA | `BI100_GDN_COREX_QK_MAP` (default=True) | ✅ 代码引用 4 处 |
| corex_gdn_packed_decode | 自研 CUDA | `BI100_GDN_COREX_PACKED_DECODE` (default=False) | ✅ yaml 已开 |
| corex_gdn_chunk_recurrent | 自研 CUDA | 自动检测 | ✅ 代码引用 4 处 |
| corex_attn_head_rms_norm | 自研 CUDA | `BI100_ATTN_COREX_HEAD_RMS_NORM` (default=True) | ✅ 代码引用 5 处 |
| corex_moe_direct_routed | 自研 CUDA | `BI100_MOE_COREX_DIRECT_ROUTED` (default=False) | ✅ yaml 已开 |
| corex_moe_exact_reduce | 自研 CUDA | `BI100_MOE_COREX_EXACT_REDUCE` (default=True) | ✅ 代码引用 4 处 |
| corex_moe_weight_gather | 自研 CUDA | `BI100_MOE_COREX_WEIGHT_GATHER` (default=True) | ✅ 代码引用 4 处 |
| corex_moe_topk_softmax | 自研 CUDA | `BI100_MOE_COREX_TOPK_SOFTMAX` (default=True) | ✅ yaml 已开 |
| corex_moe_index_combine | 自研 CUDA | `BI100_MOE_COREX_INDEX_COMBINE` (default=True) | ✅ 代码引用 4 处 |
| xllm_moe | 搬自 xllm upstream | `BI100_MOE_XLLM` (default=True) | ✅ 代码引用 7 处 |
| xllm_activation | 搬自 xllm upstream | 无直接 env | ❌ 编了但没接入 |
| xllm_norm | 搬自 xllm upstream | 无直接 env | ❌ 编了但没接入 |

### B. 已编译但未接入（7 个） — 需要修复

| .so | 来源 | 提供的函数 | 为什么没接入 | 接入方案 |
|-----|------|-----------|------------|---------|
| **ix_full_bridge** | ix_full_bridge.cpp → ixformer::infer | silu_and_mul, rms_norm, fused_add_rms_norm, ix_linear, ix_linear_ex | qwen3_5.py 没有 import | patch_vllm_ops.py 已写好（最新 commit），通过 ix_startup_patch.py 自动 hook |
| **xllm_activation** | xllm activation.cu | silu_and_mul, gelu_and_mul, act_and_mul | 与 _custom_ops→ixf_F 冗余 | 作为 backup，当 ixf_F 不可用时走 xllm kernel |
| **xllm_norm** | xllm norm.cu | rms_norm, fused_add_rms_norm | 与 _custom_ops→ixf_F 冗余 | 同上 |
| **xllm_rope** | xllm rope.cu | rotary_embedding | 与 _custom_ops→ixf_F 冗余 | 同上 |
| **xllm_cache** | xllm reshape_paged_cache.cu | reshape_paged_cache | paged_attn.py 没有调用 | 需要在 cache 写入路径接入 |
| **corex_fused_paged_prefill** | 自研 CUDA | fused prefill attention | paged_attn.py 有代码但 env 没开 | computility-run.yaml 加 `BI100_ATTN_COREX_FUSED_PAGED_PREFILL=1` |
| **corex_paged_kv_gather** | 自研 CUDA | paged KV gather | paged_attn.py 有代码但 env 没开 | 同上 |
| **corex_block_major_kv_transfer** | 自研 CUDA | block-major KV copy | 完全没有调用点 | 需要在 worker/cache_engine 接入 |

## 2. 需要从 upstream 搬过来编译的代码

### 来源: upstream_ref/xllm/xllm/core/kernels/cuda/

| 文件 | 功能 | 对应 .so | 优先级 |
|------|------|---------|--------|
| xattention/decoder_reshape_and_cache.cu | fused KV cache write | xllm_xattn_cache | P0 |
| xattention/prefill_reshape_and_cache.cu | prefill cache write | xllm_xattn_cache | P0 |
| xattention/cache_select.cu | cache select | xllm_xattn_cache | P1 |
| xattention/lse_combine.cu | LSE combine | xllm_xattn_cache | P1 |
| fused_qknorm_rope.cu | fused QK norm + RoPE | xllm_fused_qknorm_rope | P0（每层省 4 kernel launch） |
| matmul.cpp | ixformer GEMM wrapper | 已在 ilu/matmul.cpp | ✅ 已搬 |
| fp8_quant.cu | FP8 quantization | xllm_fp8 | P2 |

### 来源: upstream_ref/xllm/xllm/core/kernels/ilu/

**全部已搬到 ex_engine/xllm_kernels/ilu/**（对比确认只差 CMakeLists.txt）

### 来源: upstream_ref/ds_vllm/csrc/libtorch_stable/

| 文件 | 功能 | 可用性 |
|------|------|--------|
| attention/paged_attention_v1.cu | paged attention v1 | SM70 兼容，但依赖 vllm C++ build |
| attention/paged_attention_v2.cu | paged attention v2 | 同上 |
| layernorm_kernels.cu | RMSNorm kernel | SM70 兼容 |
| activation_kernels.cu | SiLU kernel | SM70 兼容 |
| pos_encoding_kernels.cu | RoPE kernel | SM70 兼容 |
| moe/topk_softmax_kernels.cu | topk+softmax fused | SM70 兼容 |
| moe/moe_align_sum_kernels.cu | MoE align+sum | SM70 兼容 |

## 3. ixformer::infer 可用 API（base 镜像已有）

来自 `upstream_ref/xllm_latest/core/kernels/ilu/ixformer.h`:

```
ixformer::infer::silu_and_mul(input, output)
ixformer::infer::rms_norm(input, weight, output, bias, eps)
ixformer::infer::residual_rms_norm(input, residual, weight, output, residual_out, bias, alpha, eps, is_post)
ixformer::infer::ixformer_linear(input, weight, act_type, bias, out, persistent)
ixformer::infer::ixformer_linear_ex(input, weight, bias, out)
ixformer::infer::xllm_rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
ixformer::infer::xllm_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, key_stride, value_stride)
ixformer::infer::xllm_paged_attention(out, query, key_cache, value_cache, ...)
ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables(query, key_cache, value_cache, ...)
ixformer::infer::topk_softmax(weights, indices, token_expert_indices, gating_output, renormalize)
ixformer::infer::moe_compute_token_index_api(topk_ids, src_dst, dst_src, expert_sizes, ...)
ixformer::infer::moe_expand_input(output, input, dst_to_src, src_to_dst, dst_tokens, expand_factor)
ixformer::infer::moe_w16a16_group_gemm(output, input, weights, tokens_per_experts, ...)
ixformer::infer::moe_output_reduce_sum(output, input, weight, mask, extra_residual, scaling)
```

这些函数通过 `ix_full_bridge.so` pybind11 暴露给 Python 侧。

## 4. 调用链完整性检查

### 当前断裂点:

1. **ix_full_bridge.so 的 group_gemm → MoE Python for-loop**
   - `ixformer::infer::moe_w16a16_group_gemm` 在 ix_full_bridge.so 中可用
   - 但 qwen3_5.py MoE prefill 路径 (L1813-1825) 还是 `F.linear` per-expert loop
   - 需要: ix_fused_moe.py 的 7 步 pipeline 走 group_gemm 而非 per-expert linear

2. **corex_fused_paged_prefill → paged_attn.py env 没开**
   - .so 已编译已部署
   - paged_attn.py 已有完整调用代码 (L2030)
   - computility-run.yaml 缺少 `BI100_ATTN_COREX_FUSED_PAGED_PREFILL=1`

3. **xllm_cache → reshape_and_cache 没接入**
   - base 镜像 ixformer 已有 `xllm_reshape_and_cache`
   - vllm 的 cache_ops 走的是另一条路径

## 5. 需要编出的新 .so

| 目标 .so | 源文件 | 编译方式 | 依赖 |
|---------|--------|---------|------|
| xllm_fused_qknorm_rope.so | upstream fused_qknorm_rope.cu + bind | corex clang --cuda-gpu-arch=ivcore10 | libcudart, torch |
| xllm_xattn_cache.so | upstream xattention/*.cu + bind | 同上 | 同上 |

## 6. computility-run.yaml 需要补全的 env

```yaml
- name: BI100_ATTN_COREX_FUSED_PAGED_PREFILL
  value: '1'
- name: BI100_ATTN_COREX_PAGED_KV_GATHER
  value: '1'  
- name: IX_OPS_AUTO_PATCH
  value: '1'
- name: PYTORCH_CUDA_ALLOC_CONF
  value: 'expandable_segments:True'
```
