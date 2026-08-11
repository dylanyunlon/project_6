# comp 168 Docker 诊断 → .so 开发清单

> 基于 `2d5232c5d6bc` (comp 168 docker log, 3786 行)
> 当前 HEAD: `b25fc53e` (414 commits)

## 一、comp 168 日志三大致命问题

| # | 错误 | 出现次数 | 根因 | 状态 |
|---|------|----------|------|------|
| 1 | `GDN NaN frac=0.9998` | 16次(layer 0-4) | 我们的 GDN prefill 实现产生 NaN → replace with zeros → 模型质量归零 | **P0 未修** |
| 2 | `vllm_moe_topk_softmax not found` | 39次 | `ixformer.functions` 没有 Python binding → fallback to Python for 循环 | **P0 需 .so** |
| 3 | `CUDA OOM 32 MiB` | 17次 | `max_model_len=100000` 超过 KV cache 容量 → engine 死亡 | ✅ 已修为 80000 |

## 二、真机探测确认的事实

从你贴的真机 probe 输出：

```
ixformer.functions 有:
  ✓ silu_and_mul, rms_norm, fused_add_rms_norm, rotary_embedding
  ✓ flash_attn_*, vllm_single_query_cached_kv_attention_v2
  ✓ vllm_cache_ops_reshape_and_cache, vllm_swap_blocks, vllm_copy_cache
  ✗ vllm_moe_topk_softmax (不存在!)
  ✗ moe_compute_token_index_api (不存在!)
  ✗ moe_w16a16_group_gemm (不存在!)

libixformer.so 中:
  ✓ 上述函数全部存在 (C++ 符号, xllm 的 ixformer.h 声明了它们)
  但 Python binding (_C.so) 没有暴露
```

**结论**: MoE 7 步 pipeline 中的 topk_softmax / gen_idx / expand / group_gemm / combine 全部需要通过 `ix_moe_bridge.so` 桥接。

## 三、需要开发/修复的 .so 清单

### SO-1: `ix_moe_bridge.so` (MoE 7步 pipeline) — ✅ 代码已有，需真机编译验证

**源码**: `ex_engine/csrc/ix_moe_bridge.cpp` (258行)
**编译**: `ex_engine/precompile_ix_bridge.py` → `torch.utils.cpp_extension.load(-lixformer)`
**状态**: 代码写好了，Dockerfile 有 build step，但从未在真机验证过编译成功

真机验证命令:
```bash
cd /workspace/ex_engine
python3 precompile_ix_bridge.py
ls -la build/ix_moe_bridge*.so
python3 -c "import torch; from torch.utils.cpp_extension import load; m=load('test', sources=['csrc/ix_moe_bridge.cpp'], extra_ldflags=['-L/usr/local/corex/lib64/python3/dist-packages/ixformer', '-lixformer']); print(dir(m))"
```

### SO-2: GDN prefill 修复 — **P0 最高优先级**

**现状**: 我们的 `_torch_chunk_gated_delta_rule` 在 fp16 下产生 99.98% NaN
**参考**: `upstream_ref/xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` (576行)

关键差异:
- xllm 用 `fp32` accumulation: `decay_mask = ... .exp().float()` 
- xllm 用 `torch::matmul` 而不是自定义 chunk kernel
- xllm 的 recurrent state 管理有精确的 `clamp(-20, 20)` 限制

**解决方案**: 不写新 .so，而是从 xllm 搬运 GDN 的 PyTorch 实现（C++ torch ops, 全 fp32 accumulation），替换我们的 chunk kernel。

### SO-3: `_custom_ops.py` patch — ✅ 已有 fallback 逻辑

base image 的 `_custom_ops.py` 调用 `ixf_F.vllm_moe_topk_softmax` 时会报错。
但 comp 168 的 base 镜像绕过了 `_custom_ops`，直接走 `corex_moe.py` 的 7 步 pipeline。

**如果 base 有 corex_moe.py**: 不需要 patch
**如果 base 没有 corex_moe.py**: 我们的版本 + ix_moe_bridge.so 补位

## 四、upstream 已有、不需要重写的代码

| upstream 文件 | 行数 | 我们的对应文件 | 搬运状态 |
|--------------|------|---------------|---------|
| `xllm/core/kernels/ilu/ixformer.h` | 147 | `ex_engine/csrc/ilu/ixformer.h` | ✅ 已搬 |
| `xllm/core/kernels/ilu/fused_moe.cpp` | 99 | `ex_engine/csrc/ilu_kernel_fused_moe.cpp` | ✅ 已搬 |
| `xllm/core/layers/ilu/fused_moe.cpp` | 797 | `ex_engine/csrc/ilu_layer_fused_moe.cpp` | ✅ 已搬 |
| `xllm/core/kernels/ilu/attention.cpp` | 162 | `ex_engine/csrc/ilu_kernel_attention.cpp` | ✅ 已搬 |
| `xllm/core/layers/ilu/attention.cpp` | 189 | `ex_engine/csrc/ilu_layer_attention.cpp` | ✅ 已搬 |
| `xllm/core/kernels/ilu/norm.cpp` | 50 | `ex_engine/csrc/ilu_kernel_norm.cpp` | ✅ 已搬 |
| `xllm/core/kernels/ilu/activation.cpp` | 32 | `ex_engine/csrc/ilu_kernel_activation.cpp` | ✅ 已搬 |
| `xllm/core/kernels/ilu/rope.cpp` | 31 | `ex_engine/csrc/ilu_kernel_rope.cpp` | ✅ 已搬 |
| `xllm/core/kernels/ilu/group_gemm.cpp` | 39 | `ex_engine/csrc/ilu_kernel_group_gemm.cpp` | ✅ 已搬 |
| `xllm/core/kernels/ilu/matmul.cpp` | 73 | `ex_engine/csrc/ilu_kernel_matmul.cpp` | ✅ 已搬 |
| `xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp` | 576 | `ex_engine/csrc/qwen3_gated_delta_net_base.cpp` | ✅ 已搬 |
| `ds_vllm/csrc/moe/topk_softmax_kernels.cu` | 874 | `ex_engine/csrc/moe_v055/topk_softmax_kernels.cu` | ✅ 已搬 |
| `xllm/core/kernels/cuda/moe/moe_topk_softmax_kernels.cuh` | ~400 | `ex_engine/csrc/moe/moe_topk_softmax_kernels.cuh` | ✅ 已搬 |

## 五、真机验证 checklist

在真机上按顺序执行:

```bash
# 1. 验证 ix_moe_bridge.so 编译
cd /workspace/ex_engine && python3 precompile_ix_bridge.py
ls build/ix_moe_bridge*.so   # 必须存在

# 2. 验证符号解析
python3 -c "
import torch
import importlib.util
spec = importlib.util.spec_from_file_location('ix', 'build/ix_moe_bridge.cpython-310-x86_64-linux-gnu.so')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
print([x for x in dir(m) if not x.startswith('_')])
# 应输出: ['topk_softmax', 'moe_gen_idx', 'moe_expand_input', 'moe_group_gemm', 
#          'silu_and_mul', 'moe_combine_result', 'paged_attention', 'rms_norm',
#          'fused_add_rms_norm', 'linear', 'reshape_and_cache', 'rotary_embedding']
"

# 3. 验证 topk_softmax 功能
python3 -c "
import torch
# ... load ix_moe_bridge ...
gating = torch.randn(4, 64, device='cuda', dtype=torch.float32)
tw = torch.empty(4, 8, device='cuda', dtype=torch.float32)
ti = torch.empty(4, 8, device='cuda', dtype=torch.int32)
tei = torch.empty(4, 8, device='cuda', dtype=torch.int32)
m.topk_softmax(tw, ti, tei, gating)
print('topk_weights:', tw)
print('topk_ids:', ti)
"

# 4. 验证 GDN 不再 NaN
# (需要先修复 GDN prefill 代码)

# 5. 启动服务验证
python3 -m vllm.entrypoints.openai.api_server --model /model ...
```

## 六、最关键发现：07-23 的 base image 自带完整 corex_* chain

**07-23 日志证据** (dockerrizhi.txt):
```
corex_gdn.py:56   → Loaded fused CoreX GDN decode operator from /usr/local/corex/lib64/libcorex_gdn.so ✅
corex_gdn.py:228  → Using fused CoreX GDN prefill operator ✅
corex_moe.py:339  → Using CoreX fused MoE prefill operator: tokens=4096, kernel=expert-grouped-wmma ✅
corex_fa2.py:333  → Using CoreX FA2 packed prefill: B=2 Hq=4 Hkv=1 D=256 ✅
corex_fa2.py:507  → Using CoreX paged FA2 chunked prefill ✅
```

**08-07 日志**: 零条 corex_* 加载记录。取而代之的是 `qwen3_5.py:445 NaN in prefill` + `_custom_ops.py:58 topk_softmax not found`。

**根因**: 08-07 提交部署了我们自己的 `qwen3_5.py`，覆盖了 base image 自带的版本，打断了 `corex_gdn.py` / `corex_moe.py` / `corex_fa2.py` 的调用链。

**当前状态**: `patch_ops.sh v2` 已经有条件跳过逻辑（`_QW_SIZE > 1000 → KEEPING IT`），但需要确保下次提交时不再触发 qwen3_5.py 覆盖。

**结论**: 如果 base image 有工作的 corex_* chain，我们只需要:
1. 不覆盖 qwen3_5.py
2. 只部署 serving 层（protocol/serving_chat/api_server/tool_parser）
3. `max_model_len=80000`（已修）
4. `ix_moe_bridge.so` 作为备用（如果 base 的 _custom_ops 有路径碰到 topk_softmax）

## 七、代码量评估

| 组件 | 文件数 | 总行数 | 状态 |
|------|--------|--------|------|
| ex_engine/csrc (C++) | 39 | ~8000 | 全部已有，需真机编译 |
| ex_engine/python (Python) | 7 | ~1200 | 全部已有，dispatch chain 完整 |
| qwen3_6_scripts (serving) | 20+ | ~6000 | 全部已有，patch_ops.sh 管部署 |
| upstream_ref (xllm reference) | 500+ | ~100K | 参考用，关键文件已搬到 ex_engine |

**结论**: 代码量是够的。问题不是代码不够，而是:
1. GDN NaN 没修（需要用 xllm 的 fp32 accumulation 逻辑替换）
2. ix_moe_bridge.so 从未在真机编译成功
3. 没有 "不允许 fallback" 的硬要求落实到代码里
