# 系统开发状态分析 — 基于 comp 168 日志 AST 链条

## 日志分析: 两次运行对比

### 运行1: 基础镜像原生 (07-23, Sub168) — ✅ 正常
```
AST调用链条 (真机上确实在调用):
  corex_gdn.py:56   → dlopen /usr/local/corex/lib64/libcorex_gdn.so ✅
  corex_gdn.py:228  → GDN prefill fused kernel ✅
  corex_gdn.py:138  → GDN decode fused kernel ✅
  corex_moe.py:339  → MoE prefill: expert-grouped-wmma ✅
  corex_moe.py:249  → MoE decode fused ✅
  corex_fa2.py:333  → FA2 packed prefill (B=2 Hq=4 Hkv=1 D=256) ✅
  corex_fa2.py:507  → FA2 paged chunked prefill ✅
  corex_fa2.py:225  → FA2 paged decode (partition=256) ✅

结果: generation throughput ~22 tokens/s, 无NaN, 无OOM
```

### 运行2: 我们的Docker (08-07, Sub508) — ❌ 失败
```
问题链条:
  max_model_len=100000 (yaml未生效! 应为80000)
  max_num_seqs=1 (yaml未生效! 应为2)
  qwen3_5.py NaN: GDN layer 0 frac=0.9998, layer 1-4 同样
  _custom_ops.py topk_softmax: module 'ixformer.functions' has no attribute 'vllm_moe_topk_softmax' × 500+
  MoE falling back to pure PyTorch experts permanently
  OOM crash at 03:51 → 引擎死亡

结果: 功能测试大量失败, 最终OOM崩溃
```

## 关键发现: 三个dlopen链条 (来自 comp 168 真机证据)

### 1. libcorex_gdn.so — GDN decode/prefill
- 路径: `/usr/local/corex/lib64/libcorex_gdn.so`
- 调用者: `corex_gdn.py` (我们已有, 246行)
- 状态: 我们的corex_gdn.py已部署, 但qwen3_5.py的GDN数学有NaN
- 需要: 修复qwen3_5.py中GDN的fp32 accumulation

### 2. ixformer MoE pipeline — 7步fused MoE
- 路径: 基础镜像 `/usr/local/corex/lib/python3/dist-packages/ixformer/`
- 调用者: `corex_moe.py` (我们已有, 237行)
- 7步: topk_softmax → gen_idx → expand → group_gemm(w13) → silu_mul → group_gemm(w2) → combine
- 状态: Python binding `ixf_F.vllm_moe_topk_softmax` 不存在
- 但C++层 `ixformer::infer::topk_softmax` 在 libixformer.so 中 **存在**
- 需要: ix_bridge.cpp 需要编译, 让Python能调到C++层的MoE函数

### 3. ixformer FA2 — FlashAttention2 三模式
- 路径: `ixformer.contrib.vllm_flash_attn` (Python, 基础镜像自带)
- 调用者: `corex_fa2.py` (我们已有, 279行)
- 状态: corex_fa2.py **没有被部署**, 也**没有被qwen3_5.py调用**
- 基础镜像的qwen3_5.py直接调corex_fa2, 但我们替换了qwen3_5.py后,
  attention走的是vllm内置Attention → xformers后端
- 需要: 把corex_fa2.py也部署, 并在qwen3_5.py的Qwen3_5FullAttention中
  优先走CoreX FA2 (三模式dispatch)

## upstream_ref 代码搬运状态

### 已搬运 (接口完全对齐):
| 源文件 | 目标 | 行数 | 状态 |
|--------|------|------|------|
| xllm/core/kernels/ilu/ixformer.h | ex_engine/include/ixformer.h | 147 | ✅ 完全一致 |
| xllm/core/kernels/ilu/ilu_ops_api.h | ex_engine/include/ilu_ops_api.h | 153 | ✅ 完全一致 |
| xllm/core/kernels/ilu/utils.h | ex_engine/include/ilu_utils.h | 62 | ✅ 完全一致 |
| xllm/core/kernels/ilu/fused_moe.cpp | ex_engine/csrc/ilu_kernel_fused_moe.cpp | 99 | ✅ 完全一致 |
| xllm/core/kernels/ilu/attention.cpp | ex_engine/csrc/ilu_kernel_attention.cpp | 162 | ✅ 完全一致 |
| xllm/core/kernels/ilu/activation.cpp | ex_engine/csrc/ilu_kernel_activation.cpp | 32 | ✅ 完全一致 |
| xllm/core/kernels/ilu/group_gemm.cpp | ex_engine/csrc/ilu_kernel_group_gemm.cpp | 39 | ✅ 完全一致 |
| xllm/core/kernels/ilu/matmul.cpp | ex_engine/csrc/ilu_kernel_matmul.cpp | 73 | ✅ 完全一致 |
| xllm/core/kernels/ilu/norm.cpp | ex_engine/csrc/ilu_kernel_norm.cpp | 50 | ✅ 完全一致 |
| xllm/core/kernels/ilu/rope.cpp | ex_engine/csrc/ilu_kernel_rope.cpp | 31 | ✅ 完全一致 |
| xllm/core/layers/ilu/fused_moe.cpp | ex_engine/csrc/ilu_layer_fused_moe.cpp | 797 | ✅ 完全一致 |
| xllm/core/layers/ilu/attention.cpp | ex_engine/csrc/ilu_layer_attention.cpp | 189 | ✅ 完全一致 |

### 未搬运 (需要搬运):
| 源文件 | 行数 | 用途 |
|--------|------|------|
| xllm/core/layers/ilu/fused_moe.h | 131 | MoE层头文件 |
| xllm/core/layers/ilu/attention.h | 82 | Attention层头文件 |

## 代码量统计
- 我们的代码(排除upstream/cccl/vllm): 130文件, 45,103行
- 已从upstream搬运的ILU代码: 2,047行 (接口完全对齐)
- 总代码量充足

## 立即行动项 (不需要思考, 直接写代码)

### P0: 修复 computility-run.yaml 参数不生效问题
Aug 7日志显示 max_model_len=100000, 但yaml写的80000。
需要确认yaml格式正确, enable_chunked_prefill要显式写。

### P1: 部署 corex_fa2.py 并接入 qwen3_5.py
comp 168日志证明FA2三模式dispatch是真机上跑的。
我们的qwen3_5.py替换了base的, 但丢失了FA2调用。

### P2: 搬运 fused_moe.h + attention.h (2个文件)
upstream_ref中最后2个未搬运的头文件。

### P3: 确认可提交
Dockerfile + computility-run.yaml + patch_ops.sh 链路完整。
