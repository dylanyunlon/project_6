# EX_ENGINE 基建盘点

日期: 2026-08-17
基于: commit 512f384a (CUTLASS Cu10 grouped GEMM 真机验证) + 后续 revert

---

## 一、Docker容器里实际运行的状态

### 进了Docker且工作正常的 (prebuilt .so)

| .so 文件 | qwen3_5.py flag | 状态 |
|---|---|---|
| corex_gdn_causal_conv.so | _USE_COREX_GDN_CAUSAL_CONV | ✓ 工作 |
| corex_gdn_gated_norm.so | _USE_COREX_GDN_GATED_NORM | ✓ 工作 |
| corex_gdn_beta_decay.so | _USE_COREX_GDN_BETA_DECAY | ✓ 工作 |
| corex_gdn_qk_map.so | _USE_COREX_GDN_QK_MAP | ✓ 工作 |
| corex_gdn_packed_decode.so | _USE_COREX_GDN_PACKED_DECODE | ✓ 工作 |
| corex_gdn_chunk_recurrent.so | _HAS_COREX_GDN_CHUNK | ✓ 工作 |
| corex_moe_topk_softmax.so | _USE_COREX_MOE_TOPK_SOFTMAX | ✓ 工作 |
| corex_moe_direct_routed.so | _USE_COREX_MOE_DIRECT_ROUTED | ✓ 工作 |
| corex_moe_weight_gather.so | _USE_COREX_MOE_WEIGHT_GATHER | ✓ 工作 |
| corex_moe_exact_reduce.so | _USE_COREX_MOE_EXACT_REDUCE | ✓ 工作 |
| corex_moe_index_combine.so | _USE_COREX_MOE_INDEX_COMBINE | ✓ 工作 |
| corex_attn_head_rms_norm.so | _USE_COREX_ATTN_HEAD_RMS_NORM | ✓ 工作 |
| corex_block_major_kv_transfer.so | (block_major_kv_cache.py用) | ✓ 工作 |
| corex_paged_kv_gather.so | (paged_attn.py用) | ✓ 工作 |
| corex_fused_paged_prefill.so | (corex_fa2用) | ✓ 工作 |
| xllm_moe.so | _USE_XLLM_MOE | ✓ 工作 |
| xllm_activation.so | (patch_vllm_ops用) | ? 见下 |
| xllm_norm.so | (patch_vllm_ops用) | ? 见下 |
| xllm_rope.so | (patch_vllm_ops用) | ? 见下 |
| xllm_cache.so | (patch_vllm_ops用) | ? 见下 |
| ix_full_bridge.so | (ix_ops.py用) | ? 见下 |

### 进了Docker但断裂的

| 文件 | 问题 |
|---|---|
| ix_full_bridge.so (v1) | 已cp到$VLLM_ROOT/, 但python wrapper ix_ops.py没部署 |
| xllm_activation/norm/rope/cache.so | 已cp到$VLLM_ROOT/, 但patch_vllm_ops.py没部署，没hook |
| ix_fused_moe.py | 已cp到models/, 但找不到ix_moe_bridge.so → _HAS_IX_FUSED_MOE=False |

### 没进Docker的关键文件

| 文件 | 功能 | 行数 |
|---|---|---|
| ex_engine/python/ix_ops.py | ix_full_bridge.so的Python wrapper | 343 |
| ex_engine/python/ix_ops_dispatch.py | 统一op dispatch (bridge→ixformer→raise) | 407 |
| ex_engine/python/patch_vllm_ops.py | monkey-patch vllm的silu/rms_norm/rope/cache | 201 |
| ex_engine/python/corex_moe.py | corex MoE pipeline wrapper | 237 |
| ex_engine/python/corex_gdn.py | corex GDN ops wrapper | 256 |
| ex_engine/python/corex_fa2.py | corex FlashAttn dispatch | 279 |
| ex_engine/python/corex_fa2_dispatch.py | FA2 3-mode dispatch | 231 |
| ex_engine/python/fused_moe_ilu.py | 7-step MoE pipeline (Python) | 205 |
| ex_engine/python/gemm_dispatch.py | GEMM dispatch (cutlass/cuinfer/torch) | 180 |
| ex_engine/csrc/gemm_grouped.cu | ✓ 真机验证的CUTLASS grouped GEMM | 188 |
| ex_engine/csrc/gemm_grouped_bind.cpp | pybind11 binding | 182 |
| ex_engine/build_gemm_grouped.sh | 编译脚本 | — |
| ex_engine/xllm_kernels/cuda/corex_batched_gemm_kernel.cu | CUTLASS batched GEMM | 67 |
| ex_engine/xllm_kernels/cuda/bindings/corex_batched_gemm_bind.cpp | binding | 129 |

---

## 二、两个路径断裂的根因

### 断裂1: patch_ops.sh 找不到 ex_engine/python/

patch_ops.sh 第204行:
```bash
EX_ENGINE_DIR="$(cd "$(dirname "$0")/../ex_engine" 2>/dev/null && pwd || echo "")"
```

Docker容器里的目录结构:
```
/workspace/
├── qwen3_6_scripts/          ← patch_ops.sh 在这里
│   ├── patch_ops.sh
│   ├── ex_engine_src/        ← Dockerfile COPY进来的（只有5个文件）
│   │   ├── csrc/moe_ops_impl.cu
│   │   ├── csrc/ix_full_bridge_v2.cpp
│   │   ├── build_moe_bridge.sh
│   │   └── python/moe_dispatch.py, patch_moe_hot_path.py
│   └── prebuilt/corex-3.2.3-ivcore10/*.so
└── (没有 ex_engine/ 目录)
```

`$(dirname "$0")/../ex_engine` = `/workspace/ex_engine` → **不存在**

结果: ix_ops.py, patch_vllm_ops.py, ix_startup_patch.py 全部没部署。
xllm_activation/norm/rope/cache.so 虽然被cp到$VLLM_ROOT/但没有Python层调用它们。

### 断裂2: build_moe_bridge.sh 内部路径错误

build_moe_bridge.sh 第18-19行:
```bash
MOE_CU="${SCRIPT_DIR}/ex_engine/csrc/moe_ops_impl.cu"
BRIDGE_CPP="${SCRIPT_DIR}/ex_engine/csrc/ix_full_bridge_v2.cpp"
```

SCRIPT_DIR = `/workspace/qwen3_6_scripts/ex_engine_src`
实际路径 = `${SCRIPT_DIR}/csrc/moe_ops_impl.cu`（少了 `ex_engine/` 一层）

结果: ix_moe_bridge.so 编译失败 → _HAS_IX_FUSED_MOE=False → 7-step fused MoE pipeline 未启用

---

## 三、ex_engine/ 文件去重审计

### 重复实现的功能（同一功能多个文件）

**MoE topk softmax (5个文件做同一件事)**:
1. `xllm_kernels/cuda/moe/moe_topk_softmax_kernels.cuh` (866行) ← xllm上游原版
2. `csrc/moe/moe_topk_softmax_kernels.cuh` (855行) ← 几乎相同的拷贝
3. `csrc/factor_moe_topk_softmax.cu` (260行) ← 独立提取版
4. `csrc/moe/moe_topk_softmax_ext.cu` (55行) ← 另一个入口
5. `csrc/moe_topk_softmax_v3.cu` (143行) ← 又一个版本
6. prebuilt `corex_moe_topk_softmax.so` ← 已编译可用
7. prebuilt `xllm_moe.so` ← 也包含此功能

**MoE combine (3个文件)**:
1. `xllm_kernels/cuda/moe/moe_combine.cu` (105行) ← xllm上游原版
2. files_5 的 `factor_moe_combine.cu` ← 重写
3. prebuilt `xllm_moe.so` ← 已编译可用

**MoE compute_index (2个文件)**:
1. `xllm_kernels/cuda/moe/moe_compute_index.cu` (156行) ← xllm上游原版
2. files_5 的 `factor_moe_compute_index.cu` ← 重写

**MoE Python pipeline (3个文件)**:
1. `python/fused_moe_ilu.py` (205行)
2. `python/moe_dispatch.py` (171行)
3. files_5 的 `moe_pipeline.py` ← 重写

**Attention dispatch (2个文件)**:
1. `python/corex_fa2_dispatch.py` (231行)
2. files_5 的 `attn_dispatch.py` ← 重写

**C++ bridge (3个文件)**:
1. `csrc/ix_full_bridge.cpp` (90行) ← v1, 对应prebuilt ix_full_bridge.so
2. `csrc/ix_full_bridge_v2.cpp` (387行) ← v2, 没编译
3. `csrc/ix_moe_bridge.cpp` (261行) ← MoE专用, 没编译

**ILU层 fused_moe (2处)**:
1. `xllm_layers/ilu/fused_moe.cpp` (797行) ← xllm上游
2. `csrc/ilu_layer_fused_moe.cpp` (797行) ← 拷贝

### 上游cat但未修改的文件

| 目录 | 文件数 | 来源 |
|---|---|---|
| xllm_kernels/ilu/*.cpp | 7 | xllm上游ILU kernel接口 |
| xllm_layers/ilu/*.cpp | 2 | xllm上游ILU layer |
| xllm_layers/common/*.cpp | 5 | xllm上游common layer |
| xllm_layers/npu_torch/*.cpp | 10 | xllm上游NPU实现（不适用BI-V100） |
| xllm_layers/mlu/*.cpp | 4 | xllm上游MLU实现（不适用BI-V100） |
| xllm_models/*.h | 6 | xllm上游model定义 |
| moe/*.py | 14 | ds_vllm上游MoE模块 |
| fla_kernels/ | 6 | FLA库GDN kernel（Triton，BI-V100不能跑） |

---

## 四、真机验证状态

| 组件 | commit | 真机结果 |
|---|---|---|
| gemm_grouped.cu (CUTLASS Cu10 TN) | cdcf1150 | ✓ err=0.000015 PASS, 1.97x vs torch.mm |
| corex_batched_gemm_kernel.cu | (sub 655用过) | ✓ decode单token 2.462ms |
| 16个prebuilt .so | (sub 694在用) | ✓ 正常加载 |
| ix_moe_bridge.so | 未编译 | ✗ 路径断裂 |
| ix_full_bridge_v2.so | 未编译 | ✗ 未进Docker |
| xllm_moe.so 的 fused_topk | (sub 694在用) | ✓ 正常工作 |
