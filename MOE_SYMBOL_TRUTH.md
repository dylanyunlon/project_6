# MoE 函数符号真相 (2026-08-17 确认)

## 结论

那5个 MoE 函数**确实不在任何镜像预装的 .so 里**。另一位开发者说的是对的。

但它们也**不需要**在预装 .so 里——它们是自编译的。

## 5个函数的正确命名空间

```
ixformer::infer::topk_softmax
ixformer::infer::moe_compute_token_index_api
ixformer::infer::moe_expand_input
ixformer::infer::moe_w16a16_group_gemm
ixformer::infer::moe_output_reduce_sum
```

**注意**: 是 `ixformer::infer`，不是 `ixformer::kernels::infer`。

## 声明 vs 实现的关系

| 位置 | 角色 |
|------|------|
| `ixformer_sdk/csrc/include/ixformer/kernels/kernels.h` | **头文件声明** (namespace `ixformer::kernels::infer`) — C++ 模板声明，给 SDK 用的 |
| `ex_engine/csrc/moe_ops_impl.cu` | **CUDA 实现** (namespace `ixformer::infer`) — 自己写的 kernel，不依赖任何 .so |
| `ex_engine/csrc/ix_full_bridge_v2.cpp` | **pybind11 桥** — forward-declare 然后调用 moe_ops_impl.cu 里的实现 |
| `ex_engine/build_moe_bridge.sh` | **构建脚本** — 把 v2.cpp + moe_ops_impl.cu 一起编译成 ix_full_bridge_v2.so |

## 符号表搜索结果 (4个 .so 全部搜过)

| .so 文件 | MoE 函数 | 结论 |
|----------|----------|------|
| `libixformer.so` (3937 symbols) | 无 topk_softmax/moe_compute_token_index 等 | 只有 `reduce_sum` (通用的) |
| `_ixformer_torch.so` (49 symbols) | 完全没有 MoE | 只有 norm/rope/cache/attn |
| `_C.so` (6 symbols) | 几乎空壳 | 只有 PyInit |
| `libcuinfer.so` (270 symbols) | 只有 cuinferTopK (不是 MoE 的) | GEMM/BLAS 级别 |

## 构建链

```
patch_ops.sh
  └→ build_moe_bridge.sh
       └→ ninja/CppExtension 编译:
            ix_full_bridge_v2.cpp + moe_ops_impl.cu
            → ix_full_bridge_v2.so (包含5个MoE函数的实现)
```

## `ixformer::kernels::infer` vs `ixformer::infer` 的区别

- `ixformer::kernels::infer` — SDK 头文件 (kernels.h) 中的声明，使用 raw pointer + cudaStream_t 
  - 例: `void moe_topk_softmax(const T *gating_output, T *topk_weights, int *topk_indices, ...)`
- `ixformer::infer` — 我们自己实现的 PyTorch wrapper，使用 torch::Tensor
  - 例: `void topk_softmax(torch::Tensor& topk_weights, torch::Tensor& topk_indices, ...)`

`moe_ops_impl.cu` 是直接写 CUDA kernel（不调用 kernels.h 模板），然后暴露 Tensor API。

## Python 调用链

```python
# 通过 ixformer SDK (需要真机上的 _C.so 包含 infer 子模块):
import ixformer._C as ops
ops.infer.moe_topk_softmax(...)  # 如果 _C.so 有实现

# 通过 ex_engine bridge (我们自编译的):
import ix_full_bridge_v2 as bridge
bridge.topk_softmax(...)  # 来自 moe_ops_impl.cu
```
