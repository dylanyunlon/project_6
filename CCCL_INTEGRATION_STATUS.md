# CCCL Integration Status — project_6

> **Generated**: 2026-08-02
> **Context**: CCCL as project base for ModelHub XC competition
> **Scoring**: Token吞吐加权值 = Output TPS × 16.796 (83%) + Input TPS × 2.799 (14%) + Cache TPS × 0.56 (3%)

---

## 一、CCCL 资产清单（已在仓库中）

| 类别 | 文件数 | 总行数 | 路径 | 用途 |
|------|--------|--------|------|------|
| Thrust examples | 52 | 4,582 | cccl_upstream/thrust/examples/*.cu | CCCL-verify 正确性验证 |
| CUB device examples | 14 | ~2,000 | cccl_upstream/cub/examples/device/*.cu | CCCL-verify API 验证 |
| CUB block examples | 4 | ~800 | cccl_upstream/cub/examples/block/*.cu | SMEM 边界验证 |
| CUB Catch2 tests | 234 | 70,300 | cccl_upstream/cub/test/*.cu | CCCL-test 回归矩阵 |
| Thrust tests | 169 | ~15,000 | cccl_upstream/thrust/testing/*.cu | Thrust 算法回归 |
| CUB benchmarks | 78 | ~8,000 | cccl_upstream/cub/benchmarks/bench/**/*.cu | muh-bench 标定数据源 |
| CCCL tuning headers | 27 | 17,000+ | cccl_upstream/cub/cub/device/dispatch/tuning/*.cuh | 参数空间定义（NVIDIA 原版）|
| CCCL dispatch headers | 32 | ~12,000 | cccl_upstream/cub/cub/device/dispatch/*.cuh | 算法调度逻辑 |
| Thrust include headers | 530 | ~40,000 | cccl_upstream/thrust/thrust/**/*.h | 编译依赖 |
| libcudacxx headers | 1,357 | ~80,000 | cccl_upstream/libcudacxx/include/**/* | 编译依赖 |
| **总计** | **~8,900** | **~250,000** | cccl_upstream/ (74MB) | |

**不需要 clone 更多。** 剩余 ~31K 文件是 cmake 脚手架、CI 配置、Python 绑定、cudax 实验模块。竞赛所需的全部代码已在仓库中。

---

## 二、muh 工具链状态

| 组件 | 文件 | 行数 | 状态 | 说明 |
|------|------|------|------|------|
| C++ tuning headers | muh/include/muh/tuning/*.cuh | 2,211 | ✓ 27/27 完成 | 所有 CUB 算法有 BI-V100 等效 policy_selector |
| hardware.cuh | muh/include/muh/hardware.cuh | 65 | ✓ SM=16 已修正 | bi_v100() 构造函数，sm_count=16 已确认 |
| common.cuh | muh/include/muh/tuning/common.cuh | 180 | ✓ 3 bugs 已修 | scale_mem_bound 返回顺序、上界、SMEM cap 均修正 |
| schema YAML | muh/schema/*.yaml | 27 files | ✓ 完成 | 每个算法的参数空间定义 |
| parse.py | muh/parse.py | ~150 | ✓ 基本可用 | .muh → JSON 解析（自实现 YAML parser）|
| gen_yaml.py | muh/gen_yaml.py | ~80 | ✓ 完成 | .muh → computility-run.yaml |
| extract.py | muh/extract.py | ~100 | ✓ 完成 | CCCL tuning → schema 提取 |
| muh_dispatch.py | muh_dispatch.py | ~400 | △ 概念完成 | CCCL-style 类型分派（未接入 vllm）|
| muh_kernel_map.py | muh_kernel_map.py | ~350 | △ 手写常量 | 需要从 C++ headers 自动提取闭环 |
| compile_test | muh/test/*.cu + *.cpp | 2 files | ✓ 33 项通过 | C++ 编译验证 |
| test_smem_safety | muh/tests/test_smem_safety.py | 1 file | ✓ | 全算法 SMEM 安全检查 |

---
