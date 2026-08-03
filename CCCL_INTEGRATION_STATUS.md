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
| gen_patch.py | muh/gen_patch.py | ~200 | ✓ 可提取 bi100_* | C++ header → vllm unified diff |
| extract.py | muh/extract.py | ~100 | ✓ 完成 | CCCL tuning → schema 提取 |
| muh_dispatch.py | muh_dispatch.py | ~400 | △ 概念完成 | CCCL-style 类型分派（未接入 vllm）|
| muh_kernel_map.py | muh_kernel_map.py | ~350 | △ 手写常量 | 需要从 C++ headers 自动提取闭环 |
| compile_test | muh/test/*.cu + *.cpp | 2 files | ✓ 33 项通过 | C++ 编译验证 |
| test_smem_safety | muh/tests/test_smem_safety.py | 1 file | ✓ | 全算法 SMEM 安全检查 |

---

## 三、Decode 热路径 × 资产覆盖矩阵

```
算法              CCCL  muh   schema bench test  vllm注入点                         竞赛权重
───────────────── ───── ───── ────── ───── ───── ──────────────────────────────── ─────────
reduce            ✓     ✓     ✓      ✓     ✓     csrc/attention/paged_attention   83% (Output)
scan              ✓     ✓     ✓      ✓     ✓     csrc/attention/paged_attention   14% (Input)
topk              ✓     ✓     ✓      ✓     ✓     csrc/sampling/sampling_kernels   per decode
radix_sort        ✓     ✓     ✓      ✓     ✓     csrc/sampling/sampling_kernels   per decode
transform         ✓     ✓     ✓      ✓     ✓     csrc/activation/layernorm/rope   200×/token
select_if         ✓     ✓     ✓      △     ✓     csrc/sampling (top-p filter)     per decode
batch_memcpy      ✓     ✓     ✓      △     △     csrc/cache_kernels               3% (Cache)
for_each          ✓     ✓     ✓      ✓     ✓     csrc (residual connections)      per layer
```

△ = benchmark/test 文件存在但名称不直接匹配（partition/if.cu 对应 select_if，copy/memcpy.cu 对应 batch_memcpy）

---

## 四、GitHub Issues 状态

### 已创建的 38 个 Issues（全 open，全有 labels）

**功能测试覆盖（#1-#16）**: 竞赛 50+ 功能测试用例的完整 PRD，每个含 PND 级 test cases 表

| 编号范围 | 前缀 | 数量 | 说明 |
|----------|------|------|------|
| #1-#14 | [FEA] | 14 | 功能测试: 非流式/流式/Tool/Reasoning/Cache/采样/结构化/多语言/多模态/校验/能力/截断/效果 |
| #15-#16 | [EPIC] | 2 | 性能基准 + 开发环境 |
| #17-#25 | [FEA]/[EPIC] | 9 | muh 语言设计: 语法/schema/codegen(yaml+patch+dockerfile)/bench/search/tuning提取 |
| #26-#38 | [muh] | 13 | muh 算法标定: reduce/scan/radix_sort/select_if/scan_by_key/reduce_by_key/unique_by_key/transform/batch_memcpy/topk + gen_patch管道/benchmark runner/hardware校准 |

### Project/6 面板上的 Draft Issues（72 个，无 repo 关联）

来自后续对话生成，包含：
- [muh] 语言规范 v1/v2
- [muh] 20+ 个算法标定 items（adjacent_difference, batched_topk, find, histogram, merge, rle_encode 等）
- [INFRA] CI同步/Build编译/Deploy部署/Verify回归
- [BUG] scale_mem_bound / gen_patch 管道 / select_if 坍缩 / bytes_in_flight / reduce items
- [CCCL-verify] 20 个 Thrust/CUB example 验证 items
- [CCCL-test] 10 个 Catch2 测试矩阵 items
- [muh-bench] 6 个 benchmark items
- [muh-pipe] 端到端管道验证

**这 72 个 draft 需要转为真 issue。** 内容已经写好（body 含完整 test cases 表），只是缺少 repo 关联和 labels。

---

## 五、关键发现（SM count = 16）

Phanthy Cloud 实测确认 BI-V100 只有 **16 SMs**（不是规格书的 50c）。

影响范围：
1. `hardware.cuh` — 已修正 sm_count=16
2. `tuning_transform.cuh` — bytes_in_flight 基于 900/50=18 GB/s 已失效，应为 900/16=56 GB/s
3. `tuning_reduce.cuh` — bi100_det_* 和 bi100_default 的 items 偏小（tile 仅用 23% SMEM）
4. `tuning_scan.cuh` — lookback delay 基于 50 SM 的争用模型，16 SM 下争用更低、delay 可以更短
5. 所有 benchmark 理论推导需要重跑

---

## 六、不需要 clone 更多 CCCL 的原因

完整 CCCL (github.com/NVIDIA/cccl) ≈ 40K 文件、1.2GB。我们有 8,900 文件 (74MB)。

已有的关键子集：
- ✓ 全部 27 tuning headers（muh 从这里提取参数空间）
- ✓ 全部 32 dispatch headers（tuning 参数化的对象）
- ✓ 52 Thrust examples（正确性验证的 golden reference）
- ✓ 18 CUB examples（device + block level API 验证）
- ✓ 234 CUB Catch2 tests（回归测试矩阵）
- ✓ 169 Thrust tests（Thrust 算法回归）
- ✓ 78 CUB benchmarks（标定数据的来源）
- ✓ 530 Thrust headers + 1,357 libcudacxx headers（编译依赖）

缺失的 ~31K 文件：
- libcudacxx 深层 include（6K）— 编译时用 -I 指向安装路径
- cudax 实验模块（800）— 竞赛不用
- cmake/CI 基础设施（5K）— 平台用 Dockerfile 构建
- Python 绑定 / 文档 / 其他（19K）— 不相关

---

## 七、信创魔盒核心差异（竞赛定位）

> "信创魔盒是基于系统级的架构，内置算法因子，用 EngineX 引擎把模型内部的算法因子重新置换——不是单纯的连接器。"

muh 在这个架构中的角色：
- CCCL 的 `policy_selector` 是 NVIDIA 为自家 GPU 写的"算法因子"
- muh 的 `policy_selector` 是为天垓100 写的等效"算法因子"
- EngineX 把 CCCL 的 NVIDIA 算法因子替换成 muh 的天垓100 算法因子
- 不是适配层（60% 精度），是置换层（目标 ≥100% 精度在天垓100 硬件约束下的最优解）

竞赛成绩 = 算法因子置换的精度 × 硬件实测标定的覆盖度。
