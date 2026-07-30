# MUH Project Checkpoint

> **最后更新**: 2026-07-30
> **GitHub Project**: github.com/users/dylanyunlon/projects/6
> **代码仓库**: github.com/dylanyunlon/project_6
> **竞赛截止**: 2026-09-30

---

## 一、项目是什么

参加信创模盒 ModelHub XC 的"模型适配引擎竞赛-第一届"。目标是优化 vllm 引擎，让 Qwen3.6-35B-A3B 在天数智芯天垓100（4×BI-V100 GPU）上跑出最高的 Token 吞吐加权值。

**计分公式**:
```
Token吞吐加权值 = Output TPS × 16.796 + Input TPS × 2.799 + Cache TPS × 0.56
```

Output TPS 权重占 83%——decode 阶段优化收益最大。

**奖项**:
- 基础奖 200,000 积分（1:1 兑现金）: 通过全部功能/效果测试 + 性能达标(≥8000)
- 高级奖 +100,000: 加权值提升 ≥ 30%
- 特级奖 +50,000: 加权值提升 ≥ 50%

## 二、竞赛测评流程

参赛者提交的是 **Git 仓库地址**（在 dev.modelhub.org.cn 上）。平台自动执行：

1. **构建镜像**: 读取仓库根目录的 `Dockerfile`，基于基础镜像 `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3` 构建
2. **启动服务**: 读取 `computility-run.yaml` 的 `command`，在 4×天垓100 容器里启动 vllm api server（模型权重平台预挂载在 `/model`）
3. **功能测试（门控）**: 50+ 个 OpenAI 兼容 API 测试用例，全部通过才进入下一步
4. **效果测试（门控）**: 标准 benchmark 偏差 ≤ ±4%
5. **性能测试（排名）**: 计算加权值

**你能改的**: Dockerfile + vllm 源码 + computility-run.yaml 启动参数。模型本身不能改。

## 三、muh 是什么

muh 是我们设计的 **tuning DSL（领域特定语言）**，用于：

1. 把 CCCL 的 tuning pattern（block_threads / items_per_thread / load_algorithm / cache_modifier 等）抽象成硬件无关的参数空间
2. 针对天垓100 的硬件特性搜索最优参数组合
3. Codegen 输出实际的 vllm kernel 修改 + computility-run.yaml + Dockerfile

**为什么需要它**: CCCL 有 27 个 tuning_*.cuh 文件（17000+ 行），每个算法都有针对不同 NVIDIA SM 架构的特化参数。天垓100 不是 NVIDIA GPU，不能直接用这些参数，但 tuning 的维度（block size、warp 策略、shared memory 用量、prefetch 策略）是通用的。muh 让迁移过程变成"改配置 + 跑 benchmark"而不是"手改 kernel + 祈祷"。

**muh 的状态**: PRD 设计阶段，还没有代码。

## 四、已完成的工作

### 4.1 Project 6 已有 16 个真实 GitHub Issue（不是 Draft）

都在 `dylanyunlon/project_6` 仓库里，已关联到 GitHub Project 6，有 label 和 Priority：

| # | 标题 | Labels | Priority |
|---|------|--------|----------|
| 1 | [FEA] 非流式基础对话 | 基本功能,vllm,天垓100,Qwen3.6 | P0 |
| 2 | [FEA] 流式对话 SSE | 基本功能,vllm | P0 |
| 3 | [FEA] Tool Calling | 基本功能,vllm,Qwen3.6 | P0 |
| 4 | [FEA] Reasoning/Thinking 分离 | 基本功能,thinking,Qwen3.6 | P0 |
| 5 | [FEA] Prefix Cache | 基本功能,性能测试,vllm | P0 |
| 6 | [FEA] 采样参数边界 | 采样参数,vllm | P1 |
| 7 | [FEA] max_tokens 边界 | max_tokens,vllm | P1 |
| 8 | [FEA] 结构化输出 | 结构化输出,vllm | P0 |
| 9 | [FEA] 多语言 Emoji | 多语言,Qwen3.6 | P1 |
| 10 | [FEA] 多模态 base64 PNG | 多模态,基本功能,Qwen3.6 | P0 |
| 11 | [FEA] 参数校验 | 参数校验,vllm | P1 |
| 12 | [FEA] 基础能力 | 基础能力,vllm,Qwen3.6 | P0 |
| 13 | [FEA] 输出截断 | 截断测试,vllm | P1 |
| 14 | [FEA] 效果测试 | 效果测试,Qwen3.6,天垓100 | P0 |
| 15 | [EPIC] 性能基准 | 性能测试,天垓100,vllm | P0 |
| 16 | [EPIC] 开发环境与代码提交 | infra,天垓100 | P1 |

这 16 个覆盖了竞赛功能测试的所有 50+ 用例。每个 issue 的 body 里都有 PND 级别的测试用例表（前置条件 + 原子步骤 + 二值判定标准）。

### 4.2 仓库里已有 NVIDIA CCCL 代码

`project_6/cccl_upstream/` 目录下包含完整的 CCCL：
- `cub/` — GPU 原语（reduce, scan, sort, topk, block/warp/device 三层）
- `thrust/` — 高层算法 + 60 个示例
- `libcudacxx/` — CUDA C++ 标准库
- `cudax/` — 实验性功能（allocators, memory resources）
- `cub/cub/device/dispatch/tuning/` — 27 个硬件特化 tuning 文件（17000+ 行）

### 4.3 Label 体系已建立

仓库上已创建 16 个 label：基本功能、thinking、采样参数、max_tokens、基础能力、结构化输出、多语言、多模态、参数校验、截断测试、效果测试、性能测试、infra、vllm、天垓100、Qwen3.6

### 4.4 Project 6 里有 15 个遗留 Draft Issue 需要清理

这些是早期用 addProjectV2DraftIssue 创建的，没有 repo 关联、没有 label。应该从 Project 面板里手动删除。

## 五、还没做的（下一步）

1. **muh 语言 PRD 设计** — 定义 muh 的 schema、语法、codegen target、参数空间
2. **muh PRD items 写入 project 6** — 作为真实 Issue，带 label 和测试用例
3. **从 CCCL tuning_*.cuh 提取参数空间** — 建立"CCCL tuning 维度 → muh 配置项"的映射
4. **获取 enginex-vllm-bi100-qwen36 的实际代码** — 需要在 Phanthy Cloud 开发环境里操作
5. **设计 muh → vllm kernel 的 codegen 管道**
6. **实际在天垓100 上跑 benchmark**

## 六、参考项目

- **NVIDIA CCCL Project #6**: github.com/orgs/NVIDIA/projects/6（1990 items，Issue-first 模式，label 做模块分类）
- **pub/sub-loop Project #4**: github.com/users/dylanyunlon/projects/4（1632 items，Draft-first 模式，已验证 1111 个有真实测试步骤，154 个有"按AC验证"占位符）
- **PND 测试库**: 818 条车载软件测试用例，作为 PRD 测试用例质量基准

## 七、关键文件路径

```
project_6/
├── cccl_upstream/           # NVIDIA CCCL 完整代码
│   ├── cub/cub/device/dispatch/tuning/  # 27 个 tuning policy 文件
│   ├── cub/cub/warp/        # warp-level 原语
│   ├── cub/cub/block/       # block-level 原语
│   ├── thrust/examples/     # 60 个优化模式示例
│   └── cudax/...allocators/ # 内存分配器
├── Dockerfile               # TODO: 待创建
├── computility-run.yaml     # TODO: 待创建
└── muh/                     # TODO: muh 语言实现
```

## 八、竞赛关键参数（来自 computility-run.yaml 参考）

```yaml
concurrency: 1
command:
  - python3 -m vllm.entrypoints.openai.api_server
  - --model /model
  - --served-model-name llm
  - --max-model-len 100000
  - --gpu-memory-utilization 0.9
  - -tp 4
  - --max-num-seqs 1
  - --max-num-batched-tokens 8192
  - --enable-chunked-prefill
  - --max-seq-len-to-capture 32768
  - --enable-auto-tool-choice
  - --tool-call-parser qwen3_coder
  - --reasoning-parser qwen3
  - --enable-prefix-caching
env:
  - name: VLLM_ENGINE_ITERATION_TIMEOUT_S
    value: 3600
```

基础镜像: `harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3`

## 九、CCCL Tuning 文件全量模型输入记录

**所有 27 个 tuning_*.cuh 文件的完整源码已在本 context 中作为模型输入读取。** 关键发现：

### policy_selector 统一模式

每个算法都有一个 `policy_selector` struct，接受 `::cuda::compute_capability cc` 参数，内部按 SM 版本做 if-else 分支：

```
if (cc >= {10, 0}) → sm100 tuning (Blackwell)
if (cc >= {9, 0})  → sm90 tuning (Hopper)  
if (cc >= {8, 0})  → sm80 tuning (Ampere)
if (cc >= {7, 0})  → sm70 tuning (Volta)
if (cc >= {6, 0})  → sm60 tuning (Pascal)
fallback           → sm50 tuning
```

**muh 的核心工作就是给每个 policy_selector 添加一个 `cc == {iluvatar, 100}` 分支，填入在天垓100 上跑出的最优 benchmark 数据。**

### 各算法提取的参数维度

| 算法 | 文件 | 行数 | 参数维度 |
|------|------|------|---------|
| reduce | tuning_reduce.cuh | 478 | threads, items, vec_size, reduce_algorithm, load_modifier, determinism |
| scan | tuning_scan.cuh | 1525 | threads, items, load_algo, load_mod, store_algo, scan_algo, delay_policy + lookahead variant |
| radix_sort | tuning_radix_sort.cuh | 2381 | histogram(threads,items,partitions,radix_bits) + exclusive_sum + onesweep(threads,items,store,rank,scan,partitions,radix_bits) + downsweep + upsweep + single_tile |
| reduce_by_key | tuning_reduce_by_key.cuh | 1735 | threads, items, load_algo, load_mod, scan_algo, delay_policy |
| select_if | tuning_select_if.cuh | 2729 | threads, items, load_algo, load_mod, scan_algo, delay_policy |
| histogram | tuning_histogram.cuh | 363 | threads, pixels_per_thread, vec_size, load_algo, load_mod, rle_compress, mem_preference, work_stealing |
| topk | tuning_topk.cuh | 121 | threads, items (simple, no SM-specific tuning yet) |
| batched_topk | tuning_batched_topk.cuh | 186 | worker_policy array × 6 tiers + multi_worker_policy |
| merge | tuning_merge.cuh | 180 | threads, items, load_mod, store_algo, bulk_copy_keys, bulk_copy_values |
| merge_sort | tuning_merge_sort.cuh | 193 | threads, items, load_algo, load_mod, store_algo |
| transform | tuning_transform.cuh | 549 | threads, items, load_algo, store_algo, load_mod |
| rle_encode | tuning_rle_encode.cuh | 626 | threads, items, load_algo, load_mod, scan_algo, delay_policy |
| rle_non_trivial | tuning_rle_non_trivial_runs.cuh | 691 | threads, items, load_algo, load_mod, store_time_slicing, scan_algo, delay |
| adjacent_diff | tuning_adjacent_difference.cuh | 118 | threads, items, load_algo, load_mod, store_algo (single policy, no SM branching) |
| for | tuning_for.cuh | 78 | threads, items (trivial, 256×2) |
| find | tuning_find.cuh | 90 | threads, items, vec_size, load_mod |
| batch_memcpy | tuning_batch_memcpy.cuh | 227 | small_buffer + large_buffer sub-policies |
| scan_by_key | tuning_scan_by_key.cuh | ~2000 | same as reduce_by_key pattern |
| unique_by_key | tuning_unique_by_key.cuh | ~1500 | same pattern |
| three_way_partition | tuning_three_way_partition.cuh | ~780 | same pattern |
| segmented_* | 4 files | ~1300 total | segmented variants of reduce/scan/sort |

### Benchmark 注释格式

每个 sm100 tuning 都有注释格式：
```
// ipt_22.tpb_384.ns_1904.dcid_6.l2w_830.trp_1.ld_0 1.148442 0.997167 1.139902 1.462651
```
- `ipt` = items_per_thread
- `tpb` = threads_per_block  
- `ns` = delay nanoseconds
- `dcid` = delay constructor ID
- `l2w` = L2 cache window
- `trp` = transpose (0=DIRECT, 1=WARP_TRANSPOSE)
- `ld` = load modifier (0=DEFAULT, 1=LDG, 2=CA)
- 4 个数字 = 4 种 problem size 下的加速比 (vs 前代 SM)
