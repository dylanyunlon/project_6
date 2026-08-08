# PRD: 天垓100 BI-V100 推理引擎竞赛

## 目标
首位通过全部功能测试+效果测试+性能基准的参赛者获得基础奖。

## 竞赛门槛
- 50+ 功能测试用例全部通过
- 效果偏差 ≤±4%
- 性能门槛 Token 吞吐加权值 ≥8000
- Output TPS 权重占 83%（decode kernel 优化投入产出比最高）

## 架构策略
CCCL系统设计移植 + base引擎serving层改造。

### 核心原则
1. **不覆盖模型层代码** — Sub168证明base镜像CoreX原生代码能正确运行
2. **只部署serving层** — patch_ops.sh控制部署范围
3. **通过环境变量做硬件适配** — CCCL policy_selector模式

### 部署文件清单（patch_ops.sh）
- protocol.py — OpenAI API兼容层
- serving_chat.py — 请求处理核心
- qwen3coder_tool_parser.py — Qwen3 XML tool call解析
- reasoning/ — thinking/reasoning分离
- api_server.py — 入口点
- chat_utils.py — 消息预处理
- cli_args.py — 参数注册
- registry.py — 仅当base缺少Qwen3_5时

### 不部署的文件（base镜像原生）
qwen3_5.py, model_runner.py, _custom_ops.py, sampler.py,
scheduler.py, sequence.py, xformers.py, paged_attn.py,
prefix_prefill.py, logits_processor.py, mamba_cache.py, arg_utils.py

## Sub168参数基准（已对齐）
- max_model_len=256000
- max_num_seqs=2
- gpu_memory_utilization=0.95
- max_num_batched_tokens=4096
- enable_chunked_prefill=True
- enforce_eager=True
- dtype=half
- tensor_parallel_size=4

## CCCL → base 映射记录
| CCCL源码 | 映射到base位置 | 改动类型 |
|----------|---------------|---------|
| buddy_allocator.cu | computility-run.yaml env | PYTORCH_CUDA_ALLOC_CONF |
| device_reduce policy_selector | computility-run.yaml params | 启动参数对齐Sub168 |
| agent_reduce_by_key ConsumeTile | serving_chat.py | fast path/safe path分离 |
| tuning_find_bound_sorted_values | yaml --dtype half | 类型大小自适应 |

## 已修复的Sub508/509失败点
1. ✅ n>1 OOM级联 → 允许n=2（匹配max_num_seqs=2）
2. ✅ max_completion_tokens 400 → protocol.py接受
3. ✅ tool_calls content=None → chat_utils.py容错
4. ✅ d03 tool_call thinking耗尽 → 自动禁用thinking
5. ✅ 内存碎片OOM → PYTORCH_CUDA_ALLOC_CONF
6. ✅ 模型层代码破坏CoreX → patch_ops.sh只部署serving层

## CCCL tuning_select_if.cuh → serving_chat.py 映射

### 设计思想翻译
CCCL三级分发：compute_capability → sm_tuning → benchmark参数
我们三级分发：请求类型 → 处理路径 → Sub168实测参数

### 参数对应关系
| CCCL概念 | 我们的对应 |
|---------|-----------|
| compute_capability (SM80/90/100) | 请求类型 (tool_call/reasoning/basic) |
| input_size (1/2/4/8 bytes) | 请求复杂度 (simple/multimodal/multi-turn) |
| flagged/unflagged | has_tools/no_tools |
| keep_rejects/discard | enable_thinking/disable_thinking |
| threads_per_block | max_tokens cap |
| items_per_thread | default_max_tokens计算 |
| delay_constructor | token budget 分配策略 |
| benchmark注释 (4个加速比) | Sub168日志实测数据 |

### Sub168 benchmark数据（=我们的tuning表）
| 请求类型 | 时间 | token数 | TPS |
|---------|------|---------|-----|
| d01 basic | 8.49s | 139 | 16.4 |
| d03 tool_call | 2.12s | ~34 | ~16 |
| d04 reasoning | 17.78s | 1192 | 67 |
| d07 reasoning+content | 61.11s | 4451 | 72.8 |
| replay avg | - | - | 11.86 |
