# 竞赛对比分析 & 修复计划

## 一、核心数据对比

| 模块 | 对手 Sub168 | 我们 Sub508 | 差距 |
|------|-----------|-----------|------|
| **functional** | 48/52 PASS (92.3%) | 21/51 PASS (41.2%) | **-51%** |
| **case_truncation** | score=1.0 (8192 tokens输出完整) | score=0.0 (引擎崩溃) | **致命** |
| **replay_tencent** | score=60194 (94/881成功,tps avg 11.86) | score=0.0 (881/881 connection refused) | **致命** |
| **opencompass** | 0.0 (server也崩了) | 0.0 (同上) | 平 |
| **总分** | **60194.6** | **0.0** | -- |

## 二、Sub508 崩溃根因链

```
t2_n_2 (n=2请求) → get_scheduler_config() 异常 → 引擎进程死亡
→ 后续所有请求 Connection Refused → 30个FAIL级联
→ case_truncation/replay/opencompass 全部0分
```

**关键事实：t2_n_2 崩溃发生在 06:42:45，之后所有模块都是在引擎已死的情况下跑的。**

## 三、对手 Sub168 的弱点（我们已经修复的）

1. **`max_completion_tokens` 被拒** — 对手 `extra="forbid"` 导致 replay 中所有带此字段的请求返回 400。我们已添加该字段到 protocol.py，replay 中不会被拒。
2. **`tool_calls` content=None 被拒** — 对手的 replay preflight 失败（"Each message must have at least one of 'content' or 'reasoning_content'"）。我们已修复 chat_utils.py 中 content=None 的处理。
3. **d06_cache_hit FAIL** — 对手没有 prefix caching，我们 PASS。
4. **t3_max_tokens_1/64/max 3个FAIL** — 对手也有3个max_tokens测试失败。

**对手 replay 中 787/881 失败(89.3%)，只有 94 个成功。我们的目标是超越这个。**

## 四、我们需要修复的问题（按优先级排序）

### P0 — 引擎稳定性（决定能否拿分的前提）

| 问题 | 根因 | 修复位置 |
|------|------|----------|
| **t2_n_2 → 引擎崩溃级联** | `get_scheduler_config()` 异常 + n>1 未处理 | `qwen3_6_scripts/serving_chat.py` + `protocol.py` |
| **引擎OOM死亡** | 单个长请求耗尽GPU内存后整个进程死 | 需要在 worker/model_runner.py 加 OOM catch |

已有 commit 修复（994c657 clamp n>1, c241764 try-catch scheduler），但 **Sub508 用的是修复前的代码**。Sub509 日志确认 d01 能跑（95.85s），但 d03 仍然 FAIL。

### P1 — d03_tool_call FAIL（功能测试核心分）

**Sub508**: `tools=0 finish=stop reasoning[0]` (49.04s)
**Sub509**: `tools=0 finish=stop reasoning[0]` (49.04s)
**对手**: `tool=get_weather args="{'city': 'Beijing'}" finish=tool_calls` (2.12s)

**根因分析**：
- 对手 d03 只用了 2.12s，模型直接输出 tool_call XML，tool parser 正确解析
- 我们用了 49.04s，模型在 thinking 中耗尽了时间，没有产生 `<tool_call>` 标签
- commit e0344b1 说"禁用 tool_call 请求的 thinking"，但 Sub509 的 d03 仍显示 `reasoning[0]`
- **真正的问题**：当 `tool_choice=auto` 且有 tools 时，需要在 chat_template 中设置 `enable_thinking=False`，否则 Qwen3 会先 think 再输出，大量token浪费在思考上

**修复方案**：在 `serving_chat.py` 的 `create_chat_completion` 中，当检测到 `request.tools` 且 `tool_choice != "none"` 时，在 `chat_template_kwargs` 中注入 `enable_thinking=False`。

### P1 — d05_multimodal HTTP 400

对手 PASS (content[374])，我们 HTTP 400。
可能是多模态请求格式/图片解码问题。需要检查 chat_utils.py 的图片处理路径。

### P1 — d07_reasoning_plus_content

对手 PASS (reasoning[3489] content[962])，我们 FAIL (reasoning[131] content[0])。
模型 think 后不产生 content。这是模型行为问题，但可以通过调低 thinking budget 或调整 temperature 来缓解。

### P2 — t1a_thinking_true / t1c_thinking_default

对手 PASS (reasoning[541] / [411])，我们 FAIL (reasoning[0])。
**根因**：模型在短回答场景下不触发 thinking。可能需要在 chat_template 中确保 `enable_thinking=True` 是默认值。检查 Qwen3.6 的 chat_template 是否正确注入了 `<think>` 标签。

### P2 — d10_thinking_disable_ctk 乱码输出

对手输出 `'4'`（正确），我们输出乱码 `"presت< **sama一..."`。
模型在 thinking disabled 模式下输出质量极差。这是模型+chat_template 的交互问题。

### P3 — 速度差距

| 测试 | 对手 | 我们 | 倍数 |
|------|------|------|------|
| d01 | 8.49s | 95.85s | **11x慢** |
| d04 | 17.78s | 128.74s | **7x慢** |
| d03 | 2.12s | 49.04s | **23x慢** |

速度问题核心：BI-V100 硬件本身比 NVIDIA GPU 慢，但 10x 的差距说明还有架构问题。对手的 output_tps 平均 11.86，decode 阶段 tps 在 2.4-22.7 之间。

## 五、修复代码的具体文件

需要修改的文件（全部在 `qwen3_6_scripts/` 中，会被 patch_ops.sh 部署）：

1. **`serving_chat.py`** — tool_call 时注入 `enable_thinking=False`
2. **`protocol.py`** — 确认 `extra="forbid"` 已经去掉（已做），确认 `thinking` 字段被正确传递
3. **`chat_utils.py`** — 多模态请求处理、content=None 容错
4. **`model_runner.py`** — OOM recovery
5. **`qwen3_5.py`** — 检查模型是否正确处理 `enable_thinking` 参数
6. **`computility-run.yaml`** — 考虑调整 `--max-num-seqs` / `--gpu-memory-utilization`

## 六、对手的 replay 得分结构

对手 881 个请求中：
- 94 个成功 (10.7%)
- 77 个因 `max_completion_tokens` extra_forbidden 而 400
- 704 个 connection refused（server也崩了！）
- output_tps_avg = 11.86, output_tps_p50 = 12.97

**关键发现：对手的 server 也在 replay 后期崩溃了（704 个 connection refused）。但他在崩溃前完成了 94 个请求。**

我们的优势：
- 我们已修复 `max_completion_tokens` → 对手的 77 个 400 我们不会有
- 我们已修复 `tool_calls content=None` → 对手的 tool preflight fail 我们不会有
- 我们有 prefix caching → 对手没有

**如果我们能保持引擎稳定不崩溃，仅靠不拒绝 max_completion_tokens 的请求，就能多处理 77+ 个请求，超过对手。**

## 七、下一步行动

1. 修复 `serving_chat.py`：tool_call 时禁用 thinking
2. 确认 n>1 clamp 和 scheduler try-catch 在 patch 文件中生效
3. 测试 OOM 恢复逻辑
4. 调整 computility-run.yaml 参数确保稳定性
5. 提交部署，跑测试
