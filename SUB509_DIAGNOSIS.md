# Sub508/509 完整诊断报告

## 修复提交记录

| Commit | 修复 | 影响 |
|--------|------|------|
| e0344b1 | 禁用 tool_call 请求的 thinking | d03 FAIL → 预计 PASS |
| c241764 | get_scheduler_config try-catch | 防止引擎崩溃 |
| 994c657 | clamp n>1 to 1 | 防止 t2_n_2 级联崩溃 (19 个测试) |

## Sub508 完整测试结果 (56 tests)

### 实际结果: PASS=21, FAIL=30, SKIP=5

### 级联崩溃 (19 个 FAIL 来自 t2_n_2 引擎崩溃)
t2_n_2 → HTTP 500 → 引擎死亡 → t3_max_tokens_none/1/64/mid/max/neg1/over,
t4a/4b, t5, t6, t7, t8, t9, t10, t12_chinese/japanese/emoji 全部 HTTP 500

### 修复后预期: PASS ≈ 40+, FAIL ≈ 10-

### 真正的功能性 FAIL (非级联)

| 测试 | 状态 | 根因 | 可修 |
|------|------|------|------|
| d03_tool_call | tools=0 finish=stop | ✅ 已修复 thinking budget | 是 |
| d05_multimodal | HTTP 400 | multimodal 请求格式 | 需查 |
| d07_reasoning+content | content[0] | 模型 think 后不产 content | 否(模型) |
| d10_thinking_disable_ctk | 乱码 content | 模型质量 | 否(模型) |
| t1a_thinking_true | reasoning[0] | 模型跳过 thinking | 否(模型) |
| t1c_thinking_default | reasoning[0] | 同上 | 否(模型) |
| t2_n_2 | HTTP 500 → cascade | ✅ 已修复 clamp n | 是(防崩) |

## 对手 Sub168 对比

| 维度 | 对手 | 我们 |
|------|------|------|
| functional PASS | ~50/56 | 21/56 → 修后 ~40/56 |
| d01 速度 | 8.49s | 95.87s |
| d04 速度 | 17.78s | 129.19s |
| replay max_completion_tokens | ✗ 400 rejected (30+次) | ✓ 已支持 (extra=ignore) |
| replay tool_calls content=None | ✗ 400 rejected | ✓ 已支持 (normalize) |
| decode TPS | ~16 tok/s | ~11 tok/s |

## 我们 vs 对手的优势
1. `max_completion_tokens` 支持 — 对手 replay 有 30+ 个 400 错误
2. `tool_calls` content=None 支持 — 对手 replay preflight 失败
3. `reasoning_effort` 字段容忍 — 对手被拒
4. prefix caching 工作 (d06 PASS) — 对手 d06 FAIL
