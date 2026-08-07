# Sub509 深度诊断 — 基于CCCL源码阅读的系统级分析

## 一、Sub509 vs Sub168 关键数据对比

| 测试 | 对手Sub168 | 我们Sub509 | 差距分析 |
|------|-----------|-----------|---------|
| d01_basic_nostream | 8.49s, content[11] tok=139 | 95.85s, content[0] reasoning[1102] tok=1085 | 11x慢; 我们产了1085个token全是reasoning |
| d02_stream_usage | 2.75s, chunks=53 | 1.84s, chunks=9 | 我们居然更快(但只产了9个chunks vs 53) |
| d03_tool_call | 2.12s, tool=get_weather | **49.04s, tools=0 finish=stop** | **致命**: 模型不输出<tool_call> XML |
| d04_reasoning | 17.78s, content[181] reasoning[1011] | 128.74s, content[0] reasoning[1447] | 7x慢; 我们有reasoning但没有content |

## 二、三大根因(按严重程度排序)

### 根因1: GatedDeltaNet每层产NaN → 模型"智力"丧失

docker日志证据:
```
WARNING qwen3_5.py:445] NaN in prefill GatedDeltaNet layer 0 (frac=0.9998)
WARNING qwen3_5.py:445] NaN in prefill GatedDeltaNet layer 1 (frac=0.9997)
WARNING qwen3_5.py:445] NaN in prefill GatedDeltaNet layer 2 (frac=1.0000)
WARNING qwen3_5.py:445] NaN in prefill GatedDeltaNet layer 4 (frac=1.0000)
```

**99.98%-100% NaN率**。`nan_to_num(result, nan=0.0)` 将这些NaN替换为零，等于整个DeltaNet层输出全是零。
这是一种"活着但脑死亡"的状态——前向传播不报错，但模型失去了DeltaNet层的能力。

**NaN来源追踪**:
1. `_torch_chunk_gated_delta_rule` 中 `g.cumsum(dim=-1)` → 累积值可能极大
2. `g.clamp(-20,20)` 后 `g.exp()` → 最大 ~5e8，但这些值进入矩阵乘法后仍可能溢出
3. `decay_mask = (g_diff).tril().exp()` → 即使单个exp不溢出，大矩阵乘法的累加也可能溢出
4. `_forward_sub_lower` 中的前向替代: `x[i] = rhs[i] + A[i,:i] @ x[:i]`，如果A中有大值，误差逐行放大

**对手为什么没有这个问题**: 对手可能用的是不同的模型架构(不含DeltaNet)，或者在NVIDIA GPU上float32精度够高不会溢出。

### 根因2: FusedMoE完全fallback → 性能灾难

```
ERROR _custom_ops.py:58] module 'ixformer.functions' has no attribute 'vllm_moe_topk_softmax'
WARNING qwen3_5.py:913] FusedMoE native kernel failed, falling back to pure PyTorch experts permanently.
```

BI-V100的ixformer没有MoE kernel，所有MoE层都用纯PyTorch:
- 256个expert × top_k=8 → 最多256次F.linear调用(prefill)
- 每次decode也需要top_k=8次expert forward
- 对比native kernel的1次fused launch，这是数量级的差距

### 根因3: computility-run.yaml vs 实际参数不一致

yaml写的: `--max-model-len 256000 --max-num-seqs 2 --gpu-memory-utilization 0.95`
docker日志: `max_seq_len=100000, max_num_seqs=1, gpu_memory_utilization=0.9`

**可能原因**: 部署时还在用旧的配置。需要确认yaml是否真的被用于部署。

## 三、d03_tool_call为什么FAIL

d03日志: `tools=0 finish=stop reasoning[0] (tool_choice=auto) (49.04s)`

**reasoning[0]说明enable_thinking=False确实生效了**。但模型仍然不输出`<tool_call>` XML。

analysis:
1. enable_thinking=False → 模型不产生`<think>...</think>`块 ✓
2. 但模型的输出内容不包含`<tool_call><function=get_weather>...` 格式
3. tool parser `Qwen3CoderToolParser` 在输出中找不到 `<function=` → tools_called=False
4. 49.04s意味着模型在漫长生成纯文本回答(可能是口头描述天气而不是调用tool)

**核心问题: GatedDeltaNet的NaN导致模型质量太差,不能正确follow tool_call格式**

这不是serving_chat.py的问题。serving_chat.py和protocol.py中的tool_call thinking禁用逻辑是正确的。问题在模型本身。

## 四、对手Sub168分析

对手最终得分60194.6:
- functional: ~48/52 PASS
- case_truncation: score=1.0
- replay_tencent: score=60194 (94/881成功, tps avg 11.86)
- opencompass: 0.0 (server也崩了)

**对手server在replay后期也崩溃了**(704个connection refused)。
**对手的replay也只有94/881成功(10.7%)**。

但对手赢在:
1. functional高通过率 → 基础分
2. case_truncation通过 → 引擎稳定
3. replay中94个成功请求 × tps → 得到分数

## 五、修复路径(按投入产出比排序)

### 修复1: NaN问题 — 强制float32精度 + 更激进的clamp

当前: `g.clamp(-20, 20)` 不够。cumsum后再clamp太晚了。
需要: 在cumsum之前就对g的原始值做clamp。

在 `_torch_chunk_gated_delta_rule`:
```python
# 现在: g = g.cumsum(dim=-1).clamp(-20, 20)
# 改为: g = g.clamp(-5, 5).cumsum(dim=-1).clamp(-15, 15)
```

在 GatedDeltaNet.forward 的 prefill path:
```python
# 现在: _A_safe = self.A_log.float().clamp(-20.0, 20.0)
# 改为更窄: _A_safe = self.A_log.float().clamp(-10.0, 10.0)
```

### 修复2: MoE性能 — 尝试真正使用native kernel

docker日志说 `vllm_moe_topk_softmax` 不存在，但 _custom_ops.py 里应该有PyTorch fallback。
问题是 `_hw_policy.moe_native_align` 和 `_hw_policy.moe_native_invoke` 也是False。
如果这两个真的不存在，那PyTorch fallback就是唯一选择。

**性能改进**: 在 `_pure_pytorch_experts` 的 prefill path 中:
- 现在: for-loop over experts, 每个一次F.linear
- 改为: 按expert batch size排序，大batch的expert合并成一个大F.linear (CCCL histogram pattern已经实现了，但可以更激进)

### 修复3: computility-run.yaml 参数对齐

确保部署时真的用了yaml里的参数。max-num-seqs=2让n=2请求不会崩溃。

### 修复4: d01/d04速度

d01: 95.85s产了1085个token，约11.3 tok/s — 其实tps不太差
对手d01: 8.49s产了139个token，约16.4 tok/s

**关键差异不是tps，是产了多少token!** 我们1085 vs 对手139。
我们的模型在thinking里产了大量token。
d01是basic_nostream，没有tool，所以enable_thinking=True是默认的。
thinking产了1102个reasoning token + 0个content token。

**问题: d01测试的content[0]意味着没有实际内容输出!**
对手content[11]说明他输出了内容。

这又回到了GatedDeltaNet NaN → 模型质量差的问题。

## 六、CCCL启示

CCCL在处理数值稳定性方面的核心设计:
1. `overflow_cast_t<T>` — 在可能溢出的地方用更高精度的中间类型
2. `cc_dispatch` — 不同硬件不同策略，不硬编码
3. `policy_selector` — 基于benchmark数据选择参数，不拍脑袋

我们的DeltaNet实现缺少CCCL级别的数值稳定性保证。
