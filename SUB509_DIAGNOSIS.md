# Sub509 诊断报告 — 2026-08-07

## 评测结果对比

| 测试 | Sub168 (对手) | Sub509 (我们) | 状态 | 根因 |
|------|--------------|--------------|------|------|
| d01_basic_nostream | 8.49s, content[11], 139tok | 95.85s, content[0], 1085tok | ✓ PASS | decode 慢 + thinking 过长 |
| d02_stream_usage | 2.75s | 1.84s | ✓ PASS | OK |
| **d03_tool_call** | **2.12s, tools=1** | **49.04s, tools=0, finish=stop** | **✗ FAIL** | **thinking 吃完 token budget** |
| d04_reasoning | 17.78s | 128.74s | ✓ PASS | decode 慢 |

## d03_tool_call FAIL 根因链

```
评测器发 tool_choice=auto + tools=[get_weather]，不带 thinking 参数
  → enable_thinking 默认 True
  → 模型进入 <think>...</think> 模式
  → BI-V100 decode 极慢 (~11 tok/s)，49 秒全在 thinking
  → max_tokens 耗尽，finish=stop
  → 从未输出 <tool_call> XML
  → tool_parser 检测不到 <function=
  → tools_called=False, finish_reason="stop"
  → ✗ FAIL
```

## 已提交修复 (commit e0344b1)

1. **protocol.py**: `normalize_messages` 里当 tools 活跃 + tool_choice=auto + thinking 未显式设置时，自动 `enable_thinking=False`
2. **qwen3coder_tool_parser.py**: `adjust_request()` 做相同检查，defense-in-depth
3. **baseline.muh**: 同步实际部署配置 (max_num_seqs=1, 去掉 chunked_prefill)

## 性能差距分析

对手 decode 速度约 16 tok/s (139tok / 8.49s)
我们 decode 速度约 11 tok/s (1085tok / 95.85s)

差距来源：
- BI-V100 vs 对手硬件（未知，可能是 A100/H100）
- enforce_eager=True 禁用了 CUDA graphs
- 单卡 SMEM 48KB vs 高端卡 192KB+
- xformers head_dim>128 fallback 走 PyTorch SDPA（非 FlashAttention）

## 下一步优化方向

### 紧急 (影响评分)
- [ ] 提交后看完整 d05~d16+ 测试结果
- [ ] 如有其他 FAIL，同样根因分析

### 中期 (提升 TPS)
- [ ] 检查 _PYTORCH_DECODE_THRESHOLD=999999 是否可以降低以启用 PyTorch fallback 对特定 seq_len 范围
- [ ] sampler 的 .tolist()/.cpu() sync 点优化（但这是 vllm 标准路径）
- [ ] V2 attention partition_size 根据 BI-V100 SM count 调优

### CCCL 架构借鉴
- [ ] tuning_reduce.cuh 的 scale_mem_bound 公式应用到 BI-V100：max_smem=49152
- [ ] tuning_scan.cuh 的 ScanLookbackPolicy 参数对标 muh schema YAML
- [ ] policy_selector DSL：compute_capability 维度改为 BI-V100 硬件描述
