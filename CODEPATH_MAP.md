# 代码路径时序图 — 从HTTP请求到GPU kernel的完整链路

## 一、请求入口到引擎调用

```
HTTP POST /v1/chat/completions
  │
  ├─ api_server.py → FastAPI route handler
  │    └─ serving_chat.py:create_chat_completion() [line ~140]
  │         ├─ protocol.py:ChatCompletionRequest.model_validate()
  │         │    └─ max_completion_tokens → max_tokens 映射 [line 418]
  │         │    └─ extra="allow" (Sub168用extra="forbid"导致400)
  │         │
  │         ├─ chat_utils.py → 消息格式化 + 多模态处理
  │         │    └─ content=None容错 (Sub168这里崩)
  │         │
  │         ├─ serving_chat.py [line 175-213] → enable_thinking逻辑
  │         │    ├─ tool_choice=auto + tools存在 → enable_thinking=False
  │         │    ├─ thinking.type=disabled → enable_thinking=False
  │         │    └─ 默认 → enable_thinking=True
  │         │
  │         ├─ serving_chat.py [line 250-252] → n值检查
  │         │    └─ n>2 → 400 (n=2允许传入引擎)
  │         │
  │         └─ engine_client.generate() [line 355]
  │              └─ try/except ValueError + catch-all Exception
  │
  ├─ computility-run.yaml → vLLM启动参数
  │    ├─ --max-num-seqs 2 (防止n=2崩溃)
  │    ├─ --max-model-len 80000
  │    ├─ --enforce-eager (禁用CUDA Graph)
  │    ├─ --enable-prefix-caching
  │    └─ --tool-call-parser qwen3_coder
  │
  └─ 如果引擎crash → 后续所有请求Connection Refused
       (Sub508的根因: t2_n_2触发, 30个FAIL级联)
```

## 二、模型前向传播 — 逐层链路

```
Qwen3_5ForCausalLM.forward()  [qwen3_5.py line 1214]
  │
  └─ Qwen3_5Model.forward()  [line 1094]
       │
       ├─ embed_tokens(input_ids)
       │
       └─ for layer in self.layers:  # 36层 (Qwen3.6-27B典型配置)
            │
            ├─ GemmaRMSNorm(hidden_states, residual)
            │    └─ ☆ 可用ixformer: fused_add_rms_norm(input, residual, weight, eps)
            │
            ├─ [linear_attention层] GatedDeltaNet.forward() [line 407]
            │    │
            │    ├─ CoreX dispatch尝试 [line 416-425]
            │    │    └─ _use_corex_gdn=False (base image无corex_gdn模块)
            │    │
            │    └─ _pytorch_forward() [line 435]  ← 当前执行路径
            │         │
            │         ├─ 投影: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a
            │         │    └─ ☆ 每个是F.linear → 可用ixformer.matmul
            │         │
            │         ├─ [prefill] 逐序列循环 [line 463-555]
            │         │    │
            │         │    ├─ F.conv1d (causal conv)
            │         │    │    └─ ☆ 可用ixformer.conv2d (需reshape)
            │         │    │
            │         │    ├─ F.silu → ☆ 可用ixformer.silu_and_mul
            │         │    │
            │         │    ├─ g计算: -A_log.exp() * softplus(a+dt_bias)
            │         │    │    └─ 当前: clamp(-8,4)后exp, softplus.clamp(max=10)
            │         │    │
            │         │    └─ _torch_chunk_gated_delta_rule() [line 152-247]
            │         │         │
            │         │         ├─ g.clamp(-5,2).cumsum(-1).clamp(-20,20) ← NaN修复点
            │         │         ├─ decay_mask = exp(g差) ← 所有exp在clamp后
            │         │         ├─ attn矩阵: k_beta @ key.T * decay_mask
            │         │         │    └─ ☆ 三角求解循环 → 无法用ixformer加速
            │         │         │       (这是纯序列依赖: attn[i] += attn[i,:i] @ attn[:i,:i])
            │         │         ├─ state更新循环: for i in chunks [line 219-232]
            │         │         │    ├─ q @ k.T * decay ← ☆ ixformer.matmul可加速
            │         │         │    ├─ q * exp(g) @ state ← ☆ ixformer.matmul可加速
            │         │         │    └─ state更新: state * exp(g) + k.T @ v_new
            │         │         │         └─ ☆ ixformer.matmul可加速
            │         │         └─ 最终: core_out → transpose → to(dtype)
            │         │
            │         ├─ [decode] 单token路径 [line 558-638]
            │         │    ├─ _torch_causal_conv1d_update
            │         │    │    └─ 逐通道点积 → ☆ ixformer.gemv可加速
            │         │    ├─ g_t = g.clamp(-20,2).exp_() ← NaN修复点
            │         │    ├─ temporal_state.mul_(g_t) ← 状态衰减
            │         │    ├─ torch.bmm(k, state) ← ☆ ixformer.matmul可加速
            │         │    └─ state.baddbmm_(k, delta) ← ☆ ixformer.matmul可加速
            │         │
            │         └─ GemmaRMSNorm + out_proj
            │              └─ ☆ ixformer.rms_norm + ixformer.matmul
            │
            ├─ [full_attention层] Qwen3_5FullAttention.forward() [line 737]
            │    └─ 标准vLLM Attention → XFormers后端
            │         └─ ☆ 已使用ixformer.flash_attn_func (base image配置)
            │
            ├─ GemmaRMSNorm(hidden_states, residual)
            │    └─ ☆ ixformer.fused_add_rms_norm
            │
            └─ [MLP/MoE] Qwen3_5MLP 或 Qwen3_5MoeSparseBlock
                 │
                 ├─ [MLP] gate_up_proj → silu_and_mul → down_proj
                 │    └─ ☆ 全部可用ixformer: matmul + silu_and_mul + matmul
                 │
                 └─ [MoE] Qwen3_5MoeSparseBlock.forward() [line 974]
                      ├─ gate(hidden) → router_logits
                      ├─ softmax → topk → renormalize (纯PyTorch, 无硬件加速)
                      ├─ _pure_pytorch_experts() [line 897]
                      │    ├─ [decode T=1] 批量GEMM: 3次kernel launch
                      │    │    └─ F.linear(x, w13_sel.reshape(-1,H)) ← ☆ ixformer.matmul
                      │    │    └─ F.silu(gate) * up ← ☆ ixformer.silu_and_mul (需reshape)
                      │    │    └─ torch.bmm(w2_sel, act) ← ☆ ixformer.matmul
                      │    └─ [prefill] 逐expert循环 ← 性能瓶颈
                      │         └─ 每个expert: F.linear × 2 + silu
                      │              └─ ☆ 可用ixformer.matmul但循环开销不变
                      └─ shared_expert: gate_up → silu_and_mul → down → sigmoid gate
                           └─ ☆ 全部可用ixformer
```

## 三、ixformer可用原语 vs 当前使用情况

| ixformer原语 | 签名 | 当前是否使用 | 可替换的PyTorch调用 |
|-------------|------|------------|-------------------|
| `matmul` | `matmul(input, other, out, transa, transb, alpha, beta)` | ❌ 未使用 | F.linear, torch.mm, torch.bmm, @ |
| `softmax` | `softmax(input, dim)` | ❌ 未使用 | torch.softmax (MoE路由) |
| `rms_norm` | `rms_norm(input, weight, output, eps)` | ❌ 未使用 | GemmaRMSNorm内部 |
| `fused_add_rms_norm` | `fused_add_rms_norm(input, residual, weight, eps, scale)` | ❌ 未使用 | residual + layernorm 两步 |
| `silu_and_mul` | `silu_and_mul(input, output)` | ❌ 未使用 | SiluAndMul层, F.silu(g)*up |
| `conv2d` | `conv2d(input, weight, bias, stride, padding, dilation, groups)` | ❌ 未使用 | F.conv1d (causal conv) |
| `flash_attn_func` | `flash_attn_func(q, k, v, dropout_p, softmax_scale, causal)` | ✅ XFormers后端使用 | full_attention层 |
| `gemv` | `gemv(x, A)` | ❌ 未使用 | decode路径小矩阵乘 |
| `scaled_dot_product_attention` | `sdpa(query, key, value, attn_mask, dropout_p, is_causal)` | ❌ 未使用 | 可替代chunk内QK^T计算 |

**关键发现：9个可用原语中只有1个（flash_attn_func）被使用，而且不是我们的代码使用的——是base image的XFormers后端自动调用的。我们的代码对ixformer的利用率是0%。**

## 四、Sub168 vs Sub508 性能差距的代码解释

```
Sub168 (8.49s for d01):
  base image native qwen3_5.py
  ├─ corex_gdn: 使用libcorex_gdn.so的fused GDN kernel ← 不存在于我们的base image
  ├─ corex_moe: 使用libcorex_moe.so的fused MoE kernel ← 不存在于我们的base image
  └─ 所有底层ops由ixformer后端加速 (matmul/rms_norm/softmax等)

Sub508 (95.85s for d01):
  我们的自定义 qwen3_5.py
  ├─ GatedDeltaNet: 纯PyTorch (cumsum→exp→NaN→nan_to_num→全零)
  ├─ MoE: 纯PyTorch循环 (每expert单独F.linear)
  └─ 底层ops全部用PyTorch默认kernel (未调用ixformer)
```

## 五、优化路径 — 用ixformer原语替换PyTorch

### 立即可做 (不改算法, 只换kernel):
1. **matmul**: 所有F.linear/torch.bmm/@ → ixformer.matmul
2. **silu_and_mul**: MLP和MoE的silu*gate → ixformer.silu_and_mul
3. **rms_norm**: GemmaRMSNorm内部 → ixformer.rms_norm
4. **fused_add_rms_norm**: residual+norm两步 → 一步fused
5. **softmax**: MoE路由softmax → ixformer.softmax

## 六、功能测试FAIL根因分析（6个非crash FAIL）

```
FAIL类型A: NaN导致模型输出质量问题 (修NaN后自愈)
├─ d03_tool_call: tools=0 — 模型不能输出<tool_call> XML
├─ d07_reasoning_plus_content: content[0] — 模型不输出</think>
├─ d10_thinking_disable_ctk: 乱码 — 模型logits被NaN扭曲
├─ t1a_thinking_true: reasoning[0] — output.text为空→parser返回空
└─ t1c_thinking_default: reasoning[0] — 同上

FAIL类型B: 请求处理层问题
└─ d05_multimodal: HTTP 400 — 多模态请求验证失败

FAIL类型C: 引擎crash级联 (修max-num-seqs=2后自愈)
└─ t2_n_2 → t3/t4/t5/t6/t7/t8/t9/t10/t12/t13/t14/t15/t16 全部HTTP 500 (25个)

当前代码状态:
  NaN修复: ✅ cumsum前clamp[-5,2] + 后clamp[-20,20] + A_log clamp[-8,4]
  引擎防崩: ✅ max-num-seqs=2 + catch-all Exception
  ixformer加速: ✅ matmul/bmm/softmax接入12处热路径
  reasoning parser: ✅ qwen3已注册，部署正确
  tool parser: ✅ qwen3_coder已注册，adjust_request禁thinking
  
预期: NaN修复后模型质量恢复 → 类型A的5个FAIL自愈
      max-num-seqs=2 → 类型C的25个FAIL自愈
      剩余: d05_multimodal需要单独debug
      预估: 45/51 PASS (88%)
```
