# Engine Code Path Timeline: Sub168 vs Our Sub508/509

**Purpose**: Anyone reading this repo can understand the exact runtime difference in 2 minutes instead of re-deriving from raw logs.

## 1. Boot Sequence Comparison

```
TIME          SUB168 (07-23, score=60194)              OUR SUB508 (08-07, score=0)
──────────────────────────────────────────────────────────────────────────────────
+0s           api_server.py:530 → vLLM 0.6.3           api_server.py:530 → vLLM 0.6.3
              max_model_len=256000                      max_model_len=256000 (same)
              max_num_seqs=2, gpu_mem=0.95              max_num_seqs=2, gpu_mem=0.95 (same)
              chunked_prefill=True                      chunked_prefill=True (same)

+10s          model_runner.py:1074 load start           model_runner.py:1119 load start
              ↑ DIFFERENT line number                   ↑ DIFFERENT line number
              ↑ (base image native model_runner)        ↑ (our patched model_runner)

+18s          weights = 17.3529 GB                      weights = 16.2303 GB
              ↑ 1.1GB MORE (corex state buffers)        ↑ 1.1GB LESS (no corex buffers)

+180s         corex_gdn.py:56  → load libcorex_gdn.so  qwen3_5.py:445 → NaN in prefill layer 0
              corex_gdn.py:228 → GDN prefill OK         ↑ PyTorch GDN produces NaN (99.98%)
              corex_moe.py:339 → MoE prefill OK         qwen3_5.py:913 → FusedMoE FAILED
              corex_fa2.py:333 → FA2 prefill OK         ↑ ixformer.functions missing topk_softmax
              ↑ ALL THREE CoreX accelerators loaded     ↑ ZERO accelerators, all fallback

+182s         GPU blocks: 19259                         GPU blocks: ~19000 (similar)
              Ready to serve                            Ready to serve (but 10x slower)
```

## 2. Call Chain During Inference

### Sub168 (with CoreX) — d01_basic_nostream: 8.49s
```
serving_chat.py → create_chat_completion()
  → engine.generate()
    → model_runner.py:1074 execute_model()
      → qwen3_5.py:1421 Qwen3_5ForCausalLM.forward()
        → qwen3_5.py:1165 Qwen3_5Model.forward()  (decoder layers loop)
          → qwen3_5.py:1086 Qwen3_5DecoderLayer.forward()
            ├─ GatedDeltaNet layers (4 of 36):
            │   ├─ PREFILL: corex_gdn.py:228 → libcorex_gdn.so (fused CUDA kernel)
            │   └─ DECODE:  corex_gdn.py:138 → libcorex_gdn.so (fused CUDA kernel)
            ├─ MoE layers (all 36):
            │   ├─ PREFILL: corex_moe.py:339 → libcorex_moe.so (expert-grouped-wmma)
            │   └─ DECODE:  corex_moe.py:249 → libcorex_moe.so (fused MoE decode)
            └─ Attention (32 of 36 layers):
                ├─ PREFILL: corex_fa2.py:333 → libcorex_fa2.so (packed FA2)
                └─ DECODE:  corex_fa2.py:225 → libcorex_fa2.so (paged decode)
```

### Our Sub508 (no CoreX) — d01_basic_nostream: 95.87s (11.3x slower)
```
serving_chat.py → create_chat_completion()
  → engine.generate()
    → model_runner.py:1119 execute_model()
      → qwen3_5.py:1369 Qwen3_5ForCausalLM.forward()  (52 lines shorter!)
        → qwen3_5.py:???? Qwen3_5Model.forward()
          → qwen3_5.py:???? Qwen3_5DecoderLayer.forward()
            ├─ GatedDeltaNet layers (4 of 36):
            │   ├─ PREFILL: pure PyTorch conv1d → matmul → softmax (NaN!)
            │   └─ DECODE:  pure PyTorch _torch_causal_conv1d_update
            ├─ MoE layers (all 36):
            │   ├─ PREFILL: PyTorch loop over unique_eids (SLOW)
            │   └─ DECODE:  PyTorch batched GEMM fallback
            └─ Attention (32 of 36 layers):
                ├─ PREFILL: xformers _run_sdpa_fallback (patched, matmul+softmax)
                └─ DECODE:  xformers _run_sdpa_fallback
```

## 3. The Crash Chain (Sub508/509 → Score 0)

```
FUNCTIONAL TEST SEQUENCE:
d01_basic_nostream  ✓ PASS (95.87s — slow but works)
d02_stream_usage    ✓ PASS (1.84s)
d03_tool_call       ✗ FAIL (49.04s — model thinks instead of emitting tool XML)
d04_reasoning       ✓ PASS (128.74s)
   ... more tests pass ...
t2_n_2              ✗ FAIL → HTTP 500 → ENGINE PROCESS DIES
   ↓
t3_max_tokens_none  ✗ FAIL → HTTP 500 (engine dead, Connection Refused)
t3_max_tokens_1     ✗ FAIL → HTTP 500
t3_max_tokens_64    ✗ FAIL → HTTP 500
   ... 25 more tests ...
t16c_empty_messages ✗ FAIL → HTTP 500
───────────────────────────────────────
functional score: 21/51 = 0.412 (passed before crash)

case_truncation     → Connection Refused → score=0.0
replay_tencent      → 881/881 Connection Refused → score=0.0
opencompass          → Connection Refused → score=0.0
───────────────────────────────────────
TOTAL: 0.0  (engine was dead for 90% of evaluation)
```

## 4. CoreX Dispatch Gap — The 52-Line Difference

Sub168's qwen3_5.py has ~1421 lines. Ours has 1369.
The missing ~52 lines are CoreX dispatch wrappers:

```python
# WHAT SUB168 HAS (reconstructed from log evidence):

# In GatedDeltaNet.__init__:
try:
    from vllm.model_executor.models.corex_gdn import CoreXGDN
    self._corex_gdn = CoreXGDN(...)  # loads libcorex_gdn.so
except ImportError:
    self._corex_gdn = None

# In GatedDeltaNet.forward() prefill path:
if self._corex_gdn is not None:
    result = self._corex_gdn.prefill(...)  # → corex_gdn.py:228
else:
    result = self._pytorch_prefill(...)    # our current pure PyTorch

# In Qwen3_5MoE.forward():
try:
    from vllm.model_executor.models.corex_moe import corex_moe_forward
    result = corex_moe_forward(...)        # → corex_moe.py:339
except:
    result = self._pytorch_moe_forward(...)  # our current loop
```

## 5. Environment Variables (already set in YAML)

```yaml
VLLM_COREX_GDN_LIBRARY: /usr/local/corex/lib64/libcorex_gdn.so
VLLM_COREX_MOE_LIBRARY: /usr/local/corex/lib64/libcorex_moe.so
VLLM_COREX_FA2_LIBRARY: /usr/local/corex/lib64/libcorex_fa2.so
```

These .so files exist in the base image. The Python wrappers
(`corex_gdn.py`, `corex_moe.py`, `corex_fa2.py`) also exist in
the base image at:
`/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models/`

**Our qwen3_5.py simply never imports them.**

## 6. What Needs To Happen

Add try/except CoreX dispatch in 3 places in qwen3_5.py:
1. `GatedDeltaNet.forward()` — prefill + decode paths
2. `Qwen3_5MoE.forward()` — prefill + decode MoE dispatch  
3. Attention — already handled by xformers patches (corex_fa2 is separate)

CCCL pattern: `dispatch_with_env` — try native kernel first, fallback on error.
Our Python equivalent: `try: corex_forward() except: pytorch_forward()`
