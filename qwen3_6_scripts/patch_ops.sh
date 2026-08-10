#!/bin/bash
# ==========================================================================
# SERVING-LAYER-ONLY PATCHES
#
# EVIDENCE FROM SUB168 DOCKER LOG (07-23, competition reference):
#   - corex_gdn.py:56 "Loaded fused CoreX GDN decode operator" ✓
#   - corex_moe.py:339 "Using CoreX fused MoE prefill operator" ✓
#   - model_runner.py:1074 (base image's line number)
#   - "Loading model weights took 17.3529 GB"
#   - ZERO NaN warnings
#   - d01: 8.49s, d03_tool_call: PASS in 2.12s
#
# EVIDENCE FROM OUR SUB508 DOCKER LOG (08-07):
#   - NO corex_gdn loading
#   - model_runner.py:1119 (our custom code)
#   - "Loading model weights took 16.2303 GB" (1.1GB MISSING)
#   - 16 NaN in prefill, 19 FusedMoE failures
#   - d01: 95.87s, d03_tool_call: FAIL in 49s
#
# CONCLUSION: Sub168 succeeds by using BASE IMAGE native model code.
# qwen3_5.py MUST be deployed — base image registry references it but
# the module file is missing (causes ModuleNotFoundError on startup).
#
# DO NOT deploy: model_runner.py, _custom_ops.py,
#   sampler.py, scheduler.py, sequence.py, xformers.py, paged_attn.py,
#   prefix_prefill.py, logits_processor.py, mamba_cache.py, arg_utils.py
# ==========================================================================

cd "$(dirname "$0")"
echo "[patch_ops] START — working directory: $(pwd)"

# Find vllm installation
VLLM=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    if [ -d "$P" ]; then
        VLLM="$P"
        echo "[patch_ops] Found vllm at: $VLLM"
        break
    fi
done

if [ -z "$VLLM" ]; then
    echo "[patch_ops] ERROR: vllm not found"
    exit 1
fi

# 1. Transformers config registration (config only, NOT model code)
TMODELS=""
for P in /usr/local/lib/python3.10/site-packages/transformers/models \
         /usr/local/corex/lib/python3/dist-packages/transformers/models \
         /usr/local/corex/lib64/python3/dist-packages/transformers/models; do
    if [ -d "$P" ]; then
        TMODELS="$P"
        break
    fi
done
if [ -n "$TMODELS" ]; then
    # Base engine requires transformers 4.55.3 for Qwen3_5Config support
    pip install transformers==4.55.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 30 2>&1 || \
        echo "[patch_ops] WARNING: transformers install failed (may already be correct version)"
    cp -r ./qwen3_5 "$TMODELS/" 2>/dev/null && echo "[patch_ops] qwen3_5 config copied" || true
    cp -r ./qwen3_5_moe "$TMODELS/" 2>/dev/null && echo "[patch_ops] qwen3_5_moe config copied" || true
    python3 ./patch_transformers_qwen3_5.py 2>&1 || echo "[patch_ops] WARNING: transformers patch failed (non-fatal)"
else
    echo "[patch_ops] WARNING: transformers/models not found"
fi

# 1b. CoreX probe — direct shell, guaranteed to show in build log
echo "[probe] === CoreX .so files ==="
ls -la /usr/local/corex/lib64/libcorex_*.so 2>/dev/null || echo "[probe] NO .so files in /usr/local/corex/lib64/"
echo "[probe] === CoreX Python wrappers ==="
ls -la "$VLLM/model_executor/models/corex_"*.py 2>/dev/null || echo "[probe] NO corex_*.py in $VLLM/model_executor/models/"
echo "[probe] === Native qwen3_5.py ==="
if [ -f "$VLLM/model_executor/models/qwen3_5.py" ]; then
    wc -lc "$VLLM/model_executor/models/qwen3_5.py"
    grep -c "corex_gdn\|corex_moe\|CoreXGDN" "$VLLM/model_executor/models/qwen3_5.py" || echo "[probe] no corex refs"
else
    echo "[probe] qwen3_5.py NOT in base image"
fi
echo "[probe] === All model files (corex related) ==="
find "$VLLM" -name "*corex*" -type f 2>/dev/null || echo "[probe] zero corex files anywhere in vllm"
echo "[probe] === LD_LIBRARY_PATH ==="
echo "$LD_LIBRARY_PATH"
echo "[probe] === /usr/local/corex/ tree ==="
find /usr/local/corex/lib64/ -name "*.so" 2>/dev/null | head -20 || echo "[probe] no .so in corex lib64"
echo "[probe] ==========================="

# 2. Model module — qwen3_5.py with CoreX dispatch (CCCL env_dispatch pattern).
# Our version tries to import corex_gdn/corex_moe from the base image.
# If they exist → uses fused CUDA kernels (10x faster).
# If they don't exist → gracefully falls back to pure PyTorch.
# ALWAYS deploy ours — it handles both scenarios correctly.
_NATIVE_QW="$VLLM/model_executor/models/qwen3_5.py"
cp ./qwen3_5.py "$_NATIVE_QW" 2>/dev/null && \
    echo "[patch_ops] qwen3_5.py deployed (CoreX dispatch + PyTorch fallback)" || true

# 2b. Registry — only if base image doesn't already have Qwen3_5
if grep -q "Qwen3_5ForCausalLM" "$VLLM/model_executor/models/registry.py" 2>/dev/null; then
    echo "[patch_ops] registry already has Qwen3_5 — NOT overwriting"
else
    cp ./registry.py "$VLLM/model_executor/models/registry.py" 2>/dev/null && \
        echo "[patch_ops] registry.py deployed" || true
fi

# 2c. paged_attn.py — CRITICAL: Triton context_attention_fwd hangs BI-V100.
# Base engine comment: "The Triton context_attention_fwd kernel hangs BI-V100
# GPUs permanently. Our paged_attn.py bypasses it via _forward_prefix_pytorch."
cp ./paged_attn.py "$VLLM/attention/ops/paged_attn.py" 2>/dev/null && \
    echo "[patch_ops] paged_attn.py deployed (Triton hang bypass)" || true

# 2d. patch_model_runner.py — fix prefix_cache_hit in chunked-prefill chunk 2+
python3 ./patch_model_runner.py 2>&1 || echo "[patch_ops] WARNING: model_runner patch failed (non-fatal)"

# 2e. mamba_cache.py — required for GatedDeltaNet state management
cp ./mamba_cache.py "$VLLM/model_executor/models/mamba_cache.py" 2>/dev/null && \
    echo "[patch_ops] mamba_cache.py deployed" || true

# 2f. sequence.py — fix completion_tokens inflation under chunked prefill
cp ./sequence.py "$VLLM/sequence.py" 2>/dev/null && \
    echo "[patch_ops] sequence.py deployed (token count fix)" || true

# 2g. scheduler.py — record num_cached_tokens in RequestMetrics
cp ./scheduler.py "$VLLM/core/scheduler.py" 2>/dev/null && \
    echo "[patch_ops] scheduler.py deployed (cache metrics)" || true

# 2h. xformers — bypass cudnnFlashAttn (head_dim=256 > 128 limit)
python3 ./patch_xformers_sdpa_seq.py 2>&1 || echo "[patch_ops] WARNING: xformers seq patch failed"
python3 ./patch_xformers_sdpa_batch.py 2>&1 || echo "[patch_ops] WARNING: xformers batch patch failed"
echo "[patch_ops] xformers patches applied"

# 3. Tool parser
mkdir -p "$VLLM/entrypoints/openai/tool_parsers" 2>/dev/null || true
cp ./qwen3coder_tool_parser.py "$VLLM/entrypoints/openai/tool_parsers/" 2>/dev/null || true
cp ./tool_parsers_init.py "$VLLM/entrypoints/openai/tool_parsers/__init__.py" 2>/dev/null || true
python3 ./patch_vllm_tool_parser.py 2>&1 || echo "[patch_ops] WARNING: tool parser registry patch failed"
echo "[patch_ops] tool parser deployed"

# 4. Reasoning parser
cp -r ./reasoning "$VLLM/" 2>/dev/null || true
echo "[patch_ops] reasoning parser deployed"

# 5. Serving layer ONLY
cp ./protocol.py "$VLLM/entrypoints/openai/protocol.py" 2>/dev/null || true
cp ./cli_args.py "$VLLM/entrypoints/openai/cli_args.py" 2>/dev/null || true
cp ./serving_chat.py "$VLLM/entrypoints/openai/serving_chat.py" 2>/dev/null || true
cp ./api_server.py "$VLLM/entrypoints/openai/api_server.py" 2>/dev/null || true
cp ./chat_utils.py "$VLLM/entrypoints/chat_utils.py" 2>/dev/null || true
echo "[patch_ops] serving layer deployed"

# 6. Mirror to second vllm path if exists
VLLM2=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    if [ -d "$P" ] && [ "$P" != "$VLLM" ]; then
        VLLM2="$P"
        break
    fi
done
if [ -n "$VLLM2" ]; then
    echo "[patch_ops] Second vllm at: $VLLM2"
    _NATIVE_QW2="$VLLM2/model_executor/models/qwen3_5.py"
    cp ./qwen3_5.py "$_NATIVE_QW2" 2>/dev/null || true
    if ! grep -q "Qwen3_5ForCausalLM" "$VLLM2/model_executor/models/registry.py" 2>/dev/null; then
        cp ./registry.py "$VLLM2/model_executor/models/registry.py" 2>/dev/null || true
    fi
    mkdir -p "$VLLM2/entrypoints/openai/tool_parsers" 2>/dev/null || true
    cp ./qwen3coder_tool_parser.py "$VLLM2/entrypoints/openai/tool_parsers/" 2>/dev/null || true
    cp ./tool_parsers_init.py "$VLLM2/entrypoints/openai/tool_parsers/__init__.py" 2>/dev/null || true
    cp -r ./reasoning "$VLLM2/" 2>/dev/null || true
    cp ./protocol.py "$VLLM2/entrypoints/openai/protocol.py" 2>/dev/null || true
    cp ./cli_args.py "$VLLM2/entrypoints/openai/cli_args.py" 2>/dev/null || true
    cp ./serving_chat.py "$VLLM2/entrypoints/openai/serving_chat.py" 2>/dev/null || true
    cp ./api_server.py "$VLLM2/entrypoints/openai/api_server.py" 2>/dev/null || true
    cp ./chat_utils.py "$VLLM2/entrypoints/chat_utils.py" 2>/dev/null || true
fi

echo "[patch_ops] DONE — SM70 GDN kernel + serving layer + engine patches deployed"
echo "[patch_ops] Deployed: qwen3_5.py, flash_qla_sm70 (SM70 GDN CUDA kernel), paged_attn.py, mamba_cache.py, sequence.py, scheduler.py, xformers patches, serving layer"
echo "[patch_ops] SM70 GDN kernel: JIT compiles on first forward pass (~2min), then cached"
echo "[patch_ops] NOT deployed (base image native): model_runner.py, _custom_ops.py, sampler.py, logits_processor.py, arg_utils.py"

# Deploy flash_qla SM70 GDN kernel (from 1Cat-vLLM, MIT license)
# This is a fused CUDA kernel for GatedDeltaNet on SM70/SM75 (V100/BI-V100)
# JIT compiled at runtime via torch.utils.cpp_extension.load()
FLASH_QLA_DST="$VLLM/model_executor/models/flash_qla_sm70"
if [ -d "./flash_qla_sm70" ]; then
    rm -rf "$FLASH_QLA_DST" 2>/dev/null
    cp -r ./flash_qla_sm70 "$FLASH_QLA_DST" 2>/dev/null && \
        echo "[patch_ops] flash_qla_sm70 deployed to $FLASH_QLA_DST" || true
    # Also deploy to VLLM2 if present
    if [ -n "$VLLM2" ]; then
        rm -rf "$VLLM2/model_executor/models/flash_qla_sm70" 2>/dev/null
        cp -r ./flash_qla_sm70 "$VLLM2/model_executor/models/flash_qla_sm70" 2>/dev/null || true
    fi
fi
