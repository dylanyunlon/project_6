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
    cp -r ./qwen3_5 "$TMODELS/" 2>/dev/null && echo "[patch_ops] qwen3_5 config copied" || true
    cp -r ./qwen3_5_moe "$TMODELS/" 2>/dev/null && echo "[patch_ops] qwen3_5_moe config copied" || true
    python3 ./patch_transformers_qwen3_5.py 2>&1 || echo "[patch_ops] WARNING: transformers patch failed (non-fatal)"
else
    echo "[patch_ops] WARNING: transformers/models not found"
fi

# 2. Model module — qwen3_5.py MUST exist for registry to import.
# Base image registry lists Qwen3_5ForCausalLM/Qwen3_5MoeForCausalLM
# but the actual module file may be missing (causes ModuleNotFoundError
# on startup: "No module named 'vllm.model_executor.models.qwen3_5'").
# Deploy our qwen3_5.py so the module can be imported.
cp ./qwen3_5.py "$VLLM/model_executor/models/qwen3_5.py" 2>/dev/null && \
    echo "[patch_ops] qwen3_5.py deployed (model module)" || true

# 2b. Registry — only if base image doesn't already have Qwen3_5
if grep -q "Qwen3_5ForCausalLM" "$VLLM/model_executor/models/registry.py" 2>/dev/null; then
    echo "[patch_ops] registry already has Qwen3_5 — NOT overwriting"
else
    cp ./registry.py "$VLLM/model_executor/models/registry.py" 2>/dev/null && \
        echo "[patch_ops] registry.py deployed" || true
fi

# 3. Tool parser
mkdir -p "$VLLM/entrypoints/openai/tool_parsers" 2>/dev/null || true
cp ./qwen3coder_tool_parser.py "$VLLM/entrypoints/openai/tool_parsers/" 2>/dev/null || true
cp ./tool_parsers_init.py "$VLLM/entrypoints/openai/tool_parsers/__init__.py" 2>/dev/null || true
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
    cp ./qwen3_5.py "$VLLM2/model_executor/models/qwen3_5.py" 2>/dev/null || true
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

echo "[patch_ops] DONE — serving layer + qwen3_5.py model module deployed"
echo "[patch_ops] Deployed: qwen3_5.py (model module, required for registry import)"
echo "[patch_ops] NOT deployed (base image native): model_runner.py, _custom_ops.py, sampler.py, scheduler.py, sequence.py, xformers.py, paged_attn.py, prefix_prefill.py, logits_processor.py, mamba_cache.py, arg_utils.py"
