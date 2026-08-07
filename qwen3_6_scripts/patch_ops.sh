#!/bin/bash
set -eo pipefail
# BI-V100 engine patches for Qwen3.6-35B-A3B (Qwen3_5 architecture)
#
# STRATEGY: Only patch serving/protocol layer. NEVER replace core compute
# files (qwen3_5.py model, _custom_ops.py, model_runner.py, xformers.py,
# paged_attn.py, prefix_prefill.py, logits_processor.py, sampler.py).
#
# The base image has optimized CoreX kernels:
#   - corex_gdn.py  — fused GatedDeltaNet (decode + prefill)
#   - corex_moe.py  — fused MoE (expert-grouped-wmma)
#   - corex_fa2.py  — FlashAttention2 (packed prefill + paged chunked)
# Replacing model files breaks these kernel paths and causes:
#   - DeltaNet NaN (99.98% of activations) → model output garbage
#   - MoE fallback to pure PyTorch → 10x slower
#   - FA2 → XFormers fallback → slower attention
#
# Reference: competitor sub168 uses base image qwen3_5.py + these CoreX
# kernels and achieves d03_tool_call in 2.12s (vs our sub509's 49s FAIL).

cd "$(dirname "$0")"
echo "[patch_ops] working directory: $(pwd)"

VLLM=/usr/local/corex/lib/python3/dist-packages/vllm
VLLM64=/usr/local/corex/lib64/python3/dist-packages/vllm

TARGETS=()
if [ -d "$VLLM" ]; then
    TARGETS+=("$VLLM")
fi
if [ -d "$VLLM64" ]; then
    TARGETS+=("$VLLM64")
fi

if [ ${#TARGETS[@]} -eq 0 ]; then
    echo "[patch_ops] ERROR: vllm not found at lib or lib64 path"
    exit 1
fi

echo "[patch_ops] vllm paths found: ${TARGETS[*]}"

deploy() {
    local src="$1"
    local rel_dst="$2"
    for V in "${TARGETS[@]}"; do
        local dst="$V/$rel_dst"
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
    done
}

# ============================================================
# 1. Transformers: register Qwen3_5 / Qwen3_5_MoE model types
# ============================================================
pip install transformers==4.55.3 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
pip install transformers==4.55.3 2>/dev/null || \
echo "[patch_ops] WARNING: pip install transformers failed, using pre-installed version"
cp -r ./qwen3_5 /usr/local/lib/python3.10/site-packages/transformers/models/
cp -r ./qwen3_5_moe /usr/local/lib/python3.10/site-packages/transformers/models/
python3 ./patch_transformers_qwen3_5.py
echo "[patch_ops] transformers Qwen3_5 models installed"

# ============================================================
# 2. Model registry: ensure qwen3_5 is registered in vllm
# ============================================================
deploy ./registry.py "model_executor/models/registry.py"
echo "[patch_ops] registry.py deployed"

# ============================================================
# 3. Serving layer patches (protocol, chat, tool parsing, reasoning)
# ============================================================

# --- Tool parser: Qwen3 XML tool call format ---
for V in "${TARGETS[@]}"; do
    cp ./qwen3coder_tool_parser.py "$V/entrypoints/openai/tool_parsers/"
    cp ./tool_parsers_init.py "$V/entrypoints/openai/tool_parsers/__init__.py"
done
echo "[patch_ops] qwen3_coder tool parser deployed"

# --- Reasoning parser + serving files ---
for V in "${TARGETS[@]}"; do
    cp -r ./reasoning "$V/"
    cp ./protocol.py "$V/entrypoints/openai/protocol.py"
    cp ./cli_args.py "$V/entrypoints/openai/cli_args.py"
    cp ./serving_chat.py "$V/entrypoints/openai/serving_chat.py"
    cp ./api_server.py "$V/entrypoints/openai/api_server.py"
    cp ./chat_utils.py "$V/entrypoints/chat_utils.py"
done
echo "[patch_ops] reasoning parser + serving files installed"

# ============================================================
# 4. DO NOT PATCH sequence.py or scheduler.py
#    168 (reference competitor) did not patch these.
#    Our custom versions may conflict with base image internals.
#    Token counting fixes are minor; NaN-free output is critical.
# ============================================================

# ============================================================
# 5. DO NOT PATCH these files — base image has optimized versions:
#    - qwen3_5.py (model) — has corex_gdn/corex_moe/corex_fa2 integration
#    - _custom_ops.py — base image ixformer bindings
#    - model_runner.py — base image worker
#    - xformers.py — base image attention backend
#    - paged_attn.py — base image paged attention
#    - prefix_prefill.py — base image prefix prefill
#    - logits_processor.py — base image logits
#    - sampler.py — base image sampler
#    - arg_utils.py — base image arg parsing
#    - paged_attention_v2_pytorch.py — not needed with native kernels
# ============================================================

echo "[patch_ops] DONE — serving-layer-only patches applied"
echo "[patch_ops] Core compute files preserved from base image (corex_gdn + corex_moe + corex_fa2)"
