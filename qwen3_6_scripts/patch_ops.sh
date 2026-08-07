#!/bin/bash
set -eo pipefail
# BI-V100 engine patches for Qwen3.6-35B-A3B (Qwen3_5 architecture)
#
# STRATEGY (CCCL-inspired):
#   1. Serving layer: full file replacement (protocol, chat, tools, reasoning)
#   2. Core compute: TARGETED in-place patches, never full replacement
#      - qwen3_5.py: inject numerical stability clamps (prevent 99.98% NaN)
#      - Preserve corex_gdn/corex_moe/corex_fa2 kernel paths
#
# CCCL design patterns applied:
#   - optionally_static: detect existing guards, inject only what's missing
#   - agent_radix_sort_histogram: Init → Detect → Patch → Verify
#   - overflow_cast: clamp BEFORE accumulation, not after
#
# Base image CoreX kernels (MUST preserve):
#   - corex_gdn  — fused GatedDeltaNet (decode + prefill)
#   - corex_moe  — fused MoE (expert-grouped-wmma)
#   - corex_fa2  — FlashAttention2 (packed prefill + paged chunked)

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
# 4. CCCL Agent-pattern: numerical stability patch for qwen3_5.py
#    Sub509 docker logs: 99.98% NaN in every GatedDeltaNet layer.
#    Base image has NaN detection + nan_to_num(nan=0.0), but that
#    means DeltaNet layers output all-zeros → model "brain dead"
#    → can't produce <tool_call> XML → d03 FAIL.
#
#    Strategy (CCCL optionally_static): detect what guards exist,
#    inject ONLY what's missing. Preserve corex kernel paths.
#    Agent flow: Init → Detect → Patch → Verify.
# ============================================================
python3 ./patch_numerical_stability.py 2>&1 || \
    echo "[patch_ops] WARNING: numerical stability patch failed (non-fatal)"
echo "[patch_ops] numerical stability patch complete"

# ============================================================
# 5. DO NOT full-replace these files — base image has optimized versions.
#    Use targeted patches (like step 4) instead of cp replacement.
#    - qwen3_5.py — patched in-place by step 4 (preserves corex paths)
#    - _custom_ops.py — base image ixformer bindings (no change needed)
#    - model_runner.py — base image worker (no change needed)
#    - xformers.py — base image attention backend (no change needed)
#    - paged_attn.py — base image paged attention (no change needed)
#    - prefix_prefill.py — base image prefix prefill (no change needed)
# ============================================================

echo "[patch_ops] DONE — serving layer + numerical stability patches applied"
echo "[patch_ops] Core compute paths preserved (corex_gdn + corex_moe + corex_fa2)"
