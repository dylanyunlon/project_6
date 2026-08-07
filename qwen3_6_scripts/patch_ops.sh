#!/bin/bash
set -eo pipefail
# BI-V100 engine patches for Qwen3.6-35B-A3B (Qwen3_5 architecture)
#
# All modifications are FULL FILE REPLACEMENTS — no AST patch scripts.
# Each file was read in full from the base image vllm source, modified
# with the necessary fixes, and placed here as a complete copy.
#
# Base image: git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3
# vllm install path: /usr/local/corex/lib/python3/dist-packages/vllm/

# CRITICAL: cd into this script's directory so all ./relative paths work
# regardless of WORKDIR in Dockerfile or caller's cwd.
cd "$(dirname "$0")"
echo "[patch_ops] working directory: $(pwd)"

VLLM=/usr/local/corex/lib/python3/dist-packages/vllm
VLLM64=/usr/local/corex/lib64/python3/dist-packages/vllm

# Deploy to ALL existing vllm paths — Python may load from either one
# depending on PYTHONPATH ordering and namespace package resolution.
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

# Helper: copy file to all target vllm roots
deploy() {
    local src="$1"
    local rel_dst="$2"  # relative path within vllm, e.g. "attention/ops/paged_attn.py"
    for V in "${TARGETS[@]}"; do
        local dst="$V/$rel_dst"
        mkdir -p "$(dirname "$dst")"
        cp "$src" "$dst"
    done
}

# --- _custom_ops.py: SMEM 48KB fix + hardware ops bindings -------------------
# Base image returns 32KB (32768) for get_max_shared_memory_per_block, but
# BI-V100 actually has 48KB (49152) confirmed via ixsmi. This limits Triton
# tile sizes and ixformer internal allocations if not corrected.
# CCCL GridEvenShare test (catch2_test_grid_even_share.cu) validates that
# work distribution depends on correct hardware parameters — wrong SMEM
# means wrong tile_size means wrong grid_size.
# FULL FILE REPLACEMENT.
deploy ./_custom_ops.py "_custom_ops.py"
echo "[patch_ops] _custom_ops.py → / (SMEM 32KB→48KB fix)"

# --- paged_attn.py: pure-PyTorch attention fallback --------------------------
deploy ./paged_attn.py "attention/ops/paged_attn.py"
echo "[patch_ops] paged_attn.py → attention/ops/"

# --- prefix_prefill.py: Triton-free prefix attention -------------------------
deploy ./prefix_prefill.py "attention/ops/prefix_prefill.py"
echo "[patch_ops] prefix_prefill.py → attention/ops/"

# --- model_runner.py: prefix_cache_hit fix -----------------------------------
deploy ./model_runner.py "worker/model_runner.py"
echo "[patch_ops] model_runner.py → worker/"

# --- xformers.py: head_dim>128 fallback + Q-tiling --------------------------
deploy ./xformers.py "attention/backends/xformers.py"
echo "[patch_ops] xformers.py → attention/backends/"

# --- arg_utils.py: disable auto chunked-prefill for 32K+ --------------------
deploy ./arg_utils.py "engine/arg_utils.py"
echo "[patch_ops] arg_utils.py → engine/"

# --- logits_processor.py: seq_groups=None guard ------------------------------
deploy ./logits_processor.py "model_executor/layers/logits_processor.py"
echo "[patch_ops] logits_processor.py → model_executor/layers/"

# --- sampler.py: CCCL-ported top-k fast path for sampling --------------------
deploy ./sampler.py "model_executor/layers/sampler.py"
echo "[patch_ops] sampler.py → model_executor/layers/"

# --- transformers: Qwen3_5 tokenizer / model files --------------------------
# NOTE: patch_transformers_qwen3_5.py is the ONLY remaining patch script.
# It modifies pip-installed transformers' configuration_auto.py and __init__.py
# to register qwen3_5/qwen3_5_moe. These files come from pip (version-specific)
# so we can't pre-copy them — the patch script inserts lines after known anchors.
pip install transformers==4.55.3 -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null || \
pip install transformers==4.55.3 2>/dev/null || \
echo "[patch_ops] WARNING: pip install transformers failed, using pre-installed version"
cp -r ./qwen3_5 /usr/local/lib/python3.10/site-packages/transformers/models/
cp -r ./qwen3_5_moe /usr/local/lib/python3.10/site-packages/transformers/models/
python3 ./patch_transformers_qwen3_5.py
echo "[patch_ops] transformers Qwen3_5 models installed"

# --- vllm model: Qwen3.6 (Qwen3_5 arch) ------------------------------------
for V in "${TARGETS[@]}"; do
    cp ./mamba_cache.py "$V/model_executor/models/"
done
deploy ./qwen3_5.py "model_executor/models/qwen3_5.py"
deploy ./registry.py "model_executor/models/registry.py"
echo "[patch_ops] qwen3_5.py + registry.py deployed"

# --- paged_attention_v2_pytorch.py: PyTorch V2 attention fallback ------------
for V in "${TARGETS[@]}"; do
    cp ./paged_attention_v2_pytorch.py "$V/paged_attention_v2_pytorch.py"
done
cp ./paged_attention_v2_pytorch.py /workspace/paged_attention_v2_pytorch.py
echo "[patch_ops] paged_attention_v2_pytorch.py → all paths + /workspace/"

# --- sequence.py: fix completion_tokens inflation ----------------------------
deploy ./sequence.py "sequence.py"
echo "[patch_ops] sequence.py → /"

# --- scheduler.py: record num_cached_tokens ---------------------------------
deploy ./scheduler.py "core/scheduler.py"
echo "[patch_ops] scheduler.py → core/"

# --- tool parser: Qwen3 XML tool call format --------------------------------
for V in "${TARGETS[@]}"; do
    cp ./qwen3coder_tool_parser.py "$V/entrypoints/openai/tool_parsers/"
    cp ./tool_parsers_init.py "$V/entrypoints/openai/tool_parsers/__init__.py"
done
echo "[patch_ops] qwen3_coder tool parser deployed"

# --- reasoning parser: Qwen3 <think>...</think> split -----------------------
for V in "${TARGETS[@]}"; do
    cp -r ./reasoning "$V/"
    cp ./protocol.py "$V/entrypoints/openai/protocol.py"
    cp ./cli_args.py "$V/entrypoints/openai/cli_args.py"
    cp ./serving_chat.py "$V/entrypoints/openai/serving_chat.py"
    cp ./api_server.py "$V/entrypoints/openai/api_server.py"
    cp ./chat_utils.py "$V/entrypoints/chat_utils.py"
done
echo "[patch_ops] reasoning parser + serving files installed"

echo "[patch_ops] DONE — all patches applied via full file replacement"
