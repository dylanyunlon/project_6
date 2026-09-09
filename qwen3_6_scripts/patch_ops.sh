echo "[build] trigger 20260901"
echo "[build] trigger 202609011237"
echo "[build] trigger 202609011636"
echo "[build] trigger 202609011655"
echo "[build] trigger 202609011710"
echo "[build] trigger 202609031035"
echo "[build] trigger 202609031049"
#!/usr/bin/env bash
# BI-V100 patch script for Qwen3.6-35B-A3B (Qwen3_5 MoE architecture)
#
# Triton situation on BI-V100:
#   - Standard Triton 2.3.1 is already present in the image.
#   - HAS_TRITON = False (hardcoded in vendor vllm), but Triton is still used
#     for TP-mode cache management (custom_cache_manager / libentry).
#   - The vendor's triton_utils/__init__.py, custom_cache_manager.py, libentry.py
#     are already correct for standard Triton 2.3.1 — do NOT overwrite them.
#   - DO NOT install BI-V150 corex Triton 2.1.0 (pkgs/triton): that causes
#     GPU hang on BI-V100 because the Triton CUDA PTX kernels are incompatible.

# Recommended server start command for TP=4 support 256K, needs chunked prefill
# CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3.6-35B-A3B --port 1111 --served-model-name llm \
#     --max-model-len 262144 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \
#     --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
#     --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
#     --max-seq-len-to-capture 32768 --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder --reasoning-parser qwen3
#
# With prefix caching (GDN align-mode, requires chunked prefill):
# CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3.6-35B-A3B --port 1111 --served-model-name llm \
#     --max-model-len 262144 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \
#     --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
#     --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
#     --max-seq-len-to-capture 32768 --enable-auto-tool-choice \
#     --tool-call-parser qwen3_coder --reasoning-parser qwen3

set -eo pipefail

# cd into this script's directory so ./relative paths work
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "[patch_ops] working directory: $(pwd)"

build_stage() { printf '[BI100 BUILD] %s\n' "$1" >&2; }
require_file() {
    local path=$1
    [[ -f "$path" ]] || {
        printf 'required patch source is missing: %s\n' "$path" >&2
        exit 2
    }
}
install_patch_file() {
    local source=$1
    local target=$2

    require_file "$source"
    mkdir -p "$(dirname "$target")"
    install -m 0644 "$source" "$target"
}

build_stage "patch script entered"

build_stage "checking offline transformers dependency"
# --- transformers: Qwen3_5 tokenizer / model files --------------------------
TRANSFORMERS_REQUIRED_VERSION="4.55.3"
if ! python3 - "$TRANSFORMERS_REQUIRED_VERSION" <<'PY'
import importlib.metadata
import sys

required = sys.argv[1]
try:
    installed = importlib.metadata.version("transformers")
except importlib.metadata.PackageNotFoundError:
    raise SystemExit(1)
raise SystemExit(0 if installed == required else 1)
PY
then
  WHEEL_DIR="./wheels"
  if ! ls "${WHEEL_DIR}/transformers-${TRANSFORMERS_REQUIRED_VERSION}"*.whl >/dev/null 2>&1; then
    echo "transformers ${TRANSFORMERS_REQUIRED_VERSION} is required, but no offline wheel was found in ${WHEEL_DIR}" >&2
    exit 2
  fi
  python3 -m pip install --no-index --no-deps --find-links="${WHEEL_DIR}" \
    "transformers==${TRANSFORMERS_REQUIRED_VERSION}"
fi

python3 - "$TRANSFORMERS_REQUIRED_VERSION" <<'PY'
import importlib.metadata
import sys

required = sys.argv[1]
installed = importlib.metadata.version("transformers")
if installed != required:
    raise SystemExit(
        f"transformers version mismatch: expected {required}, got {installed}")
print(f"[ok] transformers {installed}")
PY

build_stage "discovering Python package roots"
python3 - <<'PY' > /tmp/qwen36_patch_paths.env
from patch_utils import package_root, shell_env_line

print(shell_env_line("VLLM_ROOT", package_root("vllm")))
print(shell_env_line("TRANSFORMERS_ROOT", package_root("transformers")))
PY
source /tmp/qwen36_patch_paths.env

echo "VLLM_ROOT=${VLLM_ROOT}"
echo "TRANSFORMERS_ROOT=${TRANSFORMERS_ROOT}"
[[ -d "$VLLM_ROOT" ]] || {
    printf 'vLLM root does not exist: %s\n' "$VLLM_ROOT" >&2
    exit 2
}

VLLM_OVERRIDE_ROOT="./vendor_overrides/vllm"
[[ -d "$VLLM_OVERRIDE_ROOT" ]] || {
    printf 'vLLM override directory missing: %s\n' "$VLLM_OVERRIDE_ROOT" >&2
    exit 2
}

build_stage "installing authoritative vLLM core block overrides"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/interfaces.py" \
    "${VLLM_ROOT}/core/interfaces.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/evictor_v2.py" \
    "${VLLM_ROOT}/core/evictor_v2.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/cpu_kv_content_cache.py" \
    "${VLLM_ROOT}/core/block/cpu_kv_content_cache.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/cpu_gpu_block_allocator.py" \
    "${VLLM_ROOT}/core/block/cpu_gpu_block_allocator.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/prefix_caching_block.py" \
    "${VLLM_ROOT}/core/block/prefix_caching_block.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block/block_table.py" \
    "${VLLM_ROOT}/core/block/block_table.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/core/block_manager_v2.py" \
    "${VLLM_ROOT}/core/block_manager_v2.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/sampling_params.py" \
    "${VLLM_ROOT}/sampling_params.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/model_executor/sampling_metadata.py" \
    "${VLLM_ROOT}/model_executor/sampling_metadata.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/model_executor/layers/sampler.py" \
    "${VLLM_ROOT}/model_executor/layers/sampler.py"

build_stage "installing BI100-DP data parallel overrides"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/config.py" \
    "${VLLM_ROOT}/config.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/engine/arg_utils.py" \
    "${VLLM_ROOT}/engine/arg_utils.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/engine/llm_engine.py" \
    "${VLLM_ROOT}/engine/llm_engine.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/executor/mp_distributed_executor.py" \
    "${VLLM_ROOT}/executor/mp_distributed_executor.py"
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/worker/worker.py" \
    "${VLLM_ROOT}/worker/worker.py"

build_stage "installing hash-pinned CoreX 3.2.3 extensions"
bash ./install_prebuilt_corex.sh "${VLLM_ROOT}"

build_stage "installing BI100 runtime modules"
cp ./bi100_env.py "${VLLM_ROOT}/bi100_env.py"
cp ./bi100_profile.py "${VLLM_ROOT}/bi100_profile.py"
cp ./block_major_kv_cache.py "${VLLM_ROOT}/block_major_kv_cache.py"
cp ./gdn_prefix.py "${VLLM_ROOT}/gdn_prefix.py"
cp ./ep_fused_moe_patch.py "${VLLM_ROOT}/ep_fused_moe_patch.py"

build_stage "installing CoreX paged-KV swap compatibility"
cp ./_custom_ops.py "${VLLM_ROOT}/_custom_ops.py"
cp ./cache_engine.py "${VLLM_ROOT}/worker/cache_engine.py"
# worker swap order + block_major capacity + startup profile guard
# are pre-merged into vendor_overrides/vllm/worker/worker.py

# --- paged_attn.py: replace forward_prefix with pure-PyTorch fallback -------
# The Triton context_attention_fwd kernel hangs BI-V100 GPUs permanently
# (standard Triton 2.3.1 PTX is not supported by the corex runtime either).
# Our paged_attn.py bypasses it entirely via _forward_prefix_pytorch, which
# utilizes K-tiling techniques, and also have _forward_decode_pytorch to bypass kernel
# when context length is high
cp ./paged_attn.py "${VLLM_ROOT}/attention/ops/paged_attn.py"

# --- model_runner.py: fix prefix_cache_hit stays True in chunked-prefill chunk 2+ ---
# Bug: _compute_for_prefix_cache_hit Case 1 (prefix_cache_len <= context_len)
# leaves prefix_cache_hit=True. Then _add_seq_group uses block_table=computed_block_nums
# (only the original prefix blocks), ignoring chunk-1 KV cache blocks.
# _forward_prefix_pytorch then gets an undersized block_tables and crashes with
# "amax(): Expected reduction dim -1 to have non-zero size" on the 2nd tile.
# Fix: set prefix_cache_hit=False for Case 1 so the full block_tables is used.
cp ./model_runner.py "${VLLM_ROOT}/worker/model_runner.py"

build_stage "installing distributed module overrides (task 05/20)"
DIST_OVERRIDE_ROOT="./distributed_override"
if [[ -d "$DIST_OVERRIDE_ROOT" ]]; then
    # Top-level distributed files
    for f in __init__.py communication_op.py parallel_state.py utils.py; do
        install_patch_file "${DIST_OVERRIDE_ROOT}/${f}" "${VLLM_ROOT}/distributed/${f}"
    done

    # device_communicators (modified + new)
    for f in base_device_communicator.py cpu_communicator.py cuda_communicator.py \
             cuda_wrapper.py custom_all_reduce.py custom_all_reduce_utils.py \
             hpu_communicator.py neuron_communicator.py pynccl.py \
             pynccl_wrapper.py shm_broadcast.py tpu_communicator.py \
             xpu_communicator.py; do
        install_patch_file "${DIST_OVERRIDE_ROOT}/device_communicators/${f}" \
            "${VLLM_ROOT}/distributed/device_communicators/${f}"
    done

    # kv_transfer directory
    cp -r "${DIST_OVERRIDE_ROOT}/kv_transfer" "${VLLM_ROOT}/distributed/"

    # platforms (required by new distributed: get_device_communicator_cls, is_fully_connected)
    if [[ -d "${DIST_OVERRIDE_ROOT}/platforms" ]]; then
        for f in __init__.py interface.py cuda.py cpu.py rocm.py tpu.py xpu.py hpu.py neuron.py; do
            [[ -f "${DIST_OVERRIDE_ROOT}/platforms/${f}" ]] && \
                install_patch_file "${DIST_OVERRIDE_ROOT}/platforms/${f}" "${VLLM_ROOT}/platforms/${f}"
        done
    fi
fi

build_stage "installing executor startup diagnostics"
# executor startup debug + worker startup profile guard + block_major capacity
# are pre-merged into vendor_overrides and whole-file copies
cp ./multiproc_worker_utils.py "${VLLM_ROOT}/executor/multiproc_worker_utils.py"

build_stage "installing transformers Qwen3.5 model support"
cp -r ./qwen3_5 "${TRANSFORMERS_ROOT}/models/"
cp -r ./qwen3_5_moe "${TRANSFORMERS_ROOT}/models/"
python3 ./patch_transformers_qwen3_5.py

build_stage "installing vLLM Qwen3.6 model implementation"
# --- vllm model: Qwen3.6-35B-A3B (Qwen3_5 MoE arch) -------------------------
cp ./mamba_cache.py "${VLLM_ROOT}/model_executor/models/"
cp ./qwen3_5.py "${VLLM_ROOT}/model_executor/models/qwen3_5.py"
cp ./registry.py "${VLLM_ROOT}/model_executor/models/registry.py"

# --- sequence.py: fix completion_tokens inflation under chunked prefill ------
# Bug: get_output_token_ids_to_return(delta=True) with num_new_tokens=0
# returns _cached_all_token_ids[-0:] == [0:] (the ENTIRE prompt+output list).
# Each prefill chunk step adds prompt_len to previous_num_tokens, so a 10K
# prompt processed in 3 chunks inflates completion_tokens by ~30K.
# Also adds num_cached_tokens field to RequestMetrics for prefix-cache stats.
cp ./sequence.py "${VLLM_ROOT}/sequence.py"

# --- scheduler.py: record num_cached_tokens in RequestMetrics ----------------
# Reports only the longest prefix backed by both live KV blocks and an exact
# GDN restore state. Raw KV-only hits must not inflate cached_tokens.
# serving_chat.py exposes the value in the OpenAI-compatible usage details.
cp ./scheduler.py "${VLLM_ROOT}/core/scheduler.py"

build_stage "installing diagnostic initial allocation trace"
# block_manager_cache_trace is pre-merged into vendor_overrides/vllm/core/block_manager_v2.py
cp ./outputs.py "${VLLM_ROOT}/outputs.py"

build_stage "installing scheduler and attention patches"
# --- xformers: bypass cudnnFlashAttnForward (head_dim=256 > 128 limit) ------
# Injects _run_sdpa_fallback (pure matmul+softmax) into xformers.py.
# Required because head_dim=256 > 128 and ixformer flash attention either
# crashes (is_causal=True) or produces wrong output (attn_mask path).
# The fallback uses query_start_loc to derive actual query lengths, so it
# works correctly during profiling runs with chunked-prefill-style batches.
# also bypasses auto chunked prefill on
cp ./xformers.py "${VLLM_ROOT}/attention/backends/xformers.py"
cp ./logits_processor.py "${VLLM_ROOT}/model_executor/layers/logits_processor.py"
cp ./outlines_decoding.py "${VLLM_ROOT}/model_executor/guided_decoding/outlines_decoding.py"
# arg_utils.py xformers patches are pre-merged into vendor_overrides/vllm/engine/arg_utils.py
# bi100_timer profile instrumentation is pre-merged into xformers.py

build_stage "installing API parsers and serving modules"
# --- tool parser: Qwen3 XML tool call format ---------------------------------
# Registers "qwen3_coder" parser for Qwen3.6 XML-style tool calls:
#   <tool_call><function=name><parameter=key>\nvalue\n</parameter></function></tool_call>
# Use at server start: --tool-call-parser qwen3_coder --enable-auto-tool-choice
cp ./qwen3coder_tool_parser.py "${VLLM_ROOT}/entrypoints/openai/tool_parsers/"
cp ./tool_parsers__init__.py "${VLLM_ROOT}/entrypoints/openai/tool_parsers/__init__.py"

# --- reasoning parser: Qwen3 <think>...</think> split ------------------------
# Adds --reasoning-parser qwen3 support.
# Routes thinking tokens to reasoning_content, rest to content in the delta.
# Works together with --tool-call-parser qwen3_coder (think → tool call flow).
#
# PRD #69: Clear __pycache__ BEFORE copying patched .py files.
# Base image's compiled .pyc (protocol.py with extra="forbid") would otherwise
# shadow our patched .py, causing ~25% of requests to reject
# max_completion_tokens / reasoning_effort with HTTP 400.
find "${VLLM_ROOT}/entrypoints" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

cp -r ./reasoning "${VLLM_ROOT}/"
cp ./protocol.py "${VLLM_ROOT}/entrypoints/openai/protocol.py"
cp ./cli_args.py "${VLLM_ROOT}/entrypoints/openai/cli_args.py"
cp ./serving_chat.py "${VLLM_ROOT}/entrypoints/openai/serving_chat.py"
cp ./serving_tokenization.py \
    "${VLLM_ROOT}/entrypoints/openai/serving_tokenization.py"
cp ./api_server.py "${VLLM_ROOT}/entrypoints/openai/api_server.py"
cp ./chat_utils.py "${VLLM_ROOT}/entrypoints/chat_utils.py"
python3 - ./api_server.py \
        "${VLLM_ROOT}/entrypoints/openai/api_server.py" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_bytes()
installed = Path(sys.argv[2]).read_bytes()
if source != installed:
    raise SystemExit("runtime api_server overlay identity mismatch")
PY

build_stage "installing quantization layer overrides (task 13/20)"
# --- layers/quantization: new API (lazy imports, Fp8LinearOp, block quant,
#     EP support, ScaledMM kernels, ixformer MoE ops) -----------------------
# The vendor image ships an older quantization module whose interfaces are
# incompatible with the rest of the upgraded vLLM code (model_loader,
# attention, fused_moe all reference the new API).  We replace the entire
# subtree so every internal import resolves consistently.
QUANT_OVERRIDE_ROOT="${VLLM_OVERRIDE_ROOT}/model_executor/layers/quantization"
if [[ -d "$QUANT_OVERRIDE_ROOT" ]]; then
    # Wipe stale .pyc first so Python never loads cached old bytecode
    find "${VLLM_ROOT}/model_executor/layers/quantization" \
         -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
    # Copy the full tree (preserves new subdirs like kernels/scaled_mm,
    # kernels/mixed_precision, quark/, utils/configs/)
    cp -r "${QUANT_OVERRIDE_ROOT}/." \
          "${VLLM_ROOT}/model_executor/layers/quantization/"
fi

# --- quantization dependency: parameter.py (BlockQuantScaleParameter) -------
install_patch_file \
    "${VLLM_OVERRIDE_ROOT}/model_executor/parameter.py" \
    "${VLLM_ROOT}/model_executor/parameter.py"

# --- quantization dependency: fused_moe (BLOCK enum, EP create_weights) -----
MOE_OVERRIDE_ROOT="${VLLM_OVERRIDE_ROOT}/model_executor/layers/fused_moe"
if [[ -d "$MOE_OVERRIDE_ROOT" ]]; then
    for f in __init__.py layer.py cutlass_moe.py fused_moe.py fused_marlin_moe.py; do
        [[ -f "${MOE_OVERRIDE_ROOT}/${f}" ]] && \
            install_patch_file "${MOE_OVERRIDE_ROOT}/${f}" \
                "${VLLM_ROOT}/model_executor/layers/fused_moe/${f}"
    done
fi

build_stage "installing transformers_utils overrides (task 16/20)"
# --- transformers_utils: new API required by upgraded config.py, engine,
#     tokenizer_group, serving_chat, and chat_utils --------------------------
# The vendor image ships older transformers_utils whose interfaces are
# incompatible with the rest of the upgraded vLLM code:
#   config.py      – get_config() signature (no rope_scaling/rope_theta),
#                    patch_rope_scaling, uses_mrope, is_encoder_decoder,
#                    get_pooling_config, get_sentence_transformer_tokenizer_config,
#                    file_exists/file_or_path_exists signature, VllmConfig
#   tokenizer.py   – AnyTokenizer includes TokenizerBase, encode_tokens,
#                    decode_tokens, CachedTokenizer.max_token_id, tokenizer_mode="custom"
#   tokenizer_group – init_tokenizer_from_configs(lora_config=...) instead of enable_lora
#   tokenizers/    – MistralTokenizer(TokenizerBase), maybe_serialize_tool_calls
#   processor.py   – cached_get_processor
#   utils.py       – is_s3, maybe_model_redirect
#   s3_utils.py    – S3Model (new file)
#   tokenizer_base.py – TokenizerBase ABC + TokenizerRegistry (new file)
#   detokenizer_utils.py – extracted from detokenizer.py (new file)
#   configs/       – new model configs (Cohere2, DeepseekVLV2, H2OVL, Olmo2, etc.)
#   processors/    – DeepseekVLV2Processor (new directory)
TRANSFORMERS_UTILS_OVERRIDE="${VLLM_OVERRIDE_ROOT}/transformers_utils"
if [[ -d "$TRANSFORMERS_UTILS_OVERRIDE" ]]; then
    # Wipe stale .pyc first
    find "${VLLM_ROOT}/transformers_utils" \
         -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

    # Top-level files
    for f in __init__.py config.py detokenizer.py detokenizer_utils.py \
             processor.py tokenizer.py tokenizer_base.py utils.py s3_utils.py; do
        [[ -f "${TRANSFORMERS_UTILS_OVERRIDE}/${f}" ]] && \
            install_patch_file "${TRANSFORMERS_UTILS_OVERRIDE}/${f}" \
                "${VLLM_ROOT}/transformers_utils/${f}"
    done

    # configs/ subdirectory (modified + new model configs)
    if [[ -d "${TRANSFORMERS_UTILS_OVERRIDE}/configs" ]]; then
        for f in "${TRANSFORMERS_UTILS_OVERRIDE}/configs/"*.py; do
            [[ -f "$f" ]] && \
                install_patch_file "$f" \
                    "${VLLM_ROOT}/transformers_utils/configs/$(basename "$f")"
        done
    fi

    # tokenizer_group/ subdirectory
    if [[ -d "${TRANSFORMERS_UTILS_OVERRIDE}/tokenizer_group" ]]; then
        for f in __init__.py base_tokenizer_group.py ray_tokenizer_group.py \
                 tokenizer_group.py; do
            [[ -f "${TRANSFORMERS_UTILS_OVERRIDE}/tokenizer_group/${f}" ]] && \
                install_patch_file "${TRANSFORMERS_UTILS_OVERRIDE}/tokenizer_group/${f}" \
                    "${VLLM_ROOT}/transformers_utils/tokenizer_group/${f}"
        done
    fi

    # tokenizers/ subdirectory
    if [[ -d "${TRANSFORMERS_UTILS_OVERRIDE}/tokenizers" ]]; then
        for f in __init__.py mistral.py; do
            [[ -f "${TRANSFORMERS_UTILS_OVERRIDE}/tokenizers/${f}" ]] && \
                install_patch_file "${TRANSFORMERS_UTILS_OVERRIDE}/tokenizers/${f}" \
                    "${VLLM_ROOT}/transformers_utils/tokenizers/${f}"
        done
    fi

    # processors/ subdirectory (new)
    if [[ -d "${TRANSFORMERS_UTILS_OVERRIDE}/processors" ]]; then
        mkdir -p "${VLLM_ROOT}/transformers_utils/processors"
        cp -r "${TRANSFORMERS_UTILS_OVERRIDE}/processors/." \
              "${VLLM_ROOT}/transformers_utils/processors/"
    fi
fi

# PRD #69: Clear ALL __pycache__ under VLLM_ROOT after every cp/patch is done.
# py_compile below only compiles ./qwen3_6_scripts, not VLLM_ROOT, so this
# ensures the docker snapshot has no stale .pyc for any patched vllm module.
find "${VLLM_ROOT}" -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

build_stage "compiling submission Python sources"
find . -path './wheels' -prune -o -name '*.py' -print0 | xargs -0 python3 -m py_compile
build_stage "patch script completed"