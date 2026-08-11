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

# NOTE: intentionally NO set -e — individual patch failures must NOT abort
# the entire build.  Each step logs its own errors, and non-critical patches
# (xformers, diagnostics) may legitimately fail if the base image differs.
set -uo pipefail

build_stage() { printf '[BI100 BUILD] %s\n' "$1" >&2; }
require_file() {
    local path=$1
    [[ -f "$path" ]] || {
        printf '[WARN] patch source missing (non-fatal): %s\n' "$path" >&2
        return 1
    }
}
install_patch_file() {
    local source=$1
    local target=$2

    require_file "$source" || return 0
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
  if ls "${WHEEL_DIR}/transformers-${TRANSFORMERS_REQUIRED_VERSION}"*.whl >/dev/null 2>&1; then
    python3 -m pip install --no-index --no-deps --find-links="${WHEEL_DIR}" \
      "transformers==${TRANSFORMERS_REQUIRED_VERSION}"
  else
    echo "[WARN] offline wheel not found, trying pip install" >&2
    pip install "transformers==${TRANSFORMERS_REQUIRED_VERSION}" --timeout 30 2>&1 || \
      echo "[WARN] transformers install failed (non-fatal, base image may work)" >&2
  fi
fi

python3 - "$TRANSFORMERS_REQUIRED_VERSION" <<'PY' || echo "[WARN] transformers version check failed (non-fatal)"
import importlib.metadata
import sys

required = sys.argv[1]
try:
    installed = importlib.metadata.version("transformers")
    if installed != required:
        print(f"[WARN] transformers: expected {required}, got {installed}")
    else:
        print(f"[ok] transformers {installed}")
except Exception as e:
    print(f"[WARN] transformers check error: {e}")
PY

build_stage "discovering Python package roots"
python3 - <<'PY' > /tmp/qwen36_patch_paths.env || true
from patch_utils import package_root, shell_env_line

print(shell_env_line("VLLM_ROOT", package_root("vllm")))
print(shell_env_line("TRANSFORMERS_ROOT", package_root("transformers")))
PY
source /tmp/qwen36_patch_paths.env 2>/dev/null || true

# Fallback: if patch_utils failed, find vllm manually
if [[ -z "${VLLM_ROOT:-}" ]]; then
    for _candidate in \
        /usr/local/corex/lib/python3/dist-packages/vllm \
        /usr/local/corex/lib64/python3/dist-packages/vllm \
        /usr/local/lib/python3.10/site-packages/vllm; do
        if [[ -d "$_candidate" ]]; then
            VLLM_ROOT="$_candidate"
            break
        fi
    done
fi
if [[ -z "${TRANSFORMERS_ROOT:-}" ]]; then
    for _candidate in \
        /usr/local/corex/lib/python3/dist-packages/transformers \
        /usr/local/corex/lib64/python3/dist-packages/transformers \
        /usr/local/lib/python3.10/site-packages/transformers; do
        if [[ -d "$_candidate" ]]; then
            TRANSFORMERS_ROOT="$_candidate"
            break
        fi
    done
fi

echo "VLLM_ROOT=${VLLM_ROOT}"
echo "TRANSFORMERS_ROOT=${TRANSFORMERS_ROOT}"
[[ -d "${VLLM_ROOT:-}" ]] || {
    printf '[FATAL] vLLM root does not exist: %s\n' "${VLLM_ROOT:-UNSET}" >&2
    printf '[FATAL] Tried patch_utils + manual scan, neither found vllm\n' >&2
    exit 2
}

VLLM_OVERRIDE_ROOT="./vendor_overrides/vllm"
_HAS_OVERRIDES=true
[[ -d "$VLLM_OVERRIDE_ROOT" ]] || {
    printf '[WARN] vLLM override directory missing: %s — skipping override installs\n' "$VLLM_OVERRIDE_ROOT" >&2
    _HAS_OVERRIDES=false
}

# --- Mirror path: base image may have TWO vllm installs ---
# VLLM_ROOT (from importlib) is typically /usr/local/lib/python3.10/site-packages/vllm
# but PYTHONPATH puts /usr/local/corex/lib/python3/dist-packages/vllm first at runtime.
# We must deploy to BOTH or the runtime loads the unpatched copy.
VLLM2=""
for _candidate in \
    /usr/local/corex/lib/python3/dist-packages/vllm \
    /usr/local/corex/lib64/python3/dist-packages/vllm \
    /usr/local/lib/python3.10/site-packages/vllm; do
    if [[ -d "$_candidate" && "$_candidate" != "$VLLM_ROOT" ]]; then
        VLLM2="$_candidate"
        break
    fi
done
if [[ -n "$VLLM2" ]]; then
    echo "VLLM2=${VLLM2} (will mirror all patches)"
else
    echo "VLLM2=<none> (single vllm install)"
fi

# Helper: copy to VLLM_ROOT and VLLM2 (if exists)
deploy_both() {
    local src="$1" rel="$2"
    cp "$src" "${VLLM_ROOT}/${rel}"
    [[ -n "$VLLM2" ]] && cp "$src" "${VLLM2}/${rel}" 2>/dev/null || true
}

if $_HAS_OVERRIDES; then
build_stage "installing authoritative vLLM core block overrides"
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
else
build_stage "skipping vLLM core block overrides (vendor_overrides not found)"
fi

build_stage "installing hash-pinned CoreX 3.2.3 extensions"
bash ./install_prebuilt_corex.sh "${VLLM_ROOT}"

build_stage "compiling moe_topk_softmax CUDA kernel"
cd /workspace && bash ex_engine/build_moe_topk.sh 2>&1 || echo "[WARN] moe_topk build failed (non-fatal)"
# Deploy to workspace search path (_custom_ops.py looks in /workspace/ex_engine/build/)
cd "${OLDPWD}"

build_stage "installing BI100 runtime modules"
cp ./bi100_env.py "${VLLM_ROOT}/bi100_env.py"
cp ./bi100_profile.py "${VLLM_ROOT}/bi100_profile.py"
cp ./block_major_kv_cache.py "${VLLM_ROOT}/block_major_kv_cache.py"
cp ./gdn_prefix.py "${VLLM_ROOT}/gdn_prefix.py"

build_stage "installing CoreX paged-KV swap compatibility"
python3 ./patch_corex_swap_blocks.py 2>&1 || echo "[WARN] patch_corex_swap_blocks failed (non-fatal)"
python3 ./patch_block_major_cache_engine.py 2>&1 || echo "[WARN] patch_block_major_cache_engine failed (non-fatal)"
python3 ./patch_worker_cache_transfer_order.py 2>&1 || echo "[WARN] patch_worker_cache_transfer_order failed (non-fatal)"

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
python3 ./patch_model_runner.py 2>&1 || echo "[WARN] patch_model_runner failed (non-fatal)"

build_stage "installing executor startup diagnostics"
python3 ./patch_executor_startup_debug.py 2>&1 || echo "[WARN] patch_executor_startup_debug failed (non-fatal)"
python3 ./patch_worker_startup_profile_guard.py 2>&1 || echo "[WARN] patch_worker_startup_profile_guard failed (non-fatal)"
python3 ./patch_block_major_worker_capacity.py 2>&1 || echo "[WARN] patch_block_major_worker_capacity failed (non-fatal)"

build_stage "installing transformers Qwen3.5 model support"
cp -r ./qwen3_5 "${TRANSFORMERS_ROOT}/models/"
cp -r ./qwen3_5_moe "${TRANSFORMERS_ROOT}/models/"
python3 ./patch_transformers_qwen3_5.py 2>&1 || echo "[WARN] patch_transformers_qwen3_5 failed (non-fatal)"

build_stage "installing vLLM Qwen3.6 model implementation"
# --- vllm model: Qwen3.6-35B-A3B (Qwen3_5 MoE arch) -------------------------
cp ./mamba_cache.py "${VLLM_ROOT}/model_executor/models/"
cp ./qwen3_5.py "${VLLM_ROOT}/model_executor/models/qwen3_5.py"
python3 ./patch_vllm_qwen3_5.py 2>&1 || echo "[WARN] patch_vllm_qwen3_5 failed (non-fatal)"

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
python3 ./patch_block_manager_cache_trace.py 2>&1 || echo "[WARN] patch_block_manager_cache_trace failed (non-fatal)"

build_stage "installing scheduler and attention patches"
# --- xformers: bypass cudnnFlashAttnForward (head_dim=256 > 128 limit) ------
# Injects _run_sdpa_fallback (pure matmul+softmax) into xformers.py.
# Required because head_dim=256 > 128 and ixformer flash attention either
# crashes (is_causal=True) or produces wrong output (attn_mask path).
# The fallback uses query_start_loc to derive actual query lengths, so it
# works correctly during profiling runs with chunked-prefill-style batches.
# also bypasses auto chunked prefill on
python3 ./patch_xformers_sdpa_seq.py 2>&1 || echo "[WARN] patch_xformers_sdpa_seq failed (non-fatal)"
python3 ./patch_xformers_profile.py 2>&1 || echo "[WARN] patch_xformers_profile failed (non-fatal)"

build_stage "installing API parsers and serving modules"
# --- tool parser: Qwen3 XML tool call format ---------------------------------
# Registers "qwen3_coder" parser for Qwen3.6 XML-style tool calls:
#   <tool_call><function=name><parameter=key>\nvalue\n</parameter></function></tool_call>
# Use at server start: --tool-call-parser qwen3_coder --enable-auto-tool-choice
cp ./qwen3coder_tool_parser.py "${VLLM_ROOT}/entrypoints/openai/tool_parsers/"
python3 ./patch_vllm_tool_parser.py 2>&1 || echo "[WARN] patch_vllm_tool_parser failed (non-fatal)"

# --- reasoning parser: Qwen3 <think>...</think> split ------------------------
# Adds --reasoning-parser qwen3 support.
# Routes thinking tokens to reasoning_content, rest to content in the delta.
# Works together with --tool-call-parser qwen3_coder (think → tool call flow).
cp -r ./reasoning "${VLLM_ROOT}/"
cp ./protocol.py "${VLLM_ROOT}/entrypoints/openai/protocol.py"
cp ./cli_args.py "${VLLM_ROOT}/entrypoints/openai/cli_args.py"
cp ./serving_chat.py "${VLLM_ROOT}/entrypoints/openai/serving_chat.py"
cp ./serving_tokenization.py \
    "${VLLM_ROOT}/entrypoints/openai/serving_tokenization.py"
cp ./api_server.py "${VLLM_ROOT}/entrypoints/openai/api_server.py"
cp ./chat_utils.py "${VLLM_ROOT}/entrypoints/chat_utils.py"
python3 - ./api_server.py \
        "${VLLM_ROOT}/entrypoints/openai/api_server.py" <<'PY' || echo "[WARN] api_server identity check failed"
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_bytes()
installed = Path(sys.argv[2]).read_bytes()
if source != installed:
    print("[WARN] runtime api_server overlay identity mismatch")
PY

# --- Mirror ALL patched files to VLLM2 (if a second vllm install exists) ---
if [[ -n "$VLLM2" ]]; then
    build_stage "mirroring patches to VLLM2=${VLLM2}"
    # Critical: paged_attn.py (context_attention_fwd NameError without this)
    cp "${VLLM_ROOT}/attention/ops/paged_attn.py" \
       "${VLLM2}/attention/ops/paged_attn.py" 2>/dev/null || true
    # Model
    cp "${VLLM_ROOT}/model_executor/models/qwen3_5.py" \
       "${VLLM2}/model_executor/models/qwen3_5.py" 2>/dev/null || true
    cp "${VLLM_ROOT}/model_executor/models/mamba_cache.py" \
       "${VLLM2}/model_executor/models/mamba_cache.py" 2>/dev/null || true
    # Runtime modules
    for f in bi100_env.py bi100_profile.py block_major_kv_cache.py \
             gdn_prefix.py sequence.py; do
        cp "${VLLM_ROOT}/${f}" "${VLLM2}/${f}" 2>/dev/null || true
    done
    # Core
    cp "${VLLM_ROOT}/core/scheduler.py" \
       "${VLLM2}/core/scheduler.py" 2>/dev/null || true
    # Serving
    for f in protocol.py cli_args.py serving_chat.py serving_tokenization.py \
             api_server.py; do
        cp "${VLLM_ROOT}/entrypoints/openai/${f}" \
           "${VLLM2}/entrypoints/openai/${f}" 2>/dev/null || true
    done
    cp "${VLLM_ROOT}/entrypoints/chat_utils.py" \
       "${VLLM2}/entrypoints/chat_utils.py" 2>/dev/null || true
    # Tool parsers
    cp "${VLLM_ROOT}/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py" \
       "${VLLM2}/entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py" 2>/dev/null || true
    # Reasoning
    cp -r "${VLLM_ROOT}/reasoning" "${VLLM2}/" 2>/dev/null || true
    # Prebuilt CoreX .so extensions
    for so in "${VLLM_ROOT}"/corex_*.so; do
        [[ -f "$so" ]] && cp "$so" "${VLLM2}/" 2>/dev/null || true
    done
    # Block overrides
    for f in core/evictor_v2.py core/block_manager_v2.py \
             core/block/cpu_kv_content_cache.py core/block/cpu_gpu_block_allocator.py \
             core/block/prefix_caching_block.py core/block/block_table.py \
             model_executor/sampling_metadata.py model_executor/layers/sampler.py \
             sampling_params.py; do
        if [[ -f "${VLLM_ROOT}/${f}" ]]; then
            mkdir -p "$(dirname "${VLLM2}/${f}")"
            cp "${VLLM_ROOT}/${f}" "${VLLM2}/${f}" 2>/dev/null || true
        fi
    done
    echo "[ok] mirrored all patches to VLLM2"
fi

build_stage "deploying ex_engine package to Python path"
_SITE=""
for _s in /usr/local/corex/lib64/python3/dist-packages \
          /usr/local/corex/lib/python3/dist-packages \
          /usr/local/lib/python3.10/site-packages; do
    [[ -d "$_s" ]] && _SITE="$_s" && break
done
if [[ -n "$_SITE" ]]; then
    _EX_DST="$_SITE/ex_engine"
    mkdir -p "$_EX_DST/python" "$_EX_DST/build"
    touch "$_EX_DST/__init__.py" "$_EX_DST/python/__init__.py"
    cp /workspace/ex_engine/python/*.py "$_EX_DST/python/" 2>/dev/null || true
    if [[ -d /workspace/ex_engine/build ]]; then
        cp /workspace/ex_engine/build/*.so "$_EX_DST/build/" 2>/dev/null || true
        cp /workspace/ex_engine/build/*.so "$_EX_DST/" 2>/dev/null || true
    fi
    echo "[ok] ex_engine deployed to $_EX_DST ($(ls "$_EX_DST/build/"*.so 2>/dev/null | wc -l) .so files)"
fi

build_stage "compiling submission Python sources"
find . -path './wheels' -prune -o -name '*.py' -print0 | xargs -0 python3 -m py_compile 2>&1 || echo "[WARN] some .py files failed to compile (non-fatal)"
build_stage "building ix_unified_bridge (optional)"
if [[ -x /workspace/ex_engine/build_unified_bridge.sh ]]; then
    bash /workspace/ex_engine/build_unified_bridge.sh 2>&1 || echo "[WARN] bridge build failed (non-fatal)"
fi

build_stage "deploying ex_engine Python modules"
VLLM_DEPLOY=$(python3 -c "import vllm; print(vllm.__path__[0])" 2>/dev/null | tail -1 || echo "")
if [[ -n "$VLLM_DEPLOY" && -d "$VLLM_DEPLOY" ]]; then
    for f in ix_unified.py corex_so_loader.py moe_fused_dispatch.py ex_topk_bridge.py; do
        cp "/workspace/ex_engine/python/$f" "${VLLM_DEPLOY}/$f" 2>/dev/null || true
    done
    ls /workspace/ex_engine/build/ix_unified_bridge*.so 1>/dev/null 2>&1 && \
        cp /workspace/ex_engine/build/ix_unified_bridge*.so "${VLLM_DEPLOY}/" 2>/dev/null || true
    echo "[ok] ex_engine modules deployed to ${VLLM_DEPLOY}"
fi

build_stage "patch script completed"
