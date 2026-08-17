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

build_stage "installing hash-pinned CoreX 3.2.3 extensions (16 prebuilt .so)"
bash ./install_prebuilt_corex.sh "${VLLM_ROOT}"

build_stage "installing BI100 runtime modules"
cp ./bi100_env.py "${VLLM_ROOT}/bi100_env.py"
cp ./bi100_profile.py "${VLLM_ROOT}/bi100_profile.py"
cp ./block_major_kv_cache.py "${VLLM_ROOT}/block_major_kv_cache.py"
cp ./gdn_prefix.py "${VLLM_ROOT}/gdn_prefix.py"

build_stage "installing CoreX paged-KV swap compatibility"
python3 ./patch_corex_swap_blocks.py
python3 ./patch_block_major_cache_engine.py
python3 ./patch_worker_cache_transfer_order.py

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
python3 ./patch_model_runner.py

build_stage "installing executor startup diagnostics"
python3 ./patch_executor_startup_debug.py
python3 ./patch_worker_startup_profile_guard.py
python3 ./patch_block_major_worker_capacity.py

build_stage "installing transformers Qwen3.5 model support"
cp -r ./qwen3_5 "${TRANSFORMERS_ROOT}/models/"
cp -r ./qwen3_5_moe "${TRANSFORMERS_ROOT}/models/"
python3 ./patch_transformers_qwen3_5.py

build_stage "installing vLLM Qwen3.6 model implementation"
# --- vllm model: Qwen3.6-35B-A3B (Qwen3_5 MoE arch) -------------------------
cp ./mamba_cache.py "${VLLM_ROOT}/model_executor/models/"
cp ./qwen3_5.py "${VLLM_ROOT}/model_executor/models/qwen3_5.py"
cp ./ix_fused_moe.py "${VLLM_ROOT}/model_executor/models/ix_fused_moe.py" || true
python3 ./patch_vllm_qwen3_5.py

# --- Deploy prebuilt .so into vllm package for import -----------------------
PREBUILT_DIR="./prebuilt/corex-3.2.3-ivcore10"
if [ -d "$PREBUILT_DIR" ]; then
    for so_file in "$PREBUILT_DIR"/*.so; do
        base=$(basename "$so_file" .so)
        # Deploy corex_*.so as vllm submodules (import from vllm import corex_xxx)
        cp "$so_file" "${VLLM_ROOT}/${base}.so" 2>/dev/null || true
        echo "[patch_ops] deployed ${base}.so → ${VLLM_ROOT}/"
    done
fi

# --- Deploy ix_bridge Python integration layer --------------------------------
build_stage "deploying ix_bridge operator replacements"
EX_ENGINE_DIR="$(cd "$(dirname "$0")/ex_engine" 2>/dev/null && pwd || echo "")"
if [ -z "$EX_ENGINE_DIR" ] || [ ! -d "$EX_ENGINE_DIR/python" ]; then
    EX_ENGINE_DIR="$(cd "$(dirname "$0")/../ex_engine" 2>/dev/null && pwd || echo "")"
fi
if [ -z "$EX_ENGINE_DIR" ] || [ ! -d "$EX_ENGINE_DIR/python" ]; then
    EX_ENGINE_DIR="/workspace/ex_engine"
fi

if [ -d "$EX_ENGINE_DIR/python" ]; then
    # Create ex_engine package inside vllm with correct Python package structure
    mkdir -p "${VLLM_ROOT}/ex_engine/python"
    mkdir -p "${VLLM_ROOT}/ex_engine/csrc"

    # __init__.py with re-exports so both import styles work:
    #   from ex_engine.python import ix_ops_dispatch  (direct)
    #   from vllm.ex_engine import ix_ops_dispatch    (via re-export)
    cat > "${VLLM_ROOT}/ex_engine/__init__.py" << 'INIT_EOF'
"""ex_engine — Algorithm factor replacement for BI-V100."""
# Re-export python subpackage members at top level for backward compat
# Allows: from vllm.ex_engine import ix_ops_dispatch
try:
    from ex_engine.python.ix_ops_dispatch import *
    from ex_engine.python import ix_ops_dispatch
    from ex_engine.python import ix_ops
    from ex_engine.python import patch_vllm_ops
except ImportError:
    pass
INIT_EOF
    echo '"""ex_engine.python — dispatch and bridge modules."""' > "${VLLM_ROOT}/ex_engine/python/__init__.py"

    # Deploy ALL Python modules
    cp "$EX_ENGINE_DIR/python/"*.py "${VLLM_ROOT}/ex_engine/python/"
    echo "[patch_ops] deployed $(ls -1 "${VLLM_ROOT}/ex_engine/python/"*.py | wc -l) modules → ${VLLM_ROOT}/ex_engine/python/"

    # Deploy bridge C++ source for JIT fallback
    for cpp in "$EX_ENGINE_DIR"/csrc/ix_full_bridge*.cpp "$EX_ENGINE_DIR"/csrc/ix_moe_bridge.cpp; do
        [ -f "$cpp" ] && cp "$cpp" "${VLLM_ROOT}/ex_engine/csrc/" && \
            echo "[patch_ops] deployed $(basename $cpp) for JIT fallback"
    done

    # Create startup hook that patches vllm ops at import time
    cat > "${VLLM_ROOT}/ix_startup_patch.py" << 'STARTUP_EOF'
"""Apply ix_ops patches at vllm startup."""
import logging
_logger = logging.getLogger("ix_startup_patch")
def apply():
    import sys, os
    # Ensure ex_engine is importable from both locations
    for p in ["/workspace/qwen3_6_scripts/ex_engine/..",
              "/workspace/qwen3_6_scripts",
              "/workspace/ex_engine/..",
              "/workspace"]:
        rp = os.path.realpath(p)
        if os.path.isdir(rp) and rp not in sys.path:
            sys.path.insert(0, rp)
    n = 0
    # 1. ix_full_bridge patches (rms_norm, silu_and_mul, linear)
    try:
        from ex_engine.python.patch_vllm_ops import apply_all_patches
        k = apply_all_patches()
        n += k
        if k > 0:
            _logger.info("ix_startup_patch: %d bridge patches applied", k)
    except Exception as e:
        _logger.warning("ix_startup_patch: bridge patches failed: %s", e)
    # 2. xllm kernel patches (topk_softmax, norm, activation, rope, cache)
    try:
        from ex_engine.python.patch_vllm_hot_path import apply as apply_hot
        k = apply_hot(strict=False)
        n += k
        if k > 0:
            _logger.info("ix_startup_patch: %d hot-path patches applied", k)
    except Exception as e:
        _logger.warning("ix_startup_patch: hot-path patches failed: %s", e)
    return n
_n_patches = apply()
STARTUP_EOF
    echo "[patch_ops] deployed ix_startup_patch.py"

    # Hook into vllm __init__.py to auto-apply patches on import
    VLLM_INIT="${VLLM_ROOT}/__init__.py"
    if [ -f "$VLLM_INIT" ]; then
        if ! grep -q "ix_startup_patch" "$VLLM_INIT" 2>/dev/null; then
            echo "" >> "$VLLM_INIT"
            echo "# Auto-apply ix_bridge operator patches" >> "$VLLM_INIT"
            echo "try:" >> "$VLLM_INIT"
            echo "    from vllm import ix_startup_patch" >> "$VLLM_INIT"
            echo "except Exception:" >> "$VLLM_INIT"
            echo "    pass" >> "$VLLM_INIT"
            echo "[patch_ops] hooked ix_startup_patch into vllm/__init__.py"
        fi
    fi
else
    echo "[patch_ops] WARN: ex_engine/python not found, skip ix_bridge deployment"
fi

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
python3 ./patch_block_manager_cache_trace.py

build_stage "installing scheduler and attention patches"
# --- xformers: bypass cudnnFlashAttnForward (head_dim=256 > 128 limit) ------
# Injects _run_sdpa_fallback (pure matmul+softmax) into xformers.py.
# Required because head_dim=256 > 128 and ixformer flash attention either
# crashes (is_causal=True) or produces wrong output (attn_mask path).
# The fallback uses query_start_loc to derive actual query lengths, so it
# works correctly during profiling runs with chunked-prefill-style batches.
# also bypasses auto chunked prefill on
python3 ./patch_xformers_sdpa_seq.py
python3 ./patch_xformers_profile.py

build_stage "installing API parsers and serving modules"
# --- tool parser: Qwen3 XML tool call format ---------------------------------
# Registers "qwen3_coder" parser for Qwen3.6 XML-style tool calls:
#   <tool_call><function=name><parameter=key>\nvalue\n</parameter></function></tool_call>
# Use at server start: --tool-call-parser qwen3_coder --enable-auto-tool-choice
cp ./qwen3coder_tool_parser.py "${VLLM_ROOT}/entrypoints/openai/tool_parsers/"
python3 ./patch_vllm_tool_parser.py

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
        "${VLLM_ROOT}/entrypoints/openai/api_server.py" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_bytes()
installed = Path(sys.argv[2]).read_bytes()
if source != installed:
    raise SystemExit("runtime api_server overlay identity mismatch")
PY

# --- protocol.py identity check: ensure max_completion_tokens is accepted ---
python3 - ./protocol.py \
        "${VLLM_ROOT}/entrypoints/openai/protocol.py" <<'PY'
from pathlib import Path
import sys

source = Path(sys.argv[1]).read_bytes()
installed = Path(sys.argv[2]).read_bytes()
if source != installed:
    raise SystemExit("runtime protocol overlay identity mismatch")
# Verify max_completion_tokens field is declared (not just extra=allow)
if b"max_completion_tokens" not in installed:
    raise SystemExit("protocol.py missing max_completion_tokens field")
PY

build_stage "building CUTLASS grouped GEMM (gemm_grouped.so)"
if [[ -f "${EX_ENGINE_DIR}/build_gemm_grouped.sh" ]]; then
    bash "${EX_ENGINE_DIR}/build_gemm_grouped.sh" 2>&1 || {
        echo "[WARN] gemm_grouped build failed — will use torch.mm fallback"
    }
    # Deploy compiled .so if it exists
    for so in "${EX_ENGINE_DIR}"/gemm_grouped.so "${EX_ENGINE_DIR}"/csrc/gemm_grouped.so; do
        if [[ -f "$so" ]]; then
            cp "$so" "${VLLM_ROOT}/gemm_grouped.so"
            echo "[patch_ops] deployed gemm_grouped.so → ${VLLM_ROOT}/"
            break
        fi
    done
fi

build_stage "building CUTLASS batched GEMM (corex_batched_gemm.so)"
if [[ -f "${EX_ENGINE_DIR}/xllm_kernels/cuda/corex_batched_gemm_kernel.cu" ]]; then
    python3 << PYEOF
import os, sys, shutil
try:
    from torch.utils.cpp_extension import load
    ex = "${EX_ENGINE_DIR}"
    cutlass_inc = ""
    for d in ["/usr/local/corex-samples-3.2.3_x86_64/samples/cutlass/include",
              "/usr/local/corex/include/cutlass", "/usr/include/cutlass"]:
        if os.path.isdir(d):
            cutlass_inc = d
            break
    if not cutlass_inc:
        print("[batched_gemm] No cutlass headers — skip"); sys.exit(0)
    mod = load(
        name="corex_batched_gemm",
        sources=[
            os.path.join(ex, "xllm_kernels/cuda/corex_batched_gemm_kernel.cu"),
            os.path.join(ex, "xllm_kernels/cuda/bindings/corex_batched_gemm_bind.cpp"),
        ],
        extra_include_paths=[cutlass_inc],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=["-O2", f"-I{cutlass_inc}"],
        extra_ldflags=["/usr/local/corex/lib64/libcuinfer.so", "-Wl,-rpath,/usr/local/corex/lib64"],
        verbose=False,
    )
    print("[batched_gemm] ✓ Compiled")
    import importlib
    spec = importlib.util.find_spec("corex_batched_gemm")
    if spec and spec.origin:
        shutil.copy2(spec.origin, "${VLLM_ROOT}/corex_batched_gemm.so")
        print("[batched_gemm] ✓ Deployed to ${VLLM_ROOT}/")
except Exception as e:
    print(f"[batched_gemm] WARN: {e}")
PYEOF
fi

build_stage "building MoE bridge (ix_moe_bridge.so)"
if [[ -f "${EX_ENGINE_DIR}/csrc/ix_moe_bridge.cpp" ]]; then
    SCRIPT_DIR="${EX_ENGINE_DIR}" bash "${EX_ENGINE_DIR}/build_moe_bridge.sh" "${VLLM_ROOT}" 2>&1 || {
        echo "[WARN] MoE bridge build failed — will use Python fallback"
    }
    # Deploy .so to all paths ix_fused_moe.py searches
    for src in "${VLLM_ROOT}/ex_engine/ix_moe_bridge.so" \
               "${EX_ENGINE_DIR}/prebuilt/ix_moe_bridge.so"; do
        if [[ -f "$src" ]]; then
            cp "$src" "${VLLM_ROOT}/ix_moe_bridge.so" 2>/dev/null || true
            cp "$src" "${VLLM_ROOT}/model_executor/models/ix_moe_bridge.so" 2>/dev/null || true
            echo "[patch_ops] deployed ix_moe_bridge.so to vllm search paths"
            break
        fi
    done
fi

build_stage "deploying all ex_engine Python modules"
EX_PY_DIR="${VLLM_ROOT}/ex_engine/python"
mkdir -p "${EX_PY_DIR}"
if [[ -d "${EX_ENGINE_DIR}/python" ]]; then
    cp "${EX_ENGINE_DIR}/python/"*.py "${EX_PY_DIR}/" 2>/dev/null
    echo "[patch_ops] deployed $(ls -1 "${EX_PY_DIR}"/*.py 2>/dev/null | wc -l) Python modules → ${EX_PY_DIR}/"
fi

build_stage "compiling submission Python sources"
find . -path './wheels' -prune -o -name '*.py' -print0 | xargs -0 python3 -m py_compile

build_stage "verifying dlopen chain"
python3 ./verify_dlopen_chain.py --vllm-root "${VLLM_ROOT}" || {
    echo "[WARN] dlopen chain verification found issues (non-fatal)"
}

build_stage "patch script completed"