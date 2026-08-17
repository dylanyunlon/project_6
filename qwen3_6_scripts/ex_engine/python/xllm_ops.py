"""
xllm_ops.py — NO-FALLBACK xllm kernel loader for vllm hot path

Architecture (matching xllm/core/kernels/ilu/ dispatch):
    xllm C++:   kernels/ilu/*.cpp  →  ixformer::infer::*  (dlopen ixformer .so)
    Our Python:  xllm_ops.py       →  xllm_*.so           (dlopen our compiled .so)
                                   →  ix_full_bridge.so   (dlopen ixformer bridge)

Source mapping (upstream → us):
    xllm/core/kernels/ilu/norm.cpp        → xllm_norm.so
    xllm/core/kernels/ilu/rope.cpp        → xllm_rope.so
    xllm/core/kernels/ilu/activation.cpp  → xllm_activation.so
    xllm/core/kernels/ilu/attention.cpp   → ix_full_bridge.so (paged_attention, flash_attn)
    xllm/core/kernels/ilu/fused_moe.cpp   → xllm_moe.so + ix_full_bridge.so
    xllm/core/kernels/ilu/matmul.cpp      → ix_full_bridge.so (ixformer_linear)
    xllm/core/layers/ilu/fused_moe.cpp    → corex_moe.py (Python orchestrator)
    xllm/core/layers/ilu/attention.cpp    → corex_fa2.py (Python orchestrator)

NO FALLBACK: If a .so fails to load, we raise immediately.
The comp 168 log shows that fallback = pure PyTorch = 683 score.
We need 8000. Every kernel MUST go through hardware-accelerated path.
"""

import os
import sys
import importlib.util
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger("ex_engine.xllm_ops")

# =========================================================================
# .so search paths
# =========================================================================
_SEARCH_DIRS = []

def _init_search_dirs():
    """Build list of directories to search for .so files."""
    global _SEARCH_DIRS
    if _SEARCH_DIRS:
        return

    here = os.path.dirname(os.path.abspath(__file__))

    # 1. vllm package dir (deployed by patch_ops.sh)
    try:
        import vllm
        _SEARCH_DIRS.append(os.path.dirname(vllm.__file__))
    except ImportError:
        pass

    # 2. prebuilt dir
    _SEARCH_DIRS.append(os.path.join(here, "..", "..", "qwen3_6_scripts",
                                      "prebuilt", "corex-3.2.3-ivcore10"))

    # 3. build output dir
    _SEARCH_DIRS.append(os.path.join(here, "..", "build"))

    # 4. /workspace paths (inside docker)
    _SEARCH_DIRS.append("/workspace/qwen3_6_scripts/prebuilt/corex-3.2.3-ivcore10")
    _SEARCH_DIRS.append("/workspace/ex_engine/build")

    # Normalize
    _SEARCH_DIRS = [os.path.normpath(d) for d in _SEARCH_DIRS if os.path.isdir(d)]


def _load_so(name: str) -> Any:
    """Load a .so by name. Raises RuntimeError if not found."""
    _init_search_dirs()

    for d in _SEARCH_DIRS:
        path = os.path.join(d, f"{name}.so")
        if not os.path.isfile(path):
            continue
        try:
            spec = importlib.util.spec_from_file_location(name, path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fns = [x for x in dir(mod) if not x.startswith("_")]
            logger.info("xllm_ops: loaded %s from %s (%d functions: %s)",
                       name, path, len(fns), ", ".join(fns[:8]))
            return mod
        except Exception as e:
            logger.warning("xllm_ops: %s at %s failed: %s", name, path, e)
            continue

    raise RuntimeError(
        f"xllm_ops: CANNOT load {name}.so — searched {_SEARCH_DIRS}. "
        f"Build with: bash ex_engine/build_xllm_kernels.sh"
    )


# =========================================================================
# Module registry — lazy-loaded, no fallback
# =========================================================================
_modules: Dict[str, Any] = {}

def _get(name: str) -> Any:
    if name not in _modules:
        _modules[name] = _load_so(name)
    return _modules[name]


# =========================================================================
# Public API — matches xllm/core/kernels/ilu/ function signatures
# =========================================================================

# --- Norm (xllm/core/kernels/ilu/norm.cpp) ---
def rms_norm(input, weight, epsilon):
    """RMSNorm. Maps to ixformer::infer::rms_norm."""
    return _get("xllm_norm").rms_norm(input, weight, epsilon)

def residual_rms_norm(input, residual, weight, epsilon):
    """Fused residual + RMSNorm. Maps to ixformer::infer::residual_rms_norm."""
    return _get("xllm_norm").residual_rms_norm(input, residual, weight, epsilon)

# --- RoPE (xllm/core/kernels/ilu/rope.cpp) ---
def rotary_embedding(positions, query, key, cos_sin_cache, is_neox=True):
    """Fused rotary embedding. Maps to ixformer::infer::xllm_rotary_embedding."""
    return _get("xllm_rope").rotary_embedding(positions, query, key,
                                               cos_sin_cache, is_neox)

# --- Activation (xllm/core/kernels/ilu/activation.cpp) ---
def silu_and_mul(input, output=None):
    """Fused SiLU activation. Maps to ixformer::infer::silu_and_mul."""
    return _get("xllm_activation").silu_and_mul(input, output)

def gelu_and_mul(input, output=None):
    """Fused GeLU activation."""
    return _get("xllm_activation").gelu_and_mul(input, output)

# --- Cache (xllm/core/kernels/ilu/attention.cpp reshape part) ---
def reshape_and_cache(key, value, key_cache, value_cache, slot_mapping):
    """Write KV to paged cache. Maps to ixformer::infer::xllm_reshape_and_cache."""
    return _get("xllm_cache").reshape_and_cache(key, value, key_cache,
                                                 value_cache, slot_mapping)

# --- Attention (xllm/core/kernels/ilu/attention.cpp) ---
def paged_attention(out, query, key_cache, value_cache,
                    num_kv_heads, scale, block_tables, context_lens,
                    block_size, max_context_len, alibi_slopes=None):
    """Paged attention decode. Maps to ixformer::infer::xllm_paged_attention."""
    bridge = _get("ix_full_bridge")
    return bridge.ix_paged_attention(
        out, query, key_cache, value_cache,
        num_kv_heads, scale, block_tables, context_lens,
        block_size, max_context_len, alibi_slopes
    )

def flash_attn_prefill(query, key_cache, value_cache, out,
                       block_tables, cu_seq_q, cu_seq_k,
                       max_seq_q, max_seq_k, scale,
                       is_causal=True):
    """Flash attention prefill. Maps to ixformer::infer::ixinfer_flash_attn_unpad."""
    bridge = _get("ix_full_bridge")
    return bridge.ix_flash_attn_prefill(
        query, key_cache, value_cache, out,
        block_tables, cu_seq_q, cu_seq_k,
        max_seq_q, max_seq_k, is_causal, scale
    )

# --- MoE (xllm/core/kernels/ilu/fused_moe.cpp) ---
def topk_softmax(topk_weights, topk_ids, token_expert_ids, gating_output, topk):
    """MoE topk + softmax. Maps to ixformer::infer::topk_softmax."""
    return _get("xllm_moe").topk_softmax(
        topk_weights, topk_ids, token_expert_ids, gating_output, topk
    )

def moe_compute_token_index(sorted_token_ids, expert_ids, num_tokens_post_padded,
                            token_expert_ids, num_experts, block_size):
    """MoE token routing. Maps to ixformer::infer::moe_compute_token_index_api."""
    return _get("xllm_moe").moe_compute_token_index(
        sorted_token_ids, expert_ids, num_tokens_post_padded,
        token_expert_ids, num_experts, block_size
    )

# --- Linear (xllm/core/kernels/ilu/matmul.cpp) ---
def ixformer_linear(input, weight, act_type=0, bias=None, out=None):
    """GEMM via ixformer. Maps to ixformer::infer::ixformer_linear."""
    bridge = _get("ix_full_bridge")
    return bridge.ix_linear(input, weight, act_type, bias, out)

# --- Fused QK-Norm + RoPE ---
def fused_qknorm_rope(query, key, cos_sin_cache, positions,
                      qk_norm_weight, epsilon, interleave=False):
    """Fused QK normalization + rotary embedding (saves 128 kernel launches)."""
    return _get("xllm_fused_qknorm_rope").fused_qknorm_rope(
        query, key, cos_sin_cache, positions, qk_norm_weight, epsilon, interleave
    )


# =========================================================================
# Availability check — call at startup to verify ALL .so are loadable
# =========================================================================
def check_all(strict=True):
    """Verify all required .so files are loadable.

    Args:
        strict: If True, raise on any missing .so (NO FALLBACK mode).
                If False, return dict of {name: loaded_bool}.
    """
    required = [
        "ix_full_bridge",     # attention + linear + MoE bridge
        "xllm_norm",          # rms_norm, residual_rms_norm
        "xllm_rope",          # rotary_embedding
        "xllm_activation",    # silu_and_mul
        "xllm_cache",         # reshape_and_cache
        "xllm_moe",           # topk_softmax, moe_compute_token_index
    ]

    optional = [
        "xllm_fused_qknorm_rope",  # nice-to-have: fused QK-norm + RoPE
    ]

    results = {}
    missing = []

    for name in required:
        try:
            _get(name)
            results[name] = True
        except RuntimeError:
            results[name] = False
            missing.append(name)

    for name in optional:
        try:
            _get(name)
            results[name] = True
        except RuntimeError:
            results[name] = False
            logger.info("xllm_ops: optional %s not available", name)

    if strict and missing:
        raise RuntimeError(
            f"xllm_ops: {len(missing)} required .so MISSING: {missing}. "
            f"Score will be ~683 without these. Build with: "
            f"bash ex_engine/build_xllm_kernels.sh"
        )

    loaded = sum(1 for v in results.values() if v)
    total = len(results)
    logger.info("xllm_ops: %d/%d .so loaded", loaded, total)

    return results
