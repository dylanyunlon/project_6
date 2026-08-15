"""
patch_vllm_ops.py — Wire ix_full_bridge C++ kernels into vllm's hot path.

Architecture (CCCL policy_selector pattern):
    Base image provides fused C++ kernels in ixformer::infer namespace.
    ix_full_bridge.so wraps these with pybind11.
    This module monkey-patches vllm's Python operators to call the bridge
    instead of PyTorch fallback code.

Problem statement (683 → 8000 gap):
    vllm's _custom_ops.py fails to load on BI-V100 (no vllm C++ extensions).
    Without patches, EVERY norm/activation/rope/cache/attention call goes
    through pure PyTorch — multiple kernel launches per op instead of 1.

    Sub168 (competitor): all ops fused via xllm C++ engine → 11.9 TPS
    Sub655 (us without patches): Python fallback → 2.6 TPS

Solution:
    Patch vllm's operator dispatch points so they call our bridge .so,
    which links against the SAME ixformer .so files in the base image.

Patched modules and their vllm paths:
    1. vllm.model_executor.layers.layernorm.GemmaRMSNorm
       → ix_ops.rms_norm / ix_ops.fused_add_rms_norm
    2. vllm.model_executor.layers.activation.SiluAndMul
       → ix_ops.silu_and_mul
    3. vllm._custom_ops (ops fallback registry)
       → ix_ops for all registered ops

Source mapping:
    upstream_ref/xllm_latest/core/kernels/ilu/norm.cpp      → rms_norm patch
    upstream_ref/xllm_latest/core/kernels/ilu/activation.cpp → silu_and_mul patch
    upstream_ref/xllm_latest/core/kernels/ilu/rope.cpp       → rotary_embedding patch
    upstream_ref/xllm_latest/core/kernels/ilu/attention.cpp  → cache/attention patch
"""

import os
import sys
import logging
import torch
from typing import Optional, Tuple

logger = logging.getLogger("ex_engine.patch_vllm_ops")

_patched = False


def apply_all_patches() -> int:
    """Apply all available patches. Returns count of patches applied."""
    global _patched
    if _patched:
        return 0
    _patched = True

    from ex_engine.python import ix_ops
    if not ix_ops.is_available():
        logger.warning("ix_ops bridge not available — no patches applied")
        return 0

    n = 0
    n += _patch_layernorm()
    n += _patch_silu_and_mul()
    n += _patch_custom_ops()
    logger.info("patch_vllm_ops: %d patches applied", n)
    return n


# =========================================================================
# Patch 1: GemmaRMSNorm → fused C++ kernel
# =========================================================================
def _patch_layernorm() -> int:
    """Replace GemmaRMSNorm.forward with ix_ops.rms_norm."""
    from ex_engine.python import ix_ops
    if not ix_ops.has_rms_norm():
        logger.debug("ix_ops missing rms_norm, skip layernorm patch")
        return 0

    try:
        from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    except ImportError:
        logger.debug("Cannot import GemmaRMSNorm, skip")
        return 0

    _orig_forward = GemmaRMSNorm.forward

    def _patched_forward(self, x, residual=None):
        if residual is not None:
            # fused_add_rms_norm: norm(x + residual) → (normed, new_residual)
            if ix_ops.has_fused_add_rms_norm():
                out = torch.empty_like(x)
                residual_out = torch.empty_like(x)
                ix_ops.fused_add_rms_norm(
                    x, residual, self.weight, out, residual_out,
                    self.variance_epsilon)
                return out, residual_out
            else:
                # Two-step fallback using just rms_norm
                new_residual = x + residual
                out = torch.empty_like(x)
                ix_ops.rms_norm(out, new_residual, self.weight,
                                self.variance_epsilon)
                return out, new_residual
        else:
            out = torch.empty_like(x)
            ix_ops.rms_norm(out, x, self.weight, self.variance_epsilon)
            return out

    GemmaRMSNorm.forward = _patched_forward
    logger.info("PATCHED: GemmaRMSNorm.forward → ix_ops.rms_norm")
    return 1


# =========================================================================
# Patch 2: SiluAndMul → fused C++ kernel
# =========================================================================
def _patch_silu_and_mul() -> int:
    """Replace SiluAndMul.forward with ix_ops.silu_and_mul."""
    from ex_engine.python import ix_ops
    if not ix_ops.has_silu_and_mul():
        logger.debug("ix_ops missing silu_and_mul, skip activation patch")
        return 0

    try:
        from vllm.model_executor.layers.activation import SiluAndMul
    except ImportError:
        logger.debug("Cannot import SiluAndMul, skip")
        return 0

    def _patched_forward(self, x):
        return ix_ops.silu_and_mul(x)

    SiluAndMul.forward = _patched_forward
    logger.info("PATCHED: SiluAndMul.forward → ix_ops.silu_and_mul")
    return 1


# =========================================================================
# Patch 3: _custom_ops fallback registry
# =========================================================================
def _patch_custom_ops() -> int:
    """Patch vllm's _custom_ops to use ix_ops for registered ops."""
    from ex_engine.python import ix_ops
    count = 0

    try:
        import vllm._custom_ops as ops
    except ImportError:
        logger.debug("Cannot import vllm._custom_ops, skip")
        return 0

    # Patch silu_and_mul
    if ix_ops.has_silu_and_mul() and hasattr(ops, 'silu_and_mul'):
        def _silu_and_mul(out, x):
            result = ix_ops.silu_and_mul(x)
            out.copy_(result)
        ops.silu_and_mul = _silu_and_mul
        count += 1
        logger.info("PATCHED: _custom_ops.silu_and_mul → ix_ops")

    # Patch rms_norm
    if ix_ops.has_rms_norm() and hasattr(ops, 'rms_norm'):
        def _rms_norm(out, input, weight, eps):
            ix_ops.rms_norm(out, input, weight, eps)
        ops.rms_norm = _rms_norm
        count += 1
        logger.info("PATCHED: _custom_ops.rms_norm → ix_ops")

    # Patch fused_add_rms_norm
    if ix_ops.has_fused_add_rms_norm() and hasattr(ops, 'fused_add_rms_norm'):
        def _fused_add_rms_norm(input, residual, weight, eps):
            out = torch.empty_like(input)
            residual_out = torch.empty_like(input)
            ix_ops.fused_add_rms_norm(input, residual, weight,
                                       out, residual_out, eps)
            input.copy_(out)
            residual.copy_(residual_out)
        ops.fused_add_rms_norm = _fused_add_rms_norm
        count += 1
        logger.info("PATCHED: _custom_ops.fused_add_rms_norm → ix_ops")

    # Patch rotary_embedding
    if ix_ops.has_rotary_embedding() and hasattr(ops, 'rotary_embedding'):
        def _rotary_embedding(positions, query, key, head_size,
                              cos_sin_cache, is_neox):
            ix_ops.rotary_embedding(positions, query, key, head_size,
                                     cos_sin_cache, is_neox)
        ops.rotary_embedding = _rotary_embedding
        count += 1
        logger.info("PATCHED: _custom_ops.rotary_embedding → ix_ops")

    return count


# =========================================================================
# Auto-apply on import if requested
# =========================================================================
if os.environ.get("IX_OPS_AUTO_PATCH", "0") == "1":
    try:
        apply_all_patches()
    except Exception as e:
        logger.warning("ix_ops auto-patch failed: %s", e)
