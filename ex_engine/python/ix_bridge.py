"""
ix_bridge.py — Full ixformer MoE pipeline bridge.

Loads ix_moe_bridge.so via JIT and exposes both individual ops and the full
fused MoE forward pass that replaces the Python for-loop in qwen3_5.py.

Pipeline (mirrors xllm/core/layers/ilu/fused_moe.cpp):
  topk_softmax → moe_gen_idx → moe_expand_input → group_gemm(w13)
  → silu_and_mul → group_gemm(w2) → moe_combine_result

All 6 ixformer::infer C++ functions are called through ix_moe_bridge.cpp
which forward-declares them and links against the base image SDK.
"""

import os
import logging
import torch
from typing import Tuple, Optional, List

logger = logging.getLogger("ex_engine.ix_bridge")

_ix_bridge = None
_ix_bridge_loaded = False  # True after attempt, even if failed
_ix_bridge_available = False


def _find_cpp_source():
    """Find ix_moe_bridge.cpp in multiple locations."""
    candidates = []
    # 1. Relative to this file: ex_engine/csrc/
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.join(here, "..", "csrc", "ix_moe_bridge.cpp"))
    # 2. Deployed path inside vllm model dir
    candidates.append(os.path.join(here, "ix_moe_bridge.cpp"))
    # 3. /workspace paths
    candidates.append("/workspace/ex_engine/csrc/ix_moe_bridge.cpp")
    candidates.append("/workspace/qwen3_6_scripts/ix_moe_bridge.cpp")

    for c in candidates:
        p = os.path.normpath(c)
        if os.path.exists(p):
            return p
    return None


def _load_bridge():
    """JIT-compile and load ix_moe_bridge.so — called once."""
    global _ix_bridge, _ix_bridge_loaded, _ix_bridge_available
    if _ix_bridge_loaded:
        return _ix_bridge_available
    _ix_bridge_loaded = True

    cpp_file = _find_cpp_source()
    if cpp_file is None:
        logger.warning("ix_moe_bridge.cpp not found in any search path")
        return False

    try:
        from torch.utils.cpp_extension import load
        logger.info("JIT-compiling ix_moe_bridge.cpp from %s ...", cpp_file)
        _ix_bridge = load(
            name="ix_moe_bridge",
            sources=[cpp_file],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=False,
        )
        _ix_bridge_available = True
        fns = [x for x in dir(_ix_bridge) if not x.startswith("_")]
        logger.info("ix_moe_bridge loaded: %s", fns)
        return True
    except Exception as e:
        logger.warning("ix_moe_bridge JIT compile failed: %s", e)
        return False


def is_available() -> bool:
    """Check if bridge is available (lazy-load on first call)."""
    if not _ix_bridge_loaded:
        _load_bridge()
    return _ix_bridge_available


# =========================================================================
# Individual ops (thin wrappers with type safety)
# =========================================================================

def topk_softmax(
    gating_output: torch.Tensor,
    topk: int,
    renormalize: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Fused topk+softmax via ixformer::infer::topk_softmax.
    Returns: (topk_weights [T, K] fp32, topk_ids [T, K] int32)
    """
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available — JIT compile failed")
    return _ix_bridge.topk_softmax(gating_output, topk, renormalize)


def moe_gen_idx(
    expert_id: torch.Tensor,
    expert_num: int,
) -> List[torch.Tensor]:
    """
    Build expert permutation maps.
    Returns: [src_dst, dst_src, expert_sizes, cumsum]
    """
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available")
    return _ix_bridge.moe_gen_idx(expert_id, expert_num)


def moe_expand_input(
    input: torch.Tensor,
    gather_index: torch.Tensor,
    combine_idx: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    """Gather tokens by expert assignment."""
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available")
    return _ix_bridge.moe_expand_input(input, gather_index, combine_idx, topk)


def group_gemm(
    inputs: torch.Tensor,
    weights: torch.Tensor,
    token_count: torch.Tensor,
    output_n: int,
) -> torch.Tensor:
    """Batched expert GEMM via ixformer."""
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available")
    return _ix_bridge.group_gemm(inputs, weights, token_count, output_n)


def silu_and_mul(input: torch.Tensor) -> torch.Tensor:
    """Fused SiLU gate activation."""
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available")
    return _ix_bridge.silu_and_mul(input)


def moe_combine_result(
    input: torch.Tensor,
    weight: torch.Tensor,
) -> torch.Tensor:
    """Weighted reduce for MoE output."""
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available")
    return _ix_bridge.moe_combine_result(input, weight)


# =========================================================================
# Full fused MoE forward — replaces _pure_pytorch_experts() entirely
# =========================================================================

def fused_moe_forward(
    hidden_states: torch.Tensor,    # (T, H)
    router_logits: torch.Tensor,    # (T, E)
    w13: torch.Tensor,              # (E, 2*I, H) gate_up
    w2: torch.Tensor,               # (E, H, I) down
    topk: int,
    num_experts: int,
    renormalize: bool = True,
) -> torch.Tensor:
    """
    Full fused MoE forward via ixformer C++ pipeline.

    Pipeline: topk → gen_idx → expand → gemm1(w13) → silu → gemm2(w2) → combine

    Returns: (T, H) — partial output, needs all-reduce after.
    """
    if not is_available():
        raise RuntimeError("ix_moe_bridge not available")
    return _ix_bridge.fused_moe_forward(
        hidden_states, router_logits, w13, w2, topk, num_experts, renormalize
    )
