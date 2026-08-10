"""
ix_bridge.py — Load ix_moe_bridge C++ extension at runtime.

Calls ixformer::infer::topk_softmax() via C++ torch extension,
bypassing the missing Python binding in ixformer.functions.

Build: JIT-compiled on first import via torch.utils.cpp_extension.load()
       (same mechanism as flash_qla_sm70 GDN kernel — proven to work on BI-V100)
"""

import os
import logging
import torch

logger = logging.getLogger("ex_engine.ix_bridge")

_ix_bridge = None
_ix_bridge_available = False

def _load_bridge():
    """JIT-compile and load ix_moe_bridge.so"""
    global _ix_bridge, _ix_bridge_available
    if _ix_bridge is not None:
        return _ix_bridge_available
    
    csrc_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc")
    cpp_file = os.path.join(csrc_dir, "ix_moe_bridge.cpp")
    
    if not os.path.exists(cpp_file):
        # Try deployed path (inside vllm model dir)
        alt_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_file = os.path.join(alt_dir, "ix_moe_bridge.cpp")
    
    if not os.path.exists(cpp_file):
        logger.warning("ix_moe_bridge.cpp not found at %s", cpp_file)
        _ix_bridge_available = False
        return False
    
    try:
        from torch.utils.cpp_extension import load
        logger.info("JIT-compiling ix_moe_bridge.cpp ...")
        _ix_bridge = load(
            name="ix_moe_bridge",
            sources=[cpp_file],
            extra_cflags=["-O2"],
            verbose=False,
        )
        _ix_bridge_available = True
        logger.info("ix_moe_bridge loaded successfully: %s", dir(_ix_bridge))
        return True
    except Exception as e:
        logger.warning("ix_moe_bridge JIT compile failed: %s", e)
        _ix_bridge_available = False
        return False

def topk_softmax(gating_output: torch.Tensor, topk: int, renormalize: bool = True):
    """
    Fused topk+softmax via ixformer C++ API.
    
    Args:
        gating_output: (num_tokens, num_experts) router logits
        topk: number of experts to select
        renormalize: whether to renormalize weights
    
    Returns:
        (topk_weights, topk_indices) — both (num_tokens, topk)
    """
    if not _ix_bridge_available:
        if not _load_bridge():
            # Fallback to pure PyTorch
            probs = torch.softmax(gating_output.float(), dim=-1)
            topk_w, topk_ids = torch.topk(probs, topk, dim=-1)
            if renormalize:
                topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True)
            return topk_w, topk_ids.to(torch.int32)
    
    return _ix_bridge.topk_softmax(gating_output, topk, renormalize)
