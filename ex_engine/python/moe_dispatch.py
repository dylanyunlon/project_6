"""moe_dispatch.py — MoE forward using ix_unified 3-tier dispatch.

Replaces the pure-PyTorch for-loop over 64 experts with the ixformer
7-step pipeline (from upstream xllm/core/layers/ilu/fused_moe.cpp):

  1. topk_softmax     → select top-K experts per token
  2. moe_gen_idx      → compute scatter/gather index mapping
  3. moe_expand_input → expand tokens by topK
  4. group_gemm (w13) → gate+up projection for all experts
  5. silu_and_mul     → activation
  6. group_gemm (w2)  → down projection
  7. moe_combine      → weighted reduce back to [n_tokens, hidden]

Falls back to PyTorch per-expert loop if ix_unified bridge is unavailable.
"""

import torch
import logging

logger = logging.getLogger("moe_dispatch")

try:
    from ex_engine.python.ix_unified import ix as _ix
except ImportError:
    try:
        from ix_unified import ix as _ix
    except ImportError:
        _ix = None
        logger.warning("ix_unified not available, MoE uses pure PyTorch")


def moe_forward_unified(
    hidden_states: torch.Tensor,   # [num_tokens, hidden_size]
    gate_logits: torch.Tensor,     # [num_tokens, num_experts]
    w13_weight: torch.Tensor,      # [num_experts, 2*intermediate, hidden]
    w2_weight: torch.Tensor,       # [num_experts, hidden, intermediate]
    topk: int = 8,
    renormalize: bool = True,
    num_experts: int = 64,
) -> torch.Tensor:
    """Full MoE forward with ix_unified dispatch.

    Returns: [num_tokens, hidden_size]
    """
    if _ix is None or not hasattr(_ix, '_bridge') or _ix._bridge is None:
        # No C++ bridge → use Python-loop fallback directly
        return _moe_pytorch_fallback(
            hidden_states, gate_logits, w13_weight, w2_weight,
            topk, renormalize, num_experts)

    try:
        return _moe_bridge_pipeline(
            hidden_states, gate_logits, w13_weight, w2_weight,
            topk, renormalize, num_experts)
    except Exception as e:
        logger.warning("MoE bridge pipeline failed (%s), fallback to PyTorch", e)
        return _moe_pytorch_fallback(
            hidden_states, gate_logits, w13_weight, w2_weight,
            topk, renormalize, num_experts)


def _moe_bridge_pipeline(
    hidden_states, gate_logits, w13_weight, w2_weight,
    topk, renormalize, num_experts,
):
    """7-step MoE pipeline using ix_unified bridge."""
    n_tokens = hidden_states.size(0)

    # Step 1: topk_softmax
    topk_weights, topk_indices = _ix.moe_topk_softmax(
        gate_logits, topk, renormalize)

    # Step 2: compute token→expert index mapping
    expert_ids_flat = topk_indices.view(-1).to(torch.int32)
    src_dst, dst_src, expert_sizes, expert_cumsum = _ix.moe_gen_idx(
        expert_ids_flat, num_experts)

    # Step 3: expand input
    expanded = _ix.moe_expand_input(
        hidden_states, src_dst, dst_src, topk)

    # Step 4: group GEMM w13 (gate+up projection)
    gate_up = _ix.moe_group_gemm(expanded, w13_weight, expert_sizes)

    # Step 5: silu_and_mul activation
    activated = _ix.silu_and_mul(gate_up)

    # Step 6: group GEMM w2 (down projection)
    down = _ix.moe_group_gemm(activated, w2_weight, expert_sizes)

    # Step 7: combine results (weighted sum over topk experts)
    down_topk = down.view(n_tokens, topk, -1)
    output = _ix.moe_combine_result(down_topk, topk_weights)

    return output


def _moe_pytorch_fallback(
    hidden_states, gate_logits, w13_weight, w2_weight,
    topk, renormalize, num_experts,
):
    """Pure-PyTorch MoE fallback — per-expert loop."""
    n_tokens, hidden = hidden_states.shape

    # Gating
    scores = torch.softmax(gate_logits.float(), dim=-1)
    topk_weights, topk_indices = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    topk_weights = topk_weights.to(hidden_states.dtype)

    output = torch.zeros_like(hidden_states)

    for i in range(n_tokens):
        for j in range(topk):
            expert_id = topk_indices[i, j].item()
            w = topk_weights[i, j]

            # w13: [2*intermediate, hidden]
            gate_up = hidden_states[i] @ w13_weight[expert_id].t()
            intermediate = gate_up.size(-1) // 2
            gate_val = gate_up[:intermediate]
            up_val = gate_up[intermediate:]
            activated = torch.sigmoid(gate_val) * up_val

            # w2: [hidden, intermediate]
            down = activated @ w2_weight[expert_id].t()
            output[i] += w * down

    return output


def moe_topk_gating(
    gate_logits: torch.Tensor,
    topk: int,
    renormalize: bool = True,
):
    """Standalone gating — just topk + softmax."""
    if _ix is not None:
        return _ix.moe_topk_softmax(gate_logits, topk, renormalize)
    scores = torch.softmax(gate_logits.float(), dim=-1)
    weights, indices = torch.topk(scores, k=topk, dim=-1)
    if renormalize:
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights, indices.to(torch.int32)
