"""moe_fused_dispatch.py — Three-tier MoE dispatch (CCCL policy_selector pattern).

Port of upstream_ref/xllm/core/layers/ilu/fused_moe.cpp 7-step pipeline.

Dispatch hierarchy:
  Tier 0: ix_unified_bridge.so → ixformer::infer 7-step C++ pipeline
          topk_softmax → gen_idx → expand_input → group_gemm(w13) →
          silu_and_mul → group_gemm(w2) → combine_result
  Tier 1: corex prebuilt .so → direct_routed.w13/.w2_reduce (decode T=1 only)
  Tier 2: PyTorch fallback → per-expert F.linear loop

Usage in qwen3_5.py:
    from ex_engine.python.moe_fused_dispatch import fused_moe_forward
    out = fused_moe_forward(hidden_states, router_logits, w13, w2,
                            top_k=8, num_experts=256, act_fn=silu_and_mul)
"""

import logging
from typing import Callable, Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger("moe_fused_dispatch")

# Lazy imports — set at first call
_ix = None
_corex = None
_init_done = False


def _lazy_init():
    global _ix, _corex, _init_done
    if _init_done:
        return
    _init_done = True

    # Tier 0: ix_unified
    try:
        from ex_engine.python.ix_unified import ix
        if ix._bridge is not None:
            _ix = ix
            logger.info("moe_fused_dispatch: Tier0 ix_unified_bridge.so available")
        else:
            logger.info("moe_fused_dispatch: Tier0 unavailable (bridge=None)")
    except Exception as e:
        logger.info("moe_fused_dispatch: Tier0 unavailable (%s)", e)

    # Try import path used on real hardware
    if _ix is None:
        try:
            from ix_unified import ix
            if ix._bridge is not None:
                _ix = ix
                logger.info("moe_fused_dispatch: Tier0 ix_unified (direct) available")
        except Exception:
            pass

    # Tier 1: corex prebuilt .so
    try:
        from ex_engine.python.corex_so_loader import corex
        if corex.moe_direct_routed is not None:
            _corex = corex
            logger.info("moe_fused_dispatch: Tier1 corex prebuilt .so available")
    except Exception as e:
        logger.info("moe_fused_dispatch: Tier1 unavailable (%s)", e)


def _tier0_fused_moe(
    hidden_states: torch.Tensor,   # [T, H]
    router_logits: torch.Tensor,   # [T, E]
    w13: torch.Tensor,             # [E, 2*I, H]
    w2: torch.Tensor,              # [E, H, I]
    top_k: int,
    num_experts: int,
    act_fn: Callable,
) -> torch.Tensor:
    """Tier 0: Full 7-step ixformer::infer pipeline via ix_unified_bridge.so.

    Maps 1:1 to xllm/core/layers/ilu/fused_moe.cpp::forward().
    """
    T, H = hidden_states.shape

    # Step 1: topk_softmax — fused softmax + topk selection
    topk_weights, topk_ids = _ix.moe_topk_softmax(router_logits, top_k,
                                                    renormalize=True)

    # Step 2: gen_idx — compute scatter/gather indices for expert routing
    idx_result = _ix.moe_gen_idx(topk_ids, num_experts)
    src_dst, dst_src, expert_sizes, cumsum = idx_result

    # Step 3: expand_input — scatter tokens to expert order
    expanded = _ix.moe_expand_input(hidden_states, dst_src, src_dst, top_k)

    # Step 4: group_gemm(w13) — batched GEMM across all experts
    gate_up = _ix.moe_group_gemm(expanded, w13, expert_sizes)

    # Step 5: activation — SiLU(gate) * up
    act = act_fn(gate_up)

    # Step 6: group_gemm(w2) — down projection
    down = _ix.moe_group_gemm(act, w2, expert_sizes)

    # Step 7: combine_result — gather back and weighted sum
    output = _ix.moe_combine_result(
        down.view(T, top_k, H), topk_weights)

    return output


def _tier1_decode_single_token(
    hidden_states: torch.Tensor,   # [1, H]
    expert_ids: torch.Tensor,      # [K]
    weights: torch.Tensor,         # [K]
    w13: torch.Tensor,             # [E, 2*I, H]
    w2: torch.Tensor,              # [E, H, I]
    act_fn: Callable,
) -> torch.Tensor:
    """Tier 1: Single-token decode via prebuilt corex_moe_direct_routed.so.

    Only works for T=1 decode. The .so implements fused expert indexing +
    GEMM + reduction in a single kernel launch.
    """
    gate_up = _corex.moe_direct_routed.w13(hidden_states, w13, expert_ids)
    act = act_fn(gate_up)
    return _corex.moe_direct_routed.w2_reduce(act, w2, expert_ids, weights)


def _tier2_pytorch_loop(
    hidden_states: torch.Tensor,   # [T, H]
    router_logits: torch.Tensor,   # [T, E]
    w13: torch.Tensor,             # [E, 2*I, H]
    w2: torch.Tensor,              # [E, H, I]
    top_k: int,
    act_fn: Callable,
) -> torch.Tensor:
    """Tier 2: Pure PyTorch per-expert loop (always works, slowest)."""
    T, H = hidden_states.shape

    # Softmax → topk
    topk_logits, topk_ids = torch.topk(router_logits.float(), top_k, dim=-1)
    topk_weights = torch.softmax(topk_logits, dim=-1).to(hidden_states.dtype)

    if T == 1:
        # Fast single-token path: batched GEMM
        eids = topk_ids[0]
        ws = topk_weights[0]
        w13_sel = w13[eids]
        w2_sel = w2[eids]
        gate_up = F.linear(hidden_states, w13_sel.reshape(-1, H))
        gate_up = gate_up.view(top_k, -1)
        act = act_fn(gate_up)
        expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
        return (expert_out * ws.unsqueeze(-1)).sum(0, keepdim=True).to(
            hidden_states.dtype)
    else:
        # General prefill path: sorted per-expert loop
        out = torch.zeros_like(hidden_states)
        flat_eids = topk_ids.reshape(-1)
        order = torch.argsort(flat_eids, stable=True)
        sorted_tok_ids = torch.arange(
            T, device=topk_ids.device).repeat_interleave(top_k)[order]
        sorted_weights = topk_weights.reshape(-1)[order]
        expert_counts = torch.bincount(
            flat_eids, minlength=w13.shape[0]).tolist()

        start = 0
        for eid, count in enumerate(expert_counts):
            if count == 0:
                continue
            end = start + count
            tok_ids = sorted_tok_ids[start:end]
            tokens = hidden_states[tok_ids]
            gate_up = F.linear(tokens, w13[eid])
            act = act_fn(gate_up)
            expert_out = F.linear(act, w2[eid])
            weights_e = sorted_weights[start:end].unsqueeze(-1)
            out.index_add_(0, tok_ids, (expert_out * weights_e).to(out.dtype))
            start = end
        return out


def fused_moe_forward(
    hidden_states: torch.Tensor,   # [T, H]
    router_logits: torch.Tensor,   # [T, E]
    w13: torch.Tensor,             # [E, 2*I, H]
    w2: torch.Tensor,              # [E, H, I]
    top_k: int = 8,
    num_experts: int = 256,
    act_fn: Optional[Callable] = None,
) -> torch.Tensor:
    """Dispatch MoE through Tier 0 → 1 → 2.

    Returns partial output (pre all-reduce), same contract as vllm FusedMoE.
    """
    _lazy_init()

    if act_fn is None:
        def _default_act(x):
            gate, up = x.chunk(2, dim=-1)
            return F.silu(gate) * up
        act_fn = _default_act

    T = hidden_states.shape[0]

    # Tier 0: full ixformer pipeline (all sizes)
    if _ix is not None and _ix._bridge is not None:
        try:
            return _tier0_fused_moe(hidden_states, router_logits, w13, w2,
                                    top_k, num_experts, act_fn)
        except Exception as e:
            logger.warning("Tier0 MoE failed (%s), falling to Tier1/2", e)

    # Tier 1: corex direct routed (decode T=1 only)
    if (T == 1 and _corex is not None
            and _corex.moe_direct_routed is not None
            and hidden_states.dtype == torch.float16
            and w13.dtype == torch.float16
            and w2.dtype == torch.float16
            and hidden_states.is_contiguous()
            and w13.is_contiguous()
            and w2.is_contiguous()):
        try:
            topk_logits, topk_ids = torch.topk(
                router_logits.float(), top_k, dim=-1)
            topk_weights = torch.softmax(topk_logits, dim=-1).to(
                hidden_states.dtype)
            return _tier1_decode_single_token(
                hidden_states, topk_ids[0], topk_weights[0],
                w13, w2, act_fn)
        except Exception as e:
            logger.warning("Tier1 MoE failed (%s), falling to Tier2", e)

    # Tier 2: PyTorch fallback
    return _tier2_pytorch_loop(hidden_states, router_logits, w13, w2,
                               top_k, act_fn)
