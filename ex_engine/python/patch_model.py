"""
ex_engine/python/patch_model.py — Wire EX Engine factors into vllm model

CCCL parallel: CCCL's dispatch_reduce.cuh has a Dispatch() that selects
the tuned kernel based on compute_capability. This patch does the same:
it replaces the PyTorch fallback paths with EX factor kernel calls.

Patched paths:
  1. Qwen3_5MoeSparseBlock._pure_pytorch_experts()
     → Uses EX factor 0 (moe_topk_softmax) for routing
     → Falls back to PyTorch GEMM for expert computation (factor 2 TBD)

  2. GatedDeltaNet.forward() prefill path
     → Uses EX factor 5 (gdn_chunk_fwd) instead of _torch_chunk_gated_delta_rule
     → Eliminates NaN by using fp32 accumulation

Integration:
  Called from patch_ops.sh during Docker build, or imported at runtime:
    python -c "from ex_engine.python.patch_model import apply_patches; apply_patches()"
"""

import logging
import os
import torch
import types

logger = logging.getLogger("ex_engine.patch")


def apply_patches(build_dir: str = "/workspace/ex_engine/build"):
    """
    Apply EX Engine patches to the loaded vllm model modules.
    Must be called AFTER vllm modules are imported.
    """
    # Lazy import to avoid circular deps
    try:
        from ex_engine.python.ex_loader import EXEngine
    except ImportError:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from ex_engine.python.ex_loader import EXEngine

    engine = EXEngine(build_dir)
    loaded = engine.load_all()

    if loaded == 0:
        logger.warning("EX Engine: no factors loaded, skipping patches")
        return

    logger.info("EX Engine: %d factors loaded, applying patches", loaded)

    # -----------------------------------------------------------------------
    # Patch 1: MoE routing — replace softmax+topk with fused factor
    # -----------------------------------------------------------------------
    if engine.has_factor(0):  # EX_FACTOR_MOE_TOPK_SOFTMAX
        _patch_moe_routing(engine)

    # -----------------------------------------------------------------------
    # Patch 2: GDN prefill — replace _torch_chunk_gated_delta_rule
    # -----------------------------------------------------------------------
    if engine.has_factor(5):  # EX_FACTOR_GDN_CHUNK_FWD
        _patch_gdn_prefill(engine)

    logger.info("EX Engine: patches applied successfully")


def _patch_moe_routing(engine):
    """
    Replace the pure PyTorch softmax→topk→renormalize in MoE with
    fused EX factor kernel.

    Target: Qwen3_5MoeSparseBlock._pure_pytorch_experts()
    The first 3 lines:
        routing_weights = _ix_softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(routing_weights, self.top_k, dim=-1)
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    """
    try:
        from vllm.model_executor.models import qwen3_5 as m
    except ImportError:
        logger.warning("Cannot import qwen3_5, skipping MoE patch")
        return

    if not hasattr(m, 'Qwen3_5MoeSparseBlock'):
        logger.warning("Qwen3_5MoeSparseBlock not found, skipping MoE patch")
        return

    original_fn = m.Qwen3_5MoeSparseBlock._pure_pytorch_experts

    def patched_experts(self, hidden_states, router_logits):
        # EX fused topk+softmax (1 kernel instead of 2 + 1 normalize)
        topk_weights, topk_ids = engine.moe_topk_softmax(
            router_logits, top_k=self.top_k)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Expert computation still uses PyTorch path
        # (factor 2 will replace this with batched GEMM later)
        w13 = self.experts.w13_weight
        w2  = self.experts.w2_weight
        T = hidden_states.shape[0]

        if T == 1:
            # Decode fast path (same as original)
            eids = topk_ids[0]
            ws = topk_weights[0]
            w13_sel = w13[eids]
            w2_sel = w2[eids]
            H = hidden_states.shape[-1]

            gate_up = torch.nn.functional.linear(
                hidden_states, w13_sel.reshape(-1, H))
            gate_up = gate_up.view(self.top_k, -1)
            gate, up = gate_up.chunk(2, dim=-1)
            act = torch.nn.functional.silu(gate) * up
            expert_out = torch.bmm(w2_sel, act.unsqueeze(-1)).squeeze(-1)
            out = (expert_out * ws.unsqueeze(-1)).sum(0, keepdim=True)
            return out.to(hidden_states.dtype)
        else:
            # Prefill path — loop over experts
            out = torch.zeros_like(hidden_states)
            unique_eids = topk_ids.view(-1).unique().tolist()
            for eid in unique_eids:
                eid = int(eid)
                mask = (topk_ids == eid)
                tok_ids, topk_pos = mask.nonzero(as_tuple=True)
                tokens = hidden_states[tok_ids]
                gate_up = torch.nn.functional.linear(tokens, w13[eid])
                gate, up = gate_up.chunk(2, dim=-1)
                act = torch.nn.functional.silu(gate) * up
                expert_out = torch.nn.functional.linear(act, w2[eid])
                weights = topk_weights[tok_ids, topk_pos].unsqueeze(-1)
                out.index_add_(0, tok_ids,
                             (expert_out * weights).to(out.dtype))
            return out

    m.Qwen3_5MoeSparseBlock._pure_pytorch_experts = patched_experts
    logger.info("EX Patched: MoE routing → fused topk_softmax factor")


def _patch_gdn_prefill(engine):
    """
    Replace _torch_chunk_gated_delta_rule with EX factor 5 (gdn_chunk_fwd).
    This eliminates the NaN problem by using fp32 state accumulation.
    """
    try:
        from vllm.model_executor.models import qwen3_5 as m
    except ImportError:
        logger.warning("Cannot import qwen3_5, skipping GDN patch")
        return

    if not hasattr(m, '_torch_chunk_gated_delta_rule'):
        logger.warning("_torch_chunk_gated_delta_rule not found, skipping GDN patch")
        return

    original_fn = m._torch_chunk_gated_delta_rule

    def patched_gdn_chunk(q, k, v, gate, beta, chunk_size, state):
        """
        EX factor replacement for _torch_chunk_gated_delta_rule.

        Args match the original function signature:
            q: (1, L, H, D) or (B, L, H, D)
            k, v: same shape
            gate: (1, L, H) or (B, L, H)
            beta: same shape
            chunk_size: int (ignored — factor processes full sequence)
            state: (B, H, D, D)

        Returns: (output, new_state)
        """
        B = q.shape[0]
        L = q.shape[1]
        H = q.shape[2]
        D = q.shape[3]

        # Ensure contiguous and correct dtype
        q_c = q.contiguous().half()
        k_c = k.contiguous().half()
        v_c = v.contiguous().half()
        g_c = gate.float().contiguous()
        b_c = beta.float().contiguous()
        s_c = state.float().contiguous()

        output, new_state = engine.gdn_chunk_fwd(
            q_c, k_c, v_c, g_c, b_c, s_c)

        return output, new_state

    m._torch_chunk_gated_delta_rule = patched_gdn_chunk
    logger.info("EX Patched: GDN prefill → gdn_chunk_fwd factor (NaN-free)")


# ---------------------------------------------------------------------------
# Auto-apply on import if build dir exists
# ---------------------------------------------------------------------------
_AUTO_BUILD_DIR = os.environ.get("EX_ENGINE_BUILD_DIR", "/workspace/ex_engine/build")
if os.path.isdir(_AUTO_BUILD_DIR):
    try:
        apply_patches(_AUTO_BUILD_DIR)
    except Exception as e:
        logger.warning("EX Engine auto-apply failed: %s", e)
