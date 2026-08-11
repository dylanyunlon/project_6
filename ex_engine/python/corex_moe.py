"""
corex_moe.py — Fused MoE dispatch for BI-V100 via ix_moe_bridge.so

Sub168 log reference:
  corex_moe.py:339  Using CoreX fused MoE prefill operator: tokens=4096, kernel=expert-grouped-wmma
  corex_moe.py:249  Using CoreX fused MoE decode operator

Call chain:
  qwen3_5.py → FusedMoE.forward() → corex_moe.forward()
    → ix_moe_bridge.topk_softmax()  (Step 1: routing)
    → ix_moe_bridge.moe_gen_idx()   (Step 2: index generation)
    → ix_moe_bridge.moe_expand_input() (Step 3: expand)
    → ix_moe_bridge.moe_group_gemm()  (Step 4: w13 gate+up GEMM)
    → ix_moe_bridge.silu_and_mul()    (Step 5: activation)
    → ix_moe_bridge.moe_group_gemm()  (Step 6: w2 down GEMM)
    → ix_moe_bridge.moe_combine_result() (Step 7: weighted sum)

Source: upstream_ref/xllm/xllm/core/kernels/ilu/fused_moe.cpp
        upstream_ref/xllm/xllm/core/kernels/ilu/ixformer.h
"""

import logging
import os
import glob
import torch
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ============================================================================
# Load ix_moe_bridge.so — compiled by precompile_ix_bridge.py in Docker
# ============================================================================
_bridge = None
_bridge_load_attempted = False


def _load_bridge():
    """Try to load ix_moe_bridge.so from known paths."""
    global _bridge, _bridge_load_attempted
    if _bridge_load_attempted:
        return _bridge
    _bridge_load_attempted = True

    search_paths = [
        "/usr/local/corex/lib/python3/dist-packages/ex_engine/build",
        "/usr/local/corex/lib/python3/dist-packages/ex_engine",
        "/usr/local/corex/lib/python3/dist-packages",
        "/workspace/ex_engine/build",
        "/workspace/ex_engine",
    ]

    for d in search_paths:
        for so in glob.glob(os.path.join(d, "ix_moe_bridge*.so")):
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location("ix_moe_bridge", so)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _bridge = mod
                logger.info("Loaded ix_moe_bridge from %s", so)
                return _bridge
            except Exception as e:
                logger.debug("Failed loading %s: %s", so, e)

    # Fallback: try torch.ops (if registered via JIT during build)
    try:
        import torch.utils.cpp_extension
        _bridge = torch.utils.cpp_extension.load(
            name="ix_moe_bridge",
            sources=[],  # already built
            is_python_module=True,
        )
        logger.info("Loaded ix_moe_bridge via torch extension cache")
        return _bridge
    except Exception:
        pass

    logger.warning("ix_moe_bridge.so not found — MoE will use PyTorch fallback (SLOW)")
    return None


class CoreXMoE:
    """
    Fused MoE operator matching qwen3_5.py FusedMoE call convention.

    Interface:
        forward(hidden_states, router_logits, w13, w2, topk, renormalize,
                num_expert_groups=0, topk_group=0, n_shared_experts=0,
                shared_expert_gate=None, shared_w13=None, shared_w2=None)
        → (output, shared_expert_output_or_None)
    """

    def __init__(self, num_experts: int = 64, topk: int = 8):
        self.num_experts = num_experts
        self.topk = topk
        self._bridge = _load_bridge()
        self._prefill_logged = False
        self._decode_logged = False

    def forward(
        self,
        hidden_states: torch.Tensor,      # (num_tokens, hidden_size)
        router_logits: torch.Tensor,       # (num_tokens, num_experts)
        w13: torch.Tensor,                 # (num_local_experts, 2*intermediate, hidden)
        w2: torch.Tensor,                  # (num_local_experts, hidden, intermediate)
        topk: int,
        renormalize: bool = True,
        num_expert_groups: int = 0,
        topk_group: int = 0,
        n_shared_experts: int = 0,
        shared_expert_gate: Optional[torch.Tensor] = None,
        shared_w13: Optional[torch.Tensor] = None,
        shared_w2: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full fused MoE forward via ixformer C++ bridge."""

        num_tokens = hidden_states.size(0)
        hidden_size = hidden_states.size(1)
        num_local_experts = w13.size(0)

        # Log once per mode (match Sub168 log format)
        if num_tokens > 1 and not self._prefill_logged:
            logger.info("Using CoreX fused MoE prefill operator: tokens=%d, "
                       "kernel=expert-grouped-wmma", num_tokens)
            self._prefill_logged = True
        elif num_tokens == 1 and not self._decode_logged:
            logger.info("Using CoreX fused MoE decode operator")
            self._decode_logged = True

        if self._bridge is not None:
            return self._forward_bridge(
                hidden_states, router_logits, w13, w2, topk,
                renormalize, num_local_experts, hidden_size)
        else:
            return self._forward_pytorch(
                hidden_states, router_logits, w13, w2, topk,
                renormalize, num_local_experts, hidden_size)

    def _forward_bridge(
        self, hidden_states, router_logits, w13, w2,
        topk, renormalize, num_local_experts, hidden_size
    ) -> torch.Tensor:
        """7-step fused MoE via ix_moe_bridge.so → ixformer::infer."""
        bridge = self._bridge
        num_tokens = hidden_states.size(0)
        num_experts = router_logits.size(1)

        # Step 1: topk_softmax
        gating = router_logits.to(torch.float32)
        topk_weights = torch.empty(
            (num_tokens, topk), dtype=torch.float32, device=hidden_states.device)
        topk_ids = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=hidden_states.device)
        token_expert_indices = torch.empty(
            (num_tokens, topk), dtype=torch.int32, device=hidden_states.device)

        bridge.topk_softmax(topk_weights, topk_ids, token_expert_indices, gating)

        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # Step 2: generate index
        idx_result = bridge.moe_gen_idx(topk_ids, num_experts)
        src_dst, dst_src, expert_sizes, expert_sizes_cumsum = idx_result

        # Step 3: expand input
        expanded = bridge.moe_expand_input(
            hidden_states, src_dst, dst_src, topk)

        # Step 4: group GEMM 1 (w13: gate + up projection)
        intermediate_size_2x = w13.size(1)
        gemm1_out = expanded.new_empty((expanded.size(0), intermediate_size_2x))
        expert_sizes_cpu = expert_sizes.cpu()
        bridge.moe_group_gemm(gemm1_out, expanded, w13, expert_sizes_cpu,
                              intermediate_size_2x)

        # Step 5: silu_and_mul activation
        act_out = bridge.silu_and_mul(gemm1_out)

        # Step 6: group GEMM 2 (w2: down projection)
        gemm2_out = act_out.new_empty((act_out.size(0), hidden_size))
        bridge.moe_group_gemm(gemm2_out, act_out, w2, expert_sizes_cpu,
                              hidden_size)

        # Step 7: combine result (weighted sum back to original token order)
        final = bridge.moe_combine_result(gemm2_out, topk_weights)

        return final

    def _forward_pytorch(
        self, hidden_states, router_logits, w13, w2,
        topk, renormalize, num_local_experts, hidden_size
    ) -> torch.Tensor:
        """Pure PyTorch fallback — SLOW but correct."""
        num_tokens = hidden_states.size(0)

        # Softmax routing
        scores = torch.softmax(router_logits.float(), dim=-1)
        topk_weights, topk_ids = torch.topk(scores, topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # Expert loop
        final = torch.zeros(
            (num_tokens, hidden_size),
            dtype=hidden_states.dtype, device=hidden_states.device)

        for i in range(num_local_experts):
            mask = (topk_ids == i).any(dim=-1)
            if not mask.any():
                continue
            idx = mask.nonzero(as_tuple=True)[0]
            token_sel = hidden_states[idx]

            # Weight for this expert per token
            expert_weights = torch.zeros(
                idx.size(0), dtype=topk_weights.dtype, device=hidden_states.device)
            for k in range(topk):
                k_mask = topk_ids[idx, k] == i
                expert_weights[k_mask] += topk_weights[idx[k_mask], k]

            # gate+up → silu_and_mul → down
            gate_up = torch.mm(token_sel, w13[i].t())
            half_dim = gate_up.size(-1) // 2
            gate = gate_up[:, :half_dim]
            up = gate_up[:, half_dim:]
            activated = torch.nn.functional.silu(gate) * up
            down = torch.mm(activated, w2[i].t())

            final[idx] += down * expert_weights.unsqueeze(-1)

        return final
