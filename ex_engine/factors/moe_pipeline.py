"""
ex_engine/factors/moe_pipeline.py

Layer 2: MoE 7-step pipeline orchestrator

Upstream parallel: xllm_layers/ilu/fused_moe.cpp (806 lines)
  → FusedMoEImpl::forward_experts() orchestrates the full MoE hot path:
    Step 1: select_experts → moe_active_topk (topk_softmax)
    Step 2: moe_gen_idx → moe_compute_token_index (histogram + prefix_sum + place)
    Step 3: moe_expand_input (gather tokens by expert)
    Step 4: group_gemm (w13: gate_proj + up_proj fused)
    Step 5: activation (silu_and_mul on gated MLP)
    Step 6: group_gemm (w2: down_proj)
    Step 7: moe_combine_result (weighted reduce over topk experts)

This module mirrors the full 7-step pipeline. Each step dispatches
to the ix_ops_dispatch layer (Layer 4) which calls into ixformer::infer
C++ kernels. The pipeline ordering and tensor lifetime management
matches xllm upstream exactly.

Call chain:
  vllm model forward
    → Qwen3MoeSparseMoeBlock.forward()
      → moe_pipeline.fused_moe_forward()
        → Step 1-7 below
"""

import logging
from typing import Optional, Tuple

import torch

logger = logging.getLogger("ex_engine.moe_pipeline")


class MoEPipelineConfig:
    """
    Configuration for the MoE pipeline.

    Parallels xllm_layers/ilu/fused_moe.h FusedMoEArgs:
      num_total_experts_, topk_, hidden_size_, intermediate_size_,
      is_gated_, renormalize_, hidden_act_, scoring_func_
    """
    __slots__ = (
        'num_experts', 'topk', 'hidden_size', 'intermediate_size',
        'is_gated', 'renormalize', 'hidden_act', 'scoring_func',
        'tp_size', 'tp_rank', 'ep_size', 'ep_rank',
        'start_expert_id', 'num_experts_per_rank',
    )

    def __init__(
        self,
        num_experts: int = 64,
        topk: int = 8,
        hidden_size: int = 3584,
        intermediate_size: int = 18944,
        is_gated: bool = True,
        renormalize: bool = True,
        hidden_act: str = "silu",
        scoring_func: str = "softmax",
        tp_size: int = 4,
        tp_rank: int = 0,
        ep_size: int = 1,
        ep_rank: int = 0,
    ):
        self.num_experts = num_experts
        self.topk = topk
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.is_gated = is_gated
        self.renormalize = renormalize
        self.hidden_act = hidden_act
        self.scoring_func = scoring_func
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.ep_size = ep_size
        self.ep_rank = ep_rank
        self.num_experts_per_rank = num_experts // ep_size
        self.start_expert_id = ep_rank * self.num_experts_per_rank


class MoEPipeline:
    """
    7-step MoE pipeline matching xllm FusedMoEImpl::forward_experts().

    Each step calls through the dispatch layer. The pipeline manages
    intermediate tensor lifetimes to minimize GPU memory pressure,
    matching xllm's explicit tensor release pattern:
      - expand_hidden_states released after Step 6
      - act_out released after Step 6
    """

    def __init__(self, config: MoEPipelineConfig, dispatch_module=None):
        self.config = config
        # The dispatch module provides the per-op kernel calls
        # At runtime this is ix_ops_dispatch or direct ixformer
        if dispatch_module is None:
            try:
                from ex_engine.python import ix_ops_dispatch
                self.dispatch = ix_ops_dispatch
            except ImportError:
                self.dispatch = None
                logger.warning("ix_ops_dispatch not available, MoE pipeline "
                               "will use PyTorch fallbacks")
        else:
            self.dispatch = dispatch_module

    # ===================================================================
    # Step 1: Router — softmax + topk (36× per layer, 64 layers)
    # ===================================================================
    # Upstream: FusedMoEImpl::select_experts → kernel::ilu::moe_active_topk
    #   → infer::topk_softmax
    #   → cuda::moe_topk_softmax_kernels.cuh::topkGatingSoftmax

    def step1_topk_route(
        self,
        router_logits: torch.Tensor,  # (num_tokens, num_experts)
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            topk_weights: (num_tokens, topk) float32, renormalized
            topk_ids: (num_tokens, topk) int32
        """
        if self.dispatch is not None:
            try:
                return self.dispatch.topk_softmax(
                    router_logits, self.config.topk, self.config.renormalize)
            except (RuntimeError, AttributeError) as e:
                logger.debug("topk_softmax dispatch failed: %s", e)

        # PyTorch fallback — matches xllm ilu::moe_active_topk
        logits_f32 = router_logits.float()
        probs = torch.softmax(logits_f32, dim=-1)
        topk_weights, topk_ids = torch.topk(probs, self.config.topk, dim=-1)
        if self.config.renormalize:
            topk_weights = topk_weights / topk_weights.sum(
                dim=-1, keepdim=True)
        return topk_weights, topk_ids.to(torch.int32)

    # ===================================================================
    # Step 2: Generate expert indices (permutation maps)
    # ===================================================================
    # Upstream: kernel::ilu::moe_gen_idx
    #   → infer::moe_compute_token_index_api
    #   → cuda::moe_compute_index (3-phase: histogram, prefix_sum, place)

    def step2_gen_idx(
        self,
        topk_ids: torch.Tensor,  # (num_tokens, topk) int32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build bidirectional permutation maps for expert dispatch.

        Returns:
            src_to_dst: (num_tokens * topk,) int32 — original → sorted position
            dst_to_src: (num_tokens * topk,) int32 — sorted → original position
            expert_sizes: (num_experts,) int32 — tokens per expert
        """
        flat_ids = topk_ids.view(-1)
        num_elements = flat_ids.shape[0]
        num_experts = self.config.num_experts
        device = topk_ids.device

        if self.dispatch is not None:
            try:
                return self.dispatch.moe_compute_token_index(
                    topk_ids, num_experts, self.config.start_expert_id)
            except (RuntimeError, AttributeError):
                pass

        # PyTorch fallback — matches cuda::moe_compute_index 3-phase logic
        # Phase 1: histogram
        expert_sizes = torch.zeros(
            num_experts, dtype=torch.int32, device=device)
        for eid in range(num_experts):
            expert_sizes[eid] = (flat_ids == eid).sum().to(torch.int32)

        # Phase 2: exclusive prefix sum
        expert_offsets = torch.zeros(
            num_experts, dtype=torch.int32, device=device)
        expert_offsets[1:] = torch.cumsum(expert_sizes[:-1], dim=0)

        # Phase 3: place indices
        dst_to_src = torch.empty(
            num_elements, dtype=torch.int32, device=device)
        src_to_dst = torch.empty(
            num_elements, dtype=torch.int32, device=device)
        offsets_scratch = expert_offsets.clone()

        for i in range(num_elements):
            eid = flat_ids[i].item()
            if 0 <= eid < num_experts:
                pos = offsets_scratch[eid].item()
                offsets_scratch[eid] += 1
                dst_to_src[pos] = i
                src_to_dst[i] = pos

        return src_to_dst, dst_to_src, expert_sizes

    # ===================================================================
    # Step 3: Expand input (gather tokens by expert ordering)
    # ===================================================================
    # Upstream: kernel::ilu::moe_expand_input
    #   → infer::moe_expand_input

    def step3_expand_input(
        self,
        hidden_states: torch.Tensor,  # (num_tokens, hidden_size)
        dst_to_src: torch.Tensor,     # (num_tokens * topk,) int32
    ) -> torch.Tensor:
        """
        Reorder tokens into expert-grouped order for batched GEMM.

        Returns:
            expanded: (num_tokens * topk, hidden_size) same dtype as input
        """
        if self.dispatch is not None:
            try:
                return self.dispatch.moe_expand_input(
                    hidden_states, dst_to_src, self.config.topk)
            except (RuntimeError, AttributeError):
                pass

        # PyTorch fallback
        src_indices = dst_to_src.long()
        # Each entry in dst_to_src is a flat index into the expanded token list.
        # The source token index is flat_idx // topk
        token_indices = src_indices // self.config.topk
        expanded = hidden_states[token_indices]
        return expanded

    # ===================================================================
    # Step 4: Group GEMM 1 — gate_proj + up_proj (w13)
    # ===================================================================
    # Upstream: kernel::ilu::group_gemm
    #   → infer::moe_w16a16_group_gemm
    # weight shape: (num_experts_per_rank, intermediate_size * 2, hidden_size)
    #   for gated MLP: w1 and w3 fused into one [2*inter, hidden] matrix

    def step4_gemm1(
        self,
        expanded_input: torch.Tensor,  # (total_tokens, hidden_size)
        w13: torch.Tensor,             # (E_local, inter*2, hidden) or flat
        expert_sizes: torch.Tensor,    # (num_experts,) int32
    ) -> torch.Tensor:
        """
        Group GEMM: expanded_input × w13^T for each expert group.

        Returns:
            gemm1_out: (total_tokens, intermediate_size * 2)
        """
        if self.dispatch is not None:
            try:
                inter2 = w13.shape[1] if w13.dim() == 3 else w13.shape[0]
                return self.dispatch.moe_group_gemm(
                    expanded_input, w13, expert_sizes, inter2)
            except (RuntimeError, AttributeError):
                pass

        # PyTorch fallback: loop over experts
        total_tokens = expanded_input.shape[0]
        out_dim = w13.shape[1] if w13.dim() == 3 else w13.shape[0]
        output = torch.empty(
            total_tokens, out_dim,
            dtype=expanded_input.dtype, device=expanded_input.device)

        offset = 0
        for e in range(expert_sizes.shape[0]):
            count = expert_sizes[e].item()
            if count > 0:
                local_e = e - self.config.start_expert_id
                if 0 <= local_e < w13.shape[0]:
                    x_e = expanded_input[offset:offset + count]
                    w_e = w13[local_e]  # (inter*2, hidden)
                    # matmul: (count, hidden) × (hidden, inter*2) = (count, inter*2)
                    output[offset:offset + count] = x_e @ w_e.t()
            offset += count

        return output

    # ===================================================================
    # Step 5: Activation — SiLU-and-mul for gated MLP
    # ===================================================================
    # Upstream: kernel::ilu::act_and_mul → infer::silu_and_mul
    # Input: (total_tokens, intermediate_size * 2)
    # Output: (total_tokens, intermediate_size)
    # Split input in half: out = silu(input[:, :inter]) * input[:, inter:]

    def step5_activation(
        self,
        gemm1_out: torch.Tensor,  # (total_tokens, inter*2)
    ) -> torch.Tensor:
        """
        Gated SiLU activation.

        Returns:
            act_out: (total_tokens, intermediate_size)
        """
        if self.config.is_gated:
            half_dim = gemm1_out.shape[-1] // 2
            gate = gemm1_out[:, :half_dim]
            up = gemm1_out[:, half_dim:]

            if self.dispatch is not None:
                try:
                    # ixformer expects concatenated input, produces half-width output
                    return self.dispatch.silu_and_mul(gemm1_out)
                except (RuntimeError, AttributeError):
                    pass

            # PyTorch fallback — explicit silu_and_mul
            return torch.nn.functional.silu(gate) * up
        else:
            if self.config.hidden_act == "silu":
                return torch.nn.functional.silu(gemm1_out)
            elif self.config.hidden_act == "gelu":
                return torch.nn.functional.gelu(gemm1_out)
            else:
                return gemm1_out

    # ===================================================================
    # Step 6: Group GEMM 2 — down_proj (w2)
    # ===================================================================
    # Upstream: kernel::ilu::group_gemm (same as Step 4, different weights)
    # weight shape: (num_experts_per_rank, hidden_size, intermediate_size)

    def step6_gemm2(
        self,
        act_out: torch.Tensor,        # (total_tokens, intermediate_size)
        w2: torch.Tensor,             # (E_local, hidden, inter)
        expert_sizes: torch.Tensor,   # (num_experts,) int32
    ) -> torch.Tensor:
        """
        Group GEMM: act_out × w2^T for each expert group.

        Returns:
            gemm2_out: (total_tokens, hidden_size)
        """
        if self.dispatch is not None:
            try:
                return self.dispatch.moe_group_gemm(
                    act_out, w2, expert_sizes, self.config.hidden_size)
            except (RuntimeError, AttributeError):
                pass

        # PyTorch fallback: loop over experts
        total_tokens = act_out.shape[0]
        output = torch.empty(
            total_tokens, self.config.hidden_size,
            dtype=act_out.dtype, device=act_out.device)

        offset = 0
        for e in range(expert_sizes.shape[0]):
            count = expert_sizes[e].item()
            if count > 0:
                local_e = e - self.config.start_expert_id
                if 0 <= local_e < w2.shape[0]:
                    x_e = act_out[offset:offset + count]
                    w_e = w2[local_e]  # (hidden, inter)
                    output[offset:offset + count] = x_e @ w_e.t()
            offset += count

        return output

    # ===================================================================
    # Step 7: Combine — weighted reduce over topk experts
    # ===================================================================
    # Upstream: kernel::ilu::moe_combine_result
    #   → infer::moe_output_reduce_sum
    #   → cuda::moe_combine_kernel
    # Reorder from expert-sorted back to token order, weighted sum.

    def step7_combine(
        self,
        gemm2_out: torch.Tensor,       # (total_tokens, hidden_size) sorted
        topk_weights: torch.Tensor,    # (num_tokens, topk) float32
        src_to_dst: torch.Tensor,      # (total_tokens,) int32
    ) -> torch.Tensor:
        """
        Weighted combine of expert outputs back to token order.

        Returns:
            final: (num_tokens, hidden_size)
        """
        num_tokens = topk_weights.shape[0]
        topk = self.config.topk
        hidden_size = gemm2_out.shape[-1]

        if self.dispatch is not None:
            try:
                return self.dispatch.moe_output_reduce_sum(
                    gemm2_out, topk_weights, 1.0)
            except (RuntimeError, AttributeError):
                pass

        # PyTorch fallback — matches cuda::moe_combine_kernel logic
        output = torch.zeros(
            num_tokens, hidden_size,
            dtype=gemm2_out.dtype, device=gemm2_out.device)

        for t in range(num_tokens):
            for k in range(topk):
                flat_idx = t * topk + k
                dst_pos = src_to_dst[flat_idx].long().item()
                w = topk_weights[t, k].item()
                output[t] += w * gemm2_out[dst_pos].float()

        return output.to(gemm2_out.dtype)

    # ===================================================================
    # Full forward — orchestrates all 7 steps
    # ===================================================================
    # Upstream: FusedMoEImpl::forward_experts (main orchestrator)

    def forward(
        self,
        hidden_states: torch.Tensor,   # (num_tokens, hidden_size)
        router_logits: torch.Tensor,   # (num_tokens, num_experts)
        w13: torch.Tensor,             # (E_local, inter*2, hidden)
        w2: torch.Tensor,              # (E_local, hidden, inter)
        shared_expert_output: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Full MoE forward pass.

        Tensor lifetime management matches xllm:
          - expand_hidden_states is released after step6
          - act_out is released after step6
          - gemm1_out can be released after step5
        """
        # Step 1: Router
        topk_weights, topk_ids = self.step1_topk_route(router_logits)

        # Step 2: Generate permutation indices
        src_to_dst, dst_to_src, expert_sizes = self.step2_gen_idx(topk_ids)

        # Step 3: Expand input tokens into expert-sorted order
        expand_hidden_states = self.step3_expand_input(
            hidden_states, dst_to_src)

        # Step 4: Group GEMM 1 (gate_proj + up_proj)
        gemm1_out = self.step4_gemm1(
            expand_hidden_states, w13, expert_sizes)

        # Step 5: Activation (gated SiLU)
        act_out = self.step5_activation(gemm1_out)
        del gemm1_out  # release intermediate

        # Step 6: Group GEMM 2 (down_proj)
        gemm2_out = self.step6_gemm2(act_out, w2, expert_sizes)
        del expand_hidden_states, act_out  # release intermediates

        # Step 7: Weighted combine
        final_hidden_states = self.step7_combine(
            gemm2_out, topk_weights, src_to_dst)

        # Add shared expert output if present (Qwen3.5 has shared experts)
        if shared_expert_output is not None:
            final_hidden_states = final_hidden_states + shared_expert_output

        return final_hidden_states
