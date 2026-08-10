"""
EngineX MoE operators — replacements for missing ixformer MoE functions.

From docker log (comp 168):
  ERROR _custom_ops.py:58] module 'ixformer.functions' has no attribute 'vllm_moe_topk_softmax'
  WARNING qwen3_5.py:913] FusedMoE native kernel failed, falling back to pure PyTorch

This fires on EVERY MoE layer (36 per token), 4 workers = 144 error lines per forward pass.

Three operators needed:
  1. moe_topk_softmax — gate logits → softmax → topk expert selection
  2. moe_fused_kernel — the actual expert GEMM dispatch
  3. moe_align_block_size — pad expert assignments to block boundaries
"""

import torch
import torch.nn.functional as F


def moe_topk_softmax_pytorch(
    topk_weights: torch.Tensor,   # [num_tokens, topk] output
    topk_ids: torch.Tensor,       # [num_tokens, topk] output
    token_expert_indices: torch.Tensor,  # [num_tokens, topk] output
    gating_output: torch.Tensor,  # [num_tokens, num_experts] input
) -> None:
    """
    Replacement for ixf_F.vllm_moe_topk_softmax.

    Computes softmax over expert gating logits, selects top-k experts per token.
    This is the router in Qwen3.5's MoE layer (256 experts, topk=8).

    CCCL parallel: maps to tuning_batched_topk.cuh worker_policy pattern —
    each token is a "segment", we find top-k within each segment.
    """
    num_tokens = gating_output.shape[0]
    topk = topk_weights.shape[1]

    # Softmax over experts (dim=-1)
    probs = F.softmax(gating_output, dim=-1)

    # Top-k selection per token
    weights, ids = torch.topk(probs, k=topk, dim=-1)

    # Renormalize weights to sum to 1
    weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-8)

    # Write outputs in-place (matches vllm calling convention)
    topk_weights.copy_(weights)
    topk_ids.copy_(ids)

    # token_expert_indices: flatten assignment for scatter
    # Shape: [num_tokens, topk], value = token_idx * topk + local_expert_rank
    if token_expert_indices.numel() > 0:
        arange = torch.arange(num_tokens, device=gating_output.device)
        token_expert_indices.copy_(
            arange.unsqueeze(1) * topk +
            torch.arange(topk, device=gating_output.device).unsqueeze(0)
        )


def moe_fused_kernel_pytorch(
    hidden_states: torch.Tensor,     # [num_tokens, hidden_dim]
    w1: torch.Tensor,                # [num_experts, hidden_dim, intermediate_dim]
    w2: torch.Tensor,                # [num_experts, intermediate_dim, hidden_dim]
    topk_weights: torch.Tensor,      # [num_tokens, topk]
    topk_ids: torch.Tensor,          # [num_tokens, topk]
    inplace: bool = True,
    override_config: dict = None,
    use_fp8_w8a8: bool = False,
    use_int8_w8a16: bool = False,
    w1_scale: torch.Tensor = None,
    w2_scale: torch.Tensor = None,
    a1_scale: torch.Tensor = None,
    a2_scale: torch.Tensor = None,
) -> torch.Tensor:
    """
    Replacement for vllm_invoke_fused_moe_kernel.

    Dispatches tokens to their assigned experts, runs GEMM, combines results.
    This is the hot inner loop — called 36 times per forward pass.

    Sub168 log shows kernel=expert-grouped-wmma, meaning the native kernel
    groups tokens by expert and runs WMMA (tensor core) GEMMs.
    Our fallback loops over experts — correct but slow.

    CCCL parallel: maps to dispatch_segmented_sort + dispatch_reduce pattern.
    """
    num_tokens, hidden_dim = hidden_states.shape
    topk = topk_ids.shape[1]

    # Group tokens by expert
    # For each expert, collect which tokens use it and their weights
    output = torch.zeros_like(hidden_states)
    num_experts = w1.shape[0]

    for expert_idx in range(num_experts):
        # Find tokens assigned to this expert
        mask = (topk_ids == expert_idx)  # [num_tokens, topk]
        if not mask.any():
            continue

        # Get token indices and their weights for this expert
        token_indices, topk_positions = mask.nonzero(as_tuple=True)

        if token_indices.numel() == 0:
            continue

        weights = topk_weights[token_indices, topk_positions]  # [n_assigned]
        expert_input = hidden_states[token_indices]  # [n_assigned, hidden_dim]

        # Expert forward: gate_up → silu → down
        # w1 is [hidden_dim, intermediate_dim*2] (gate + up fused)
        expert_w1 = w1[expert_idx]  # [hidden_dim, intermediate_dim*2]
        expert_w2 = w2[expert_idx]  # [intermediate_dim, hidden_dim]

        # gate_up = input @ w1  → [n_assigned, intermediate_dim*2]
        gate_up = expert_input @ expert_w1
        intermediate_dim = gate_up.shape[-1] // 2
        gate = gate_up[..., :intermediate_dim]
        up = gate_up[..., intermediate_dim:]

        # SiLU(gate) * up
        activated = F.silu(gate) * up

        # down = activated @ w2
        expert_output = activated @ expert_w2  # [n_assigned, hidden_dim]

        # Weighted accumulate
        output.index_add_(
            0, token_indices,
            expert_output * weights.unsqueeze(-1)
        )

    return output


def moe_align_block_size_pytorch(
    topk_ids: torch.Tensor,       # [num_tokens, topk]
    num_experts: int,
    block_size: int,
    sorted_ids: torch.Tensor,     # output
    expert_ids: torch.Tensor,     # output
    num_tokens_post_pad: torch.Tensor,  # output
) -> None:
    """
    Replacement for ixf_F.vllm_moe_align_block_size.

    Pads expert assignments so each expert's token count is a multiple of
    block_size (for efficient GEMM tiling). This is the MoE equivalent of
    CCCL's dispatch_batch_memcpy tile alignment.
    """
    num_tokens = topk_ids.shape[0]
    topk = topk_ids.shape[1]

    # Flatten expert assignments
    flat_ids = topk_ids.flatten()  # [num_tokens * topk]

    # Count tokens per expert
    counts = torch.zeros(num_experts, dtype=torch.int32,
                         device=topk_ids.device)
    for e in range(num_experts):
        counts[e] = (flat_ids == e).sum()

    # Pad counts to block_size multiples
    padded_counts = ((counts + block_size - 1) // block_size) * block_size
    total_padded = padded_counts.sum().item()

    # Sort tokens by expert, pad with dummy tokens
    offsets = torch.zeros(num_experts + 1, dtype=torch.int32,
                          device=topk_ids.device)
    offsets[1:] = torch.cumsum(padded_counts, dim=0)

    # Fill sorted_ids: real tokens first, then padding
    write_pos = torch.zeros(num_experts, dtype=torch.int32,
                            device=topk_ids.device)

    for i in range(num_tokens * topk):
        token_idx = i // topk
        expert = flat_ids[i].item()
        pos = offsets[expert].item() + write_pos[expert].item()
        if pos < sorted_ids.numel():
            sorted_ids[pos] = token_idx
        write_pos[expert] += 1

    # Fill padding positions with 0 (dummy token)
    for e in range(num_experts):
        start = offsets[e].item() + counts[e].item()
        end = offsets[e].item() + padded_counts[e].item()
        if start < sorted_ids.numel() and end <= sorted_ids.numel():
            sorted_ids[start:end] = 0

    # Expert ids: one per block
    idx = 0
    for e in range(num_experts):
        n_blocks = padded_counts[e].item() // block_size
        for b in range(n_blocks):
            if idx < expert_ids.numel():
                expert_ids[idx] = e
                idx += 1

    num_tokens_post_pad.fill_(total_padded)
