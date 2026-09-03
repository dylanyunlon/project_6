"""EP-aware FusedMoE override for vllm 0.6.3.

PR #2269 Sub-task 2: Expert Parallelism weight sharding.

Problem:
  TP=2 on 4× BI-V100 (32 GiB each) OOMs during model loading because
  FusedMoE.create_weights allocates ALL experts on every card, splitting
  only the intermediate_size dimension. For Qwen3.6-35B-A3B (256 experts),
  MoE weights alone consume ~22.5 GiB/card under TP=2.

Solution:
  When enable_expert_parallel=True, each card holds only
  num_experts/ep_size experts with FULL intermediate_size (no TP split
  on MoE). Forward uses all-to-all to dispatch tokens to the correct
  expert-holding rank, compute locally, and combine results back.

  EP=4: each card holds 64/256 experts → MoE weights ~5.6 GiB/card
  (savings: 16.9 GiB/card vs TP=2)

Memory layout comparison (per card, 30 MoE layers, fp16):
  TP=2:  256 experts × (2×256×2048 + 2048×256) × 2B × 30 = 22.5 GiB
  EP=4:   64 experts × (2×512×2048 + 2048×512) × 2B × 30 =  5.6 GiB

xllm equivalent: core/layers/cuda/fused_moe.cpp
  ep_size_ = parallel_args.ep_size();
  num_experts_per_rank_ = num_experts / ep_size;
  start_expert_id_ = ep_rank * num_experts_per_rank_;
  output = cutlass_fused_moe(..., ep_size_, ep_rank_, ...);
  output = parallel_state::reduce(output, ep_pg_);
"""

import torch
import torch.distributed as dist
from typing import Callable, List, Optional, Tuple

from vllm.config import ParallelConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def _get_ep_config():
    """Read EP configuration from ParallelConfig.

    Returns (ep_enabled, ep_size, ep_rank, ep_group).
    Safe to call during model init — returns disabled if ParallelConfig
    doesn't have EP fields (backward compat with unpatched vllm).
    """
    try:
        from vllm.config import ParallelConfig as _PC
        # ParallelConfig is a per-worker singleton-ish object.
        # We check if the fields exist (set by our config.py patch).
        # During weight loading, the global TP group is already initialized.
        from vllm.distributed import (
            get_tensor_model_parallel_world_size,
            get_tensor_model_parallel_rank,
        )
        # EP config is on the parallel_config instance that was passed
        # through the worker. We can't easily get it here without plumbing,
        # so we use env vars (set in arg_utils.py) as the source of truth.
        import os
        ep_enabled = bool(int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")))
        if not ep_enabled:
            return False, 1, 0, None

        # When EP is enabled, ep_size = world_size (all ranks participate).
        # ep_rank = global rank (since EP group = all ranks).
        if dist.is_initialized():
            ep_size = dist.get_world_size()
            ep_rank = dist.get_rank()
        else:
            ep_size = 1
            ep_rank = 0
            ep_enabled = False

        return ep_enabled, ep_size, ep_rank, None
    except Exception:
        return False, 1, 0, None


def patch_fused_moe_for_ep():
    """Monkey-patch FusedMoE.__init__, weight_loader, and forward to support EP.

    Call this during model loading (e.g. in qwen3_5.py module init or
    patch_ops.sh) BEFORE any FusedMoE layers are constructed.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    ep_enabled, ep_size, ep_rank, _ = _get_ep_config()
    if not ep_enabled or ep_size <= 1:
        logger.info("[PR #2269] EP not enabled or ep_size<=1, skipping FusedMoE EP patch")
        return

    num_experts_per_rank = None  # set per-layer in __init__

    _orig_init = FusedMoE.__init__

    def _ep_init(self, num_experts, top_k, hidden_size, intermediate_size,
                 params_dtype=None, reduce_results=False, renormalize=True,
                 use_grouped_topk=False, num_expert_group=None,
                 topk_group=None, quant_config=None, tp_size=None,
                 prefix="", custom_routing_function=None):
        """EP-aware __init__: allocate only local experts, full intermediate."""

        # Store global expert info before modifying
        self._ep_enabled = True
        self._ep_size = ep_size
        self._ep_rank = ep_rank
        self._global_num_experts = num_experts
        assert num_experts % ep_size == 0, (
            f"num_experts ({num_experts}) must be divisible by "
            f"ep_size ({ep_size})")
        self._num_experts_per_rank = num_experts // ep_size
        self._start_expert_id = ep_rank * self._num_experts_per_rank

        logger.info(
            "[PR #2269] FusedMoE EP init: global_experts=%d, ep_size=%d, "
            "ep_rank=%d, local_experts=%d (ids %d..%d), "
            "intermediate_size=%d (full, no TP split)",
            num_experts, ep_size, ep_rank,
            self._num_experts_per_rank,
            self._start_expert_id,
            self._start_expert_id + self._num_experts_per_rank - 1,
            intermediate_size)

        # Call original __init__ with:
        # - num_experts = local experts only (saves memory)
        # - tp_size = 1 (no TP split on intermediate — EP replaces TP for MoE)
        # - reduce_results = False (we do EP reduce ourselves)
        _orig_init(
            self,
            num_experts=self._num_experts_per_rank,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=False,  # EP does its own reduce
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=1,  # no TP split — EP splits by expert count instead
            prefix=prefix,
            custom_routing_function=custom_routing_function,
        )

        # Override num_experts to global for routing (select_experts sees all)
        self.num_experts = num_experts
        # But weights are sized for local experts only
        self._reduce_results_ep = reduce_results

    _orig_weight_loader = FusedMoE.weight_loader

    def _ep_weight_loader(self, param, loaded_weight, weight_name,
                          shard_id, expert_id):
        """EP-aware weight_loader: skip experts not owned by this rank."""
        if not getattr(self, '_ep_enabled', False):
            return _orig_weight_loader(self, param, loaded_weight,
                                       weight_name, shard_id, expert_id)

        start = self._start_expert_id
        end = start + self._num_experts_per_rank

        # Skip experts not owned by this EP rank
        if expert_id < start or expert_id >= end:
            return

        # Remap global expert_id to local index
        local_expert_id = expert_id - start

        # Call original weight_loader with local expert_id.
        # The param tensor is sized for local experts only.
        _orig_weight_loader(self, param, loaded_weight, weight_name,
                            shard_id, local_expert_id)

    _orig_forward = FusedMoE.forward

    def _ep_forward(self, hidden_states, router_logits):
        """EP-aware forward: all-to-all dispatch, local compute, all-to-all combine.

        Flow:
        1. Route all tokens globally (all ranks see all expert logits)
        2. All-to-all dispatch: send each token to the rank that owns its expert
        3. Local expert computation on owned experts only
        4. All-to-all combine: gather results back to original token positions
        """
        if not getattr(self, '_ep_enabled', False):
            return _orig_forward(self, hidden_states, router_logits)

        # Step 1: Global routing — every rank computes topk over ALL experts
        topk_weights, topk_ids = FusedMoE.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
            use_grouped_topk=self.use_grouped_topk,
            top_k=self.top_k,
            renormalize=self.renormalize,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
        )

        ep_size = self._ep_size
        ep_rank = self._ep_rank
        start_expert = self._start_expert_id
        end_expert = start_expert + self._num_experts_per_rank
        T, H = hidden_states.shape
        K = self.top_k
        device = hidden_states.device

        # Step 2: Determine which tokens go to which EP rank.
        # expert_to_rank: expert_id → ep_rank
        # For each (token, top_k_slot), find the destination rank.
        dest_ranks = topk_ids.div(self._num_experts_per_rank, rounding_mode='trunc')  # (T, K)

        # Count tokens to send to each rank
        send_counts = torch.zeros(ep_size, dtype=torch.int64, device=device)
        for r in range(ep_size):
            send_counts[r] = (dest_ranks == r).sum()

        # Gather all ranks' send counts so we know receive counts
        all_counts = torch.zeros(ep_size, ep_size, dtype=torch.int64, device=device)
        dist.all_gather_into_tensor(
            all_counts.view(-1),
            send_counts,
        )
        recv_counts = all_counts[:, ep_rank]  # what each rank sends to us

        total_send = send_counts.sum().item()
        total_recv = recv_counts.sum().item()

        # Build dispatch buffers: for each (token, k) pair routed to each rank,
        # pack (hidden_state, topk_weight, local_expert_id, original_position)
        # Sort by destination rank for contiguous all-to-all.

        # Flatten topk dimension: each (token, k_slot) is a "work item"
        flat_ids = topk_ids.view(-1)           # (T*K,)
        flat_weights = topk_weights.view(-1)   # (T*K,)
        flat_dest = dest_ranks.view(-1)        # (T*K,)
        flat_token_idx = torch.arange(T, device=device).unsqueeze(1).expand(T, K).reshape(-1)  # (T*K,)

        # Sort by destination rank for all-to-all
        sort_idx = flat_dest.argsort()
        sorted_hidden = hidden_states[flat_token_idx[sort_idx]]   # (T*K, H)
        sorted_weights = flat_weights[sort_idx]                   # (T*K,)
        sorted_ids = flat_ids[sort_idx]                           # (T*K,) global expert ids
        sorted_token_idx = flat_token_idx[sort_idx]               # (T*K,) for scatter-back

        # All-to-all send/recv counts (in elements, each element = one work item)
        send_splits = send_counts.tolist()
        recv_splits = recv_counts.tolist()

        # All-to-all: exchange hidden states
        recv_hidden = torch.empty(total_recv, H, dtype=hidden_states.dtype, device=device)
        dist.all_to_all_single(recv_hidden, sorted_hidden,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits)

        # All-to-all: exchange expert ids
        recv_ids = torch.empty(total_recv, dtype=sorted_ids.dtype, device=device)
        dist.all_to_all_single(recv_ids, sorted_ids,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits)

        # All-to-all: exchange topk weights
        recv_weights = torch.empty(total_recv, dtype=sorted_weights.dtype, device=device)
        dist.all_to_all_single(recv_weights, sorted_weights,
                               output_split_sizes=recv_splits,
                               input_split_sizes=send_splits)

        # Step 3: Local expert computation on received tokens
        # Remap global expert ids to local
        local_ids = recv_ids - start_expert  # now 0..num_experts_per_rank-1

        if total_recv > 0:
            # Compute using the existing fused_experts kernel (or Python fallback)
            from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
            # Build per-token topk format expected by fused_experts:
            # It expects (num_tokens, top_k) but here each recv item is
            # a single (token, expert) pair. We reshape to (total_recv, 1).
            local_topk_ids = local_ids.unsqueeze(1)        # (total_recv, 1)
            local_topk_weights = recv_weights.unsqueeze(1)  # (total_recv, 1)

            local_output = fused_experts(
                hidden_states=recv_hidden,
                w1=self.w13_weight,
                w2=self.w2_weight,
                topk_weights=local_topk_weights,
                topk_ids=local_topk_ids.to(torch.int32),
                inplace=False,
            )
        else:
            local_output = torch.empty(0, H, dtype=hidden_states.dtype, device=device)

        # Step 4: All-to-all combine — send results back
        # Reverse the all-to-all: recv_splits becomes send, send_splits becomes recv
        combine_output = torch.empty(total_send, H, dtype=hidden_states.dtype, device=device)
        dist.all_to_all_single(combine_output, local_output,
                               output_split_sizes=send_splits,
                               input_split_sizes=recv_splits)

        # Unsort and scatter-add back to original token positions
        # combine_output is in the same order as sorted_* (sorted by dest rank)
        # We need to unsort and accumulate weighted results per token.
        final_output = torch.zeros(T, H, dtype=hidden_states.dtype, device=device)
        # Unsort: combine_output[i] corresponds to sorted_token_idx[i]
        unsorted_output = torch.zeros_like(combine_output)
        unsorted_output[sort_idx] = combine_output

        # Reshape back to (T, K, H) and sum over K dimension
        # Each (token, k_slot) pair has already been weighted by topk_weight
        # inside fused_experts. We just need to sum over k.
        # Actually fused_experts with top_k=1 per item already applies the weight.
        # scatter_add by token index:
        token_indices = flat_token_idx.unsqueeze(1).expand(-1, H)  # (T*K, H)
        final_output.scatter_add_(0, token_indices, unsorted_output)

        # EP reduce (if the caller requested reduce_results, e.g. for shared experts)
        if self._reduce_results_ep:
            # No TP reduce needed (tp_size=1 for EP), but if the model
            # expects an all-reduce for the MoE output (e.g. to combine
            # with shared experts), we do it here over the EP group.
            pass  # EP output is already complete — each token got results from its experts

        return final_output

    # Apply patches
    FusedMoE.__init__ = _ep_init
    FusedMoE.weight_loader = _ep_weight_loader
    FusedMoE.forward = _ep_forward

    logger.info(
        "[PR #2269] FusedMoE patched for EP: ep_size=%d, ep_rank=%d. "
        "Each card holds %d/%d-th of experts with full intermediate_size.",
        ep_size, ep_rank, 1, ep_size)