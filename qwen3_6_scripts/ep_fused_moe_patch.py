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


def _is_ep_requested():
    """Check if EP is requested via environment variable.

    This is safe to call at any time (no dist dependency).
    """
    import os
    return bool(int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")))


def _get_ep_runtime_config():
    """Get EP size and rank at runtime when dist is initialized.

    Must only be called from within worker processes (during __init__,
    weight_loader, or forward) where torch.distributed is guaranteed
    to be initialized.

    Returns (ep_size, ep_rank).
    """
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    # Fallback — should not happen during normal model loading
    logger.warning("[PR #2269] dist not initialized when querying EP config, "
                   "falling back to ep_size=1")
    return 1, 0


def patch_fused_moe_for_ep():
    """Monkey-patch FusedMoE.__init__, weight_loader, and forward to support EP.

    Call this during model loading (e.g. in qwen3_5.py module init or
    patch_ops.sh) BEFORE any FusedMoE layers are constructed.

    Only checks the env var at patch time. Actual ep_size/ep_rank are
    queried at runtime (inside __init__, weight_loader, forward) when
    torch.distributed is guaranteed to be initialized.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if not _is_ep_requested():
        logger.info("[PR #2269] EP not enabled (VLLM_ENABLE_EXPERT_PARALLEL!=1), "
                    "skipping FusedMoE EP patch")
        return

    logger.info("[PR #2269] EP requested via env var, installing FusedMoE monkey-patches "
                "(ep_size/ep_rank will be resolved at runtime)")

    _orig_init = FusedMoE.__init__

    def _ep_init(self, num_experts, top_k, hidden_size, intermediate_size,
                 params_dtype=None, reduce_results=False, renormalize=True,
                 use_grouped_topk=False, num_expert_group=None,
                 topk_group=None, quant_config=None, tp_size=None,
                 prefix="", custom_routing_function=None):
        """EP-aware __init__: allocate only local experts, full intermediate."""

        # Query ep_size/ep_rank NOW — dist is initialized in worker processes
        ep_size, ep_rank = _get_ep_runtime_config()

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
        """EP-aware weight_loader: skip experts not owned by this rank.

        Critical fix: the original weight_loader calls
        get_tensor_model_parallel_rank() internally to compute tp_rank,
        then uses it to narrow() the loaded weight for TP sharding.
        Under EP we pass tp_size=1 (no TP split on MoE), so weights are
        full-sized, but tp_rank still returns the real rank (0-3).
        rank >= 1 causes narrow(dim, 512*rank, 512) to exceed dim size 512.

        Solution: temporarily monkey-patch get_tensor_model_parallel_rank
        to return 0 during EP weight loading, so narrow() always starts
        at offset 0 and loads the full (unsplit) weight.
        """
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

        # Monkey-patch tp rank to 0 during load so narrow() doesn't OOB.
        # The weight is already full-sized (tp_size=1), no split needed.
        import vllm.model_executor.layers.fused_moe.layer as _fused_moe_mod
        _real_get_tp_rank = _fused_moe_mod.get_tensor_model_parallel_rank
        _fused_moe_mod.get_tensor_model_parallel_rank = lambda: 0
        try:
            _orig_weight_loader(self, param, loaded_weight, weight_name,
                                shard_id, local_expert_id)
        finally:
            _fused_moe_mod.get_tensor_model_parallel_rank = _real_get_tp_rank

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
        # Use the same kernel priority as _pure_pytorch_experts in qwen3_5.py:
        # corex_moe_topk_softmax → xllm_moe → PyTorch fallback
        try:
            from vllm import corex_moe_topk_softmax as _corex_topk
        except ImportError:
            _corex_topk = None
        try:
            from vllm import xllm_moe as _xllm
        except ImportError:
            try:
                import xllm_moe as _xllm
            except ImportError:
                _xllm = None

        if _xllm is not None:
            topk_weights, topk_ids = _xllm.moe_fused_topk(
                router_logits, self.top_k, True, None, "softmax")
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(hidden_states.dtype)
        elif _corex_topk is not None:
            topk_weights, topk_ids = _corex_topk.moe_topk_softmax(
                router_logits.float(), self.top_k, True)
            topk_ids = topk_ids.to(torch.int64)
            topk_weights = topk_weights.to(hidden_states.dtype)
        else:
            topk_logits, topk_ids = torch.topk(
                router_logits.float(), self.top_k, dim=-1)
            topk_weights = torch.softmax(topk_logits, dim=-1)
            topk_weights = topk_weights.to(hidden_states.dtype)

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
        "[PR #2269] FusedMoE patched for EP. "
        "ep_size/ep_rank will be resolved per-layer at runtime.")