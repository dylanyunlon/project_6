"""EP-aware FusedMoE override for vllm 0.6.3.

PR #2269 Sub-task 2: Expert Parallelism weight sharding.

Design (Google Option B, defer_all_reduce pattern):
  Each card holds num_experts/ep_size experts, TP-sharded as usual.
  Forward uses the SAME _pure_pytorch_experts() path as non-EP mode.
  The only EP-specific logic is:
    1. __init__: allocate only local experts (saves memory)
    2. weight_loader: skip non-local experts, remap expert_id
    3. forward: mask non-local experts in topk_ids, all_reduce at end

  This eliminates the need for separate EP forward/compute code.
  The all_reduce sums partial results (each rank's local experts)
  into the complete MoE output, same as TP all_reduce sums partial
  intermediate results.

Reference:
  Google tpu-inference: tpu_inference/layers/common/fused_moe_gmm.py
    - fused_moe_func() with defer_all_reduce pattern
    - PR #2577 (Attention DP + EP sharding)
"""

import os
import torch
import torch.distributed as dist

from vllm.logger import init_logger

logger = init_logger(__name__)


def _is_ep_requested():
    return bool(int(os.environ.get("VLLM_ENABLE_EXPERT_PARALLEL", "0")))


def _get_ep_runtime_config():
    if dist.is_initialized():
        return dist.get_world_size(), dist.get_rank()
    return 1, 0


def patch_fused_moe_for_ep():
    """Monkey-patch FusedMoE to support Expert Parallelism.

    Only patches __init__ and weight_loader. Forward is NOT patched —
    _pure_pytorch_experts() in qwen3_5.py handles everything, with EP
    awareness via self._ep_enabled / self._start_expert_id attributes.
    """
    from vllm.model_executor.layers.fused_moe.layer import FusedMoE

    if not _is_ep_requested():
        logger.info("[PR #2269] EP not enabled (VLLM_ENABLE_EXPERT_PARALLEL!=1), "
                    "skipping FusedMoE EP patch")
        return

    logger.info("[PR #2269] EP requested, installing FusedMoE __init__ + "
                "weight_loader patches")

    _orig_init = FusedMoE.__init__

    def _ep_init(self, num_experts, top_k, hidden_size, intermediate_size,
                 params_dtype=None, reduce_results=False, renormalize=True,
                 use_grouped_topk=False, num_expert_group=None,
                 topk_group=None, quant_config=None, tp_size=None,
                 prefix="", custom_routing_function=None):
        """EP-aware __init__: allocate only local experts, keep TP sharding."""

        ep_size, ep_rank = _get_ep_runtime_config()

        num_experts_per_rank = num_experts // ep_size
        start_expert_id = ep_rank * num_experts_per_rank
        assert num_experts % ep_size == 0

        logger.info(
            "[PR #2269] FusedMoE EP init: global=%d ep_size=%d ep_rank=%d "
            "local=%d (ids %d..%d) tp_size=%s",
            num_experts, ep_size, ep_rank,
            num_experts_per_rank,
            start_expert_id,
            start_expert_id + num_experts_per_rank - 1,
            tp_size)

        # Call original __init__ with LOCAL expert count but SAME tp_size.
        # NOTE: _orig_init calls super().__init__() which wipes self.__dict__,
        # so ALL EP attributes must be set AFTER this call.
        _orig_init(
            self,
            num_experts=num_experts_per_rank,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=False,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=1,  # EP replaces TP for MoE: full intermediate per expert
            # Why tp_size=1: With EP+TP on the same group, a single all_reduce
            # sums across all ranks. If weights are TP-sharded (I/tp), each
            # expert only exists on 1 rank with 1/tp of intermediate — the
            # all_reduce adds contributions from DIFFERENT experts (EP) but
            # does NOT sum TP shards of the SAME expert. Result: 1/tp too small.
            # With tp_size=1: each rank holds full intermediate for its local
            # experts. all_reduce sums 4 ranks × 64 experts = 256 experts,
            # each with full intermediate. Correct.
            prefix=prefix,
            custom_routing_function=custom_routing_function,
        )

        # Override num_experts back to global for routing
        self.num_experts = num_experts

        # Set EP attributes AFTER _orig_init. The call above invokes
        # super().__init__() which resets self.__dict__, so anything
        # set before it gets wiped.
        self._ep_enabled = True
        self._ep_size = ep_size
        self._ep_rank = ep_rank
        self._global_num_experts = num_experts
        self._num_experts_per_rank = num_experts_per_rank
        self._start_expert_id = start_expert_id

    _orig_weight_loader = FusedMoE.weight_loader

    def _ep_weight_loader(self, param, loaded_weight, weight_name,
                          shard_id, expert_id):
        """EP-aware weight_loader: skip non-local experts, remap id.
        
        Since we pass tp_size=1 to FusedMoE (each rank holds full intermediate),
        we must also force tp_rank=0 during loading. Otherwise ranks 1/2/3
        call narrow(dim, shard_size*tp_rank, shard_size) which goes out of bounds
        because the weight is already full-size (no TP split to index into).
        """
        if not getattr(self, '_ep_enabled', False):
            return _orig_weight_loader(self, param, loaded_weight,
                                       weight_name, shard_id, expert_id)

        start = self._start_expert_id
        end = start + self._num_experts_per_rank

        if expert_id < start or expert_id >= end:
            return

        local_expert_id = expert_id - start
        # Temporarily force tp_rank=0 so narrow() doesn't go OOB.
        # With tp_size=1 the weight is full-size, so every rank loads
        # the same full slice (offset 0).
        import vllm.distributed as _dist
        _real_tp_rank = _dist.get_tensor_model_parallel_rank
        _dist.get_tensor_model_parallel_rank = lambda: 0
        try:
            _orig_weight_loader(self, param, loaded_weight, weight_name,
                                shard_id, local_expert_id)
        finally:
            _dist.get_tensor_model_parallel_rank = _real_tp_rank

    FusedMoE.__init__ = _ep_init
    FusedMoE.weight_loader = _ep_weight_loader
    # NOTE: FusedMoE.forward is NOT patched. qwen3_5.py forward() handles EP.

    logger.info("[PR #2269] FusedMoE patches installed (__init__, weight_loader)")