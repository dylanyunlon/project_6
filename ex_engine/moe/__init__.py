"""
ex_engine.moe — MoE expert computation for BI-V100

Ported from:
  upstream_ref/ds_vllm/vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py
  upstream_ref/ds_vllm/vllm/model_executor/layers/fused_moe/activation.py
"""

from ex_engine.moe.naive_batched_experts import naive_batched_moe_forward
from ex_engine.moe.activation import MoEActivation, apply_moe_activation
