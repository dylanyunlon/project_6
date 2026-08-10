"""
EngineX activation operators.

These map to CCCL's dispatch_transform pattern — element-wise kernels
that fuse activation + multiply in a single pass.

ixformer provides these natively (confirmed working in hardware probe).
PyTorch fallbacks here for completeness.

CCCL tuning: tuning_transform.cuh bytes_in_flight = 64KB on BI-V100
             (56 GB/s per-SM × 1100ns latency, 16 SMs)
"""

import torch
import torch.nn.functional as F


def silu_and_mul_pytorch(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fused SiLU(x[..., :d]) * x[..., d:]"""
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    out.copy_(F.silu(gate) * up)


def gelu_and_mul_pytorch(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fused GELU(x[..., :d]) * x[..., d:]"""
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    out.copy_(F.gelu(gate) * up)


def gelu_tanh_and_mul_pytorch(x: torch.Tensor, out: torch.Tensor) -> None:
    """Fused GELU_tanh(x[..., :d]) * x[..., d:]"""
    d = x.shape[-1] // 2
    gate = x[..., :d]
    up = x[..., d:]
    out.copy_(F.gelu(gate, approximate='tanh') * up)
