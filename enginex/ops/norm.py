"""
EngineX norm operators.

RMSNorm is called 128 times per forward pass (pre-attn + post-attn × 64 layers).
fused_add_rms_norm fuses residual addition with normalization.

ixformer provides both natively. Fallbacks for environments without ixformer.
"""

import torch


def rms_norm_pytorch(
    input: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """RMSNorm: output = (input / rms(input)) * weight"""
    variance = input.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = input * torch.rsqrt(variance + epsilon)
    output.copy_(normed * weight)


def fused_add_rms_norm_pytorch(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float = 1e-6,
) -> None:
    """Fused: input = RMSNorm(input + residual); residual = input + residual"""
    # In-place: residual += input, then normalize
    residual.add_(input)
    variance = residual.to(torch.float32).pow(2).mean(-1, keepdim=True)
    normed = residual * torch.rsqrt(variance + epsilon)
    input.copy_(normed * weight)
