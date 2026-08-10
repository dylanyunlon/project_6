"""
EngineX sampling operators.

rotary_embedding: applies RoPE (Rotary Position Embedding) to Q and K.
Called once per attention layer per forward pass.

ixformer provides vllm_rotary_embedding_neox natively.
"""

import torch


def rotary_embedding_pytorch(
    positions: torch.Tensor,     # [num_tokens]
    query: torch.Tensor,         # [num_tokens, num_heads * head_size]
    key: torch.Tensor,           # [num_tokens, num_kv_heads * head_size]
    head_size: int,
    cos_sin_cache: torch.Tensor, # [max_position, rotary_dim]
    is_neox: bool = True,
) -> None:
    """Apply rotary position embedding in-place on query and key."""
    rotary_dim = cos_sin_cache.shape[1]
    half_rot = rotary_dim // 2

    # Gather cos/sin for each token's position
    cos = cos_sin_cache[positions, :half_rot]  # [num_tokens, half_rot]
    sin = cos_sin_cache[positions, half_rot:]   # [num_tokens, half_rot]

    def _apply_rotary(x, cos, sin, head_size, rotary_dim):
        """Apply rotary embedding to a reshaped tensor."""
        num_tokens = x.shape[0]
        num_heads = x.shape[1] // head_size
        x_view = x.view(num_tokens, num_heads, head_size)

        rot = x_view[..., :rotary_dim]
        pass_through = x_view[..., rotary_dim:]

        x1 = rot[..., :half_rot]
        x2 = rot[..., half_rot:]

        cos_exp = cos.unsqueeze(1)  # [num_tokens, 1, half_rot]
        sin_exp = sin.unsqueeze(1)

        rot_out = torch.cat([
            x1 * cos_exp - x2 * sin_exp,
            x2 * cos_exp + x1 * sin_exp,
        ], dim=-1)

        x_view[..., :rotary_dim] = rot_out
        x.copy_(x_view.reshape(num_tokens, -1))

    _apply_rotary(query, cos, sin, head_size, rotary_dim)
    _apply_rotary(key, cos, sin, head_size, rotary_dim)
