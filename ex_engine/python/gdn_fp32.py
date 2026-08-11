"""gdn_fp32.py — FP32-accumulation GatedDeltaNet implementations.

Ported from upstream xllm/core/layers/npu_torch/qwen3_gated_delta_net_base.cpp.
The key fix: all internal computation in fp32, cast back to original dtype at end.
This eliminates the 99.98% NaN problem seen in comp 168 docker logs.

Two implementations:
  - torch_recurrent_gated_delta_rule:  single-step recurrent (for decode)
  - torch_chunk_gated_delta_rule:      chunked (for prefill)
"""

import torch
import torch.nn.functional as F


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2 normalize along dim."""
    return F.normalize(x, p=2, dim=dim, eps=eps)


def torch_recurrent_gated_delta_rule(
    query: torch.Tensor,     # [B, H, L, K]
    key: torch.Tensor,       # [B, H, L, K]
    value: torch.Tensor,     # [B, H, L, V]
    g: torch.Tensor,         # [B, H, L]  (gate / log-decay)
    beta: torch.Tensor,      # [B, H, L]
    initial_state=None,      # [B, H, K, V] or None
    use_qk_l2norm: bool = True,
):
    """Single-step recurrent GDN — decode path.

    Port of: qwen3_gated_delta_net_base.cpp::torch_recurrent_gated_delta_rule()
    Key difference from our previous Python: ALL computation in fp32.
    """
    initial_dtype = query.dtype

    if use_qk_l2norm:
        query = _l2norm(query, -1)
        key = _l2norm(key, -1)

    # Upstream: to_float32_and_transpose → [B, H, L, D]
    # Our tensors are already [B, H, L, D] from the caller, so just cast
    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()

    B, H, L, K = query.shape
    V = value.size(-1)

    scale = (1.0 / (K ** 0.5))
    query = query * scale

    if initial_state is None:
        state = torch.zeros(B, H, K, V, dtype=torch.float32,
                            device=query.device)
    else:
        state = initial_state.to(dtype=torch.float32, device=query.device)

    outputs = torch.zeros(B, H, L, V, dtype=torch.float32,
                          device=query.device)

    for i in range(L):
        q_t = query[:, :, i]          # [B, H, K]
        k_t = key[:, :, i]            # [B, H, K]
        v_t = value[:, :, i]          # [B, H, V]
        g_t = g[:, :, i].exp()        # [B, H]
        beta_t = beta[:, :, i]        # [B, H]

        # Decay state
        state = state * g_t.unsqueeze(-1).unsqueeze(-1)

        # Delta update: v - sum(state * k, dim=-2)
        kv_mem = (state * k_t.unsqueeze(-1)).sum(-2)   # [B, H, V]
        delta = (v_t - kv_mem) * beta_t.unsqueeze(-1)  # [B, H, V]

        # Write to state
        state = state + k_t.unsqueeze(-1) * delta.unsqueeze(-2)

        # Query readout
        outputs[:, :, i] = (state * q_t.unsqueeze(-1)).sum(-2)

    outputs = outputs.to(initial_dtype)
    return outputs, state


def torch_chunk_gated_delta_rule(
    query: torch.Tensor,     # [B, H, L, K]
    key: torch.Tensor,       # [B, H, L, K]
    value: torch.Tensor,     # [B, H, L, V]
    g: torch.Tensor,         # [B, H, L]
    beta: torch.Tensor,      # [B, H, L]
    chunk_size: int = 64,
    initial_state=None,
    output_final_state: bool = True,
    use_qk_l2norm: bool = True,
):
    """Chunked GDN — prefill path.

    Port of: qwen3_gated_delta_net_base.cpp::torch_chunk_gated_delta_rule()
    ALL internal computation in fp32 to prevent NaN.
    """
    initial_dtype = query.dtype

    if use_qk_l2norm:
        query = _l2norm(query, -1)
        key = _l2norm(key, -1)

    # Cast to fp32
    query = query.float()
    key = key.float()
    value = value.float()
    beta = beta.float()
    g = g.float()

    B, H, L, K = query.shape
    V = value.size(-1)

    # Pad to multiple of chunk_size
    pad = (chunk_size - L % chunk_size) % chunk_size
    if pad > 0:
        query = F.pad(query, (0, 0, 0, pad))
        key = F.pad(key, (0, 0, 0, pad))
        value = F.pad(value, (0, 0, 0, pad))
        beta = F.pad(beta, (0, pad))
        g = F.pad(g, (0, pad))

    total_len = L + pad
    scale = 1.0 / (K ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)

    # Reshape to chunks: [B, H, num_chunks, chunk_size, D]
    num_chunks = total_len // chunk_size
    query = query.reshape(B, H, num_chunks, chunk_size, K)
    key = key.reshape(B, H, num_chunks, chunk_size, K)
    value_c = value.reshape(B, H, num_chunks, chunk_size, V)
    k_beta = k_beta.reshape(B, H, num_chunks, chunk_size, K)
    v_beta = v_beta.reshape(B, H, num_chunks, chunk_size, V)
    g = g.reshape(B, H, num_chunks, chunk_size)

    # Cumulative sum of g within each chunk
    g = g.cumsum(-1)

    # Decay mask within chunk
    g_diff = g.unsqueeze(-1) - g.unsqueeze(-2)  # [B,H,C,cs,cs]
    decay_mask = g_diff.tril().exp()
    decay_mask = decay_mask.tril()

    # Intra-chunk attention correction (Woodbury-like)
    mask_upper = torch.triu(torch.ones(chunk_size, chunk_size,
                                       dtype=torch.bool,
                                       device=query.device), 0)
    attn = -(torch.matmul(k_beta, key.transpose(-1, -2)) * decay_mask)
    attn = attn.masked_fill(mask_upper, 0.0)

    # Sequential correction within chunk (upstream lines 174-192)
    for i in range(1, chunk_size):
        row = attn[..., i:i+1, :i].squeeze(-2).clone()
        sub = attn[..., :i, :i].clone()
        row_sub = (row.unsqueeze(-1) * sub).sum(-2)
        attn[..., i:i+1, :i] = (row + row_sub).unsqueeze(-2)

    eye = torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    attn = attn + eye

    # Corrected value and k_cumdecay
    value_corr = torch.matmul(attn, v_beta)
    k_cumdecay = torch.matmul(attn, k_beta * g.exp().unsqueeze(-1))

    # Initialize state
    if initial_state is None:
        state = torch.zeros(B, H, K, V, dtype=torch.float32,
                            device=query.device)
    else:
        state = initial_state.to(dtype=torch.float32, device=query.device)

    out = torch.zeros_like(value_corr)

    mask_strict_upper = torch.triu(torch.ones(chunk_size, chunk_size,
                                              dtype=torch.bool,
                                              device=query.device), 1)

    for i in range(num_chunks):
        q_i = query[:, :, i]           # [B,H,cs,K]
        k_i = key[:, :, i]
        v_i = value_corr[:, :, i]      # [B,H,cs,V]

        attn_i = (torch.matmul(q_i, k_i.transpose(-1, -2))
                  * decay_mask[:, :, i])
        attn_i = attn_i.masked_fill_(mask_strict_upper, 0.0)

        # Cross-chunk: state contribution
        v_prime = torch.matmul(k_cumdecay[:, :, i], state)  # [B,H,cs,V]
        v_new = v_i - v_prime

        # Inter-chunk attention
        g_i = g[:, :, i]  # [B,H,cs]
        attn_inter = torch.matmul(
            q_i * g_i.unsqueeze(-1).exp(), state)  # [B,H,cs,V]

        out[:, :, i] = attn_inter + torch.matmul(attn_i, v_new)

        # Update state
        g_last = g_i[..., -1:]  # [B,H,1]
        g_exp_term = (g_last - g_i).exp().unsqueeze(-1)  # [B,H,cs,1]
        k_g_exp = (k_i * g_exp_term).transpose(-1, -2)   # [B,H,K,cs]
        state = (state * g_last.unsqueeze(-1).exp()
                 + torch.matmul(k_g_exp, v_new))

    # Reshape back, trim padding, cast back
    out = out.reshape(B, H, total_len, V)
    out = out[:, :, :L, :]
    out = out.to(initial_dtype)

    return out, state
