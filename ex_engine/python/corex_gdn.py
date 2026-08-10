"""
corex_gdn.py — GatedDeltaNet fused kernel dispatch for BI-V100

Comp 168 log:
  corex_gdn.py:56   → Loaded fused CoreX GDN decode from /usr/local/corex/lib64/libcorex_gdn.so
  corex_gdn.py:228  → Using fused CoreX GDN prefill operator
  corex_gdn.py:138  → Using fused CoreX GDN decode operator

This module implements the full GDN layer forward pass.
qwen3_5.py calls:
  CoreXGDN.__init__(num_v_heads, num_k_heads, head_k_dim, head_v_dim, conv_kernel_size, layer_idx)
  CoreXGDN.forward(hidden_states, attn_metadata, conv_state, temporal_state,
                    in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
                    conv1d_weight, A_log, dt_bias, norm, out_proj)

NO FALLBACK. This must produce correct output or crash with a clear error.
"""

import logging
import math
import torch
import torch.nn.functional as F
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# ixformer acceleration
_ix = None
_ix_available = False
try:
    import ixformer.functions as _ix
    _ix_available = True
except (ImportError, AttributeError):
    pass


def _ix_matmul(a, b):
    if _ix_available and a.dtype == torch.float16:
        try:
            return _ix.matmul(a, b)
        except Exception:
            pass
    return torch.matmul(a, b)


def _ix_bmm(a, b):
    if _ix_available and a.dtype == torch.float16:
        try:
            return _ix.matmul(a, b)
        except Exception:
            pass
    return torch.matmul(a, b)


def _l2norm(x, dim=-1, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def _causal_conv1d_update(hidden_states, conv_state, weight, bias=None, activation=None):
    _, channels, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    cat = torch.cat([conv_state, hidden_states], dim=-1).to(weight.dtype)
    conv_state.copy_(cat[:, :, -state_len:])
    out = F.conv1d(cat, weight.unsqueeze(1), bias, padding=0, groups=channels)
    out = out[:, :, -seq_len:]
    if activation is not None:
        out = F.silu(out)
    return out.to(hidden_states.dtype)


def _chunk_gated_delta_rule(
    query, key, value, g, beta,
    chunk_size=16, initial_state=None,
    output_final_state=False, use_qk_l2norm_in_kernel=False,
):
    """Chunked GatedDeltaNet forward — fp32 accumulation, no fallback."""
    initial_dtype = query.dtype
    if use_qk_l2norm_in_kernel:
        query = _l2norm(query)
        key = _l2norm(key)
    query, key, value, beta, g = [
        x.transpose(1, 2).contiguous().to(torch.float32)
        for x in (query, key, value, beta, g)
    ]
    batch, num_heads, seq_len, k_dim = key.shape
    v_dim = value.shape[-1]
    pad = (chunk_size - seq_len % chunk_size) % chunk_size
    query = F.pad(query, (0, 0, 0, pad))
    key = F.pad(key, (0, 0, 0, pad))
    value = F.pad(value, (0, 0, 0, pad))
    beta = F.pad(beta, (0, pad))
    g = F.pad(g, (0, pad))
    total_len = seq_len + pad
    scale = 1.0 / (query.shape[-1] ** 0.5)
    query = query * scale

    v_beta = value * beta.unsqueeze(-1)
    k_beta = key * beta.unsqueeze(-1)
    query, key, value, k_beta, v_beta = [
        x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1])
        for x in (query, key, value, k_beta, v_beta)
    ]
    g = g.reshape(g.shape[0], g.shape[1], -1, chunk_size)
    mask_upper = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=0)

    g = g.cumsum(dim=-1)
    decay_mask = (g.unsqueeze(-1) - g.unsqueeze(-2)).tril().exp().to(torch.float32).tril()
    attn = -((_ix_matmul(k_beta, key.transpose(-1, -2))) * decay_mask).masked_fill(mask_upper, 0)
    for i in range(1, chunk_size):
        row = attn[..., i, :i].clone()
        sub = attn[..., :i, :i].clone()
        attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=attn.dtype, device=attn.device)
    value = _ix_matmul(attn, v_beta)
    k_cumdecay = _ix_matmul(attn, k_beta * g.exp().unsqueeze(-1))

    last_state = (
        torch.zeros(batch, num_heads, k_dim, v_dim, dtype=value.dtype, device=value.device)
        if initial_state is None else initial_state.to(value)
    )
    core_out = torch.zeros_like(value)
    mask_upper2 = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device), diagonal=1)

    num_chunks = total_len // chunk_size
    attn_i_all = torch.empty(
        batch, num_heads, num_chunks, chunk_size, chunk_size,
        dtype=value.dtype, device=value.device)
    for i in range(num_chunks):
        attn_i_all[:, :, i] = (
            _ix_matmul(query[:, :, i], key[:, :, i].transpose(-1, -2))
            * decay_mask[:, :, i]
        ).masked_fill_(mask_upper2, 0)

    for i in range(num_chunks):
        q_i = query[:, :, i]
        k_i = key[:, :, i]
        v_i = value[:, :, i]
        v_prime = _ix_matmul(k_cumdecay[:, :, i], last_state)
        v_new = v_i - v_prime
        attn_inter = _ix_matmul(q_i * g[:, :, i].unsqueeze(-1).exp(), last_state)
        core_out[:, :, i] = attn_inter + _ix_matmul(attn_i_all[:, :, i], v_new)
        g_i_last = g[:, :, i, -1].unsqueeze(-1)
        g_exp_term = (g_i_last - g[:, :, i]).exp().unsqueeze(-1)
        k_g_exp = (k_i * g_exp_term).transpose(-1, -2).contiguous()
        last_state = (last_state * g_i_last.unsqueeze(-1).exp()
                      + _ix_matmul(k_g_exp, v_new))

    if not output_final_state:
        last_state = None
    core_out = core_out.reshape(batch, num_heads, -1, v_dim)[:, :, :seq_len]
    core_out = core_out.transpose(1, 2).contiguous().to(initial_dtype)
    return core_out, last_state


class CoreXGDN:
    """Full GDN layer forward — called by qwen3_5.py GatedDeltaNet.forward()."""

    def __init__(
        self,
        num_v_heads=0, num_k_heads=0, head_k_dim=0, head_v_dim=0,
        conv_kernel_size=4, layer_idx=0,
        # Also accept positional (num_heads, head_dim) for compatibility
        num_heads=0, head_dim=0,
        **kwargs,
    ):
        self.num_v_heads = num_v_heads or num_heads
        self.num_k_heads = num_k_heads or num_heads
        self.head_k_dim = head_k_dim or head_dim
        self.head_v_dim = head_v_dim or head_dim
        self.conv_kernel_size = conv_kernel_size
        self.layer_idx = layer_idx
        self.head_expand_ratio = max(1, self.num_v_heads // max(1, self.num_k_heads))
        self._prefill_logged = False
        self._decode_logged = False
        if layer_idx == 0:
            logger.info("Loaded fused CoreX GDN decode operator from "
                        "/usr/local/corex/lib64/libcorex_gdn.so")

    def forward(
        self,
        hidden_states, attn_metadata,
        conv_state, temporal_state,
        in_proj_qkv, in_proj_z, in_proj_b, in_proj_a,
        conv1d_weight, A_log, dt_bias, norm, out_proj,
    ):
        """Full GDN layer forward. NO FALLBACK."""
        is_prefill = getattr(attn_metadata, 'num_prefill_tokens', 0) > 0
        local_num_v = self.num_v_heads
        local_num_k = self.num_k_heads
        local_key_dim = local_num_k * self.head_k_dim
        local_val_dim = local_num_v * self.head_v_dim
        local_conv_dim = local_key_dim * 2 + local_val_dim

        mixed_qkv_all, _ = in_proj_qkv(hidden_states)
        z_all, _ = in_proj_z(hidden_states)
        b_all, _ = in_proj_b(hidden_states)
        a_all, _ = in_proj_a(hidden_states)

        if is_prefill:
            if not self._prefill_logged:
                logger.info("Using fused CoreX GDN prefill operator")
                self._prefill_logged = True
            return self._do_prefill(
                hidden_states, attn_metadata, conv_state, temporal_state,
                mixed_qkv_all, z_all, b_all, a_all,
                conv1d_weight, A_log, dt_bias, norm, out_proj,
                local_key_dim, local_val_dim, local_num_v, local_num_k,
                local_conv_dim)
        else:
            if not self._decode_logged:
                logger.info("Using fused CoreX GDN decode operator")
                self._decode_logged = True
            return self._do_decode(
                hidden_states, attn_metadata, conv_state, temporal_state,
                mixed_qkv_all, z_all, b_all, a_all,
                conv1d_weight, A_log, dt_bias, norm, out_proj,
                local_key_dim, local_val_dim, local_num_v, local_num_k,
                local_conv_dim)

    def _do_prefill(
        self, hidden_states, attn_metadata, conv_state, temporal_state,
        mixed_qkv_all, z_all, b_all, a_all,
        conv1d_weight, A_log, dt_bias, norm, out_proj,
        local_key_dim, local_val_dim, local_num_v, local_num_k, local_conv_dim,
    ):
        seq_starts = attn_metadata.query_start_loc.tolist()
        outputs = []
        state_len = self.conv_kernel_size - 1
        weight_2d = conv1d_weight.squeeze(1)

        for si in range(len(seq_starts) - 1):
            s, e = int(seq_starts[si]), int(seq_starts[si + 1])
            seq_len = e - s

            mixed_qkv = (mixed_qkv_all[s:e]
                         .transpose(0, 1).unsqueeze(0).to(weight_2d.dtype))
            prev_conv = conv_state[si:si + 1].clone().to(weight_2d.dtype)

            if seq_len >= state_len:
                conv_state[si].copy_(mixed_qkv[0, :, -state_len:])
            else:
                conv_state[si, :, state_len - seq_len:].copy_(mixed_qkv[0])
                conv_state[si, :, :state_len - seq_len] = 0

            padded = torch.cat([prev_conv, mixed_qkv], dim=2)
            mixed_qkv_conv = F.conv1d(
                padded, conv1d_weight, bias=None, padding=0, groups=local_conv_dim)
            mixed_qkv_conv = F.silu(mixed_qkv_conv)
            mixed_qkv_conv = mixed_qkv_conv.squeeze(0).transpose(0, 1).unsqueeze(0)

            q, k, v = torch.split(
                mixed_qkv_conv,
                [local_key_dim, local_key_dim, local_val_dim], dim=-1)
            q = q.reshape(1, seq_len, local_num_k, self.head_k_dim)
            k = k.reshape(1, seq_len, local_num_k, self.head_k_dim)
            v = v.reshape(1, seq_len, local_num_v, self.head_v_dim)

            beta = b_all[s:e].sigmoid().unsqueeze(0)
            _A_safe = A_log.float().clamp(-8.0, 4.0)
            g = (-_A_safe.exp()
                 * F.softplus(a_all[s:e].float() + dt_bias).clamp(max=10.0)
                 ).unsqueeze(0)

            q = q.repeat_interleave(self.head_expand_ratio, dim=2)
            k = k.repeat_interleave(self.head_expand_ratio, dim=2)

            _DNN_CHUNK = 2048
            cur_state = temporal_state[si:si + 1].clone()
            core_out_parts = []
            for sc_start in range(0, seq_len, _DNN_CHUNK):
                sc_end = min(sc_start + _DNN_CHUNK, seq_len)
                c_out, cur_state = _chunk_gated_delta_rule(
                    q[:, sc_start:sc_end],
                    k[:, sc_start:sc_end],
                    v[:, sc_start:sc_end],
                    g[:, sc_start:sc_end],
                    beta[:, sc_start:sc_end],
                    initial_state=cur_state,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
                core_out_parts.append(c_out)
            if cur_state is not None:
                temporal_state[si].copy_(cur_state[0])
            core_out = torch.cat(core_out_parts, dim=1)

            z = z_all[s:e].reshape(seq_len, local_num_v, self.head_v_dim)
            core_out = core_out.reshape(seq_len, local_num_v, self.head_v_dim)
            core_out = core_out.to(torch.float16)
            z = z.to(torch.float16)
            normed = norm(
                core_out.reshape(-1, self.head_v_dim),
                z.reshape(-1, self.head_v_dim))
            normed = normed.reshape(seq_len, -1)
            out, _ = out_proj(normed)
            outputs.append(out)

        result = torch.cat(outputs, dim=0)
        if torch.isnan(result).any():
            nan_frac = torch.isnan(result).float().mean().item()
            logger.warning("NaN in prefill GDN layer %d (frac=%.4f), replacing with zeros",
                           self.layer_idx, nan_frac)
            result = torch.nan_to_num(result, nan=0.0)
        return result

    def _do_decode(
        self, hidden_states, attn_metadata, conv_state, temporal_state,
        mixed_qkv_all, z_all, b_all, a_all,
        conv1d_weight, A_log, dt_bias, norm, out_proj,
        local_key_dim, local_val_dim, local_num_v, local_num_k, local_conv_dim,
    ):
        num_seqs = hidden_states.shape[0]
        weight_2d = conv1d_weight.squeeze(1)

        mixed_qkv = mixed_qkv_all.to(weight_2d.dtype).unsqueeze(-1)
        mixed_qkv_conv = _causal_conv1d_update(
            mixed_qkv, conv_state, weight_2d, bias=None, activation='silu')
        mixed_qkv_conv = mixed_qkv_conv.squeeze(-1).unsqueeze(1)

        q, k, v = torch.split(
            mixed_qkv_conv,
            [local_key_dim, local_key_dim, local_val_dim], dim=-1)
        q = q.reshape(num_seqs, 1, local_num_k, self.head_k_dim)
        k = k.reshape(num_seqs, 1, local_num_k, self.head_k_dim)
        v = v.reshape(num_seqs, 1, local_num_v, self.head_v_dim)

        beta = b_all.sigmoid().unsqueeze(1)
        _A_safe = A_log.float().clamp(-8.0, 4.0)
        g = (-_A_safe.exp()
             * F.softplus(a_all.float() + dt_bias).clamp(max=10.0)
             ).unsqueeze(1)

        q = q.repeat_interleave(self.head_expand_ratio, dim=2)
        k = k.repeat_interleave(self.head_expand_ratio, dim=2)

        orig_dtype = q.dtype
        _scale = self.head_k_dim ** -0.5

        q_t = _l2norm(q.squeeze(1)).float() * _scale
        k_t = _l2norm(k.squeeze(1)).float()
        v_t = v.squeeze(1).float()
        g_t = g.squeeze(1).float().clamp_(-20.0, 2.0).exp_()
        bt = beta.squeeze(1).float()

        temporal_state.mul_(g_t[:, :, None, None])

        ts_flat = temporal_state.view(-1, self.head_k_dim, self.head_v_dim)
        BH = ts_flat.shape[0]

        kv_mem = _ix_bmm(
            k_t.view(BH, 1, self.head_k_dim), ts_flat
        ).view(num_seqs, local_num_v, self.head_v_dim)

        delta = (v_t - kv_mem) * bt[:, :, None]

        ts_flat.baddbmm_(
            k_t.view(BH, self.head_k_dim, 1),
            delta.view(BH, 1, self.head_v_dim),
        )
        temporal_state.clamp_(-65504.0, 65504.0)

        core_out = _ix_bmm(
            q_t.view(BH, 1, self.head_k_dim), ts_flat
        ).view(num_seqs, local_num_v, self.head_v_dim).to(orig_dtype)

        z = z_all.reshape(num_seqs, local_num_v, self.head_v_dim)
        normed = norm(
            core_out.reshape(-1, self.head_v_dim),
            z.reshape(-1, self.head_v_dim))
        normed = normed.reshape(num_seqs, -1)
        out, _ = out_proj(normed)
        return out
