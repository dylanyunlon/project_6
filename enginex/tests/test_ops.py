"""
EngineX operator tests.

Tests each operator implementation against known-correct behavior.
Run: python -m pytest enginex/tests/test_ops.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.nn.functional as F
import pytest


class TestMoETopkSoftmax:
    """Test the critical missing operator."""

    def test_basic_topk_selection(self):
        from enginex.ops.moe import moe_topk_softmax_pytorch

        num_tokens = 4
        num_experts = 8
        topk = 2

        gating = torch.randn(num_tokens, num_experts)
        topk_weights = torch.empty(num_tokens, topk)
        topk_ids = torch.empty(num_tokens, topk, dtype=torch.long)
        token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.long)

        moe_topk_softmax_pytorch(topk_weights, topk_ids, token_expert_indices, gating)

        # Weights should sum to ~1 per token
        sums = topk_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(num_tokens), atol=0.01)

        # IDs should be valid expert indices
        assert (topk_ids >= 0).all() and (topk_ids < num_experts).all()

        # Should pick the actual top-k from softmax
        probs = F.softmax(gating, dim=-1)
        for i in range(num_tokens):
            expected_ids = torch.topk(probs[i], k=topk).indices
            # Same experts selected (order may differ)
            assert set(topk_ids[i].tolist()) == set(expected_ids.tolist())

    def test_large_expert_count(self):
        """Qwen3.5 has 128 experts with topk=8."""
        from enginex.ops.moe import moe_topk_softmax_pytorch

        num_tokens = 16
        num_experts = 128
        topk = 8

        gating = torch.randn(num_tokens, num_experts)
        topk_weights = torch.empty(num_tokens, topk)
        topk_ids = torch.empty(num_tokens, topk, dtype=torch.long)
        token_expert_indices = torch.empty(num_tokens, topk, dtype=torch.long)

        moe_topk_softmax_pytorch(topk_weights, topk_ids, token_expert_indices, gating)

        assert topk_weights.shape == (num_tokens, topk)
        assert (topk_ids >= 0).all() and (topk_ids < num_experts).all()
        assert not torch.isnan(topk_weights).any()


class TestGDN:
    """Test GatedDeltaNet implementations."""

    def test_decode_no_nan(self):
        """The critical test — decode must not produce NaN."""
        from enginex.ops.gdn import gdn_decode_pytorch

        B, H, D = 1, 4, 128
        q = torch.randn(B, H, D)
        k = torch.randn(B, H, D)
        v = torch.randn(B, H, D)
        gate = torch.sigmoid(torch.randn(B, H))
        beta = torch.sigmoid(torch.randn(B, H)) * 0.1
        conv_state = torch.randn(B, H, 4, D)
        temporal_state = torch.randn(B, H, D, D) * 0.01

        output, new_state = gdn_decode_pytorch(
            q, k, v, gate, beta, conv_state, temporal_state)

        assert not torch.isnan(output).any(), "GDN decode produced NaN!"
        assert not torch.isnan(new_state).any(), "GDN state has NaN!"

    def test_prefill_no_nan(self):
        """Prefill with chunk_size=16 must not NaN (was 99.98% NaN with 64)."""
        from enginex.ops.gdn import gdn_prefill_pytorch

        B, L, H, D = 1, 64, 4, 128
        q = torch.randn(B, L, H, D) * 0.1
        k = torch.randn(B, L, H, D) * 0.1
        v = torch.randn(B, L, H, D) * 0.1
        gate = torch.sigmoid(torch.randn(B, L, H))
        beta = torch.sigmoid(torch.randn(B, L, H)) * 0.1
        state = torch.zeros(B, H, D, D)

        output, final_state = gdn_prefill_pytorch(
            q, k, v, gate, beta, state, chunk_size=16)

        nan_frac = torch.isnan(output).float().mean().item()
        assert nan_frac < 0.01, f"GDN prefill NaN fraction: {nan_frac:.4f}"

    def test_prefill_state_updates(self):
        """State should be different after processing tokens."""
        from enginex.ops.gdn import gdn_prefill_pytorch

        B, L, H, D = 1, 32, 2, 64
        q = torch.randn(B, L, H, D)
        k = torch.randn(B, L, H, D)
        v = torch.randn(B, L, H, D)
        gate = torch.sigmoid(torch.randn(B, L, H))
        beta = torch.sigmoid(torch.randn(B, L, H)) * 0.1
        state = torch.zeros(B, H, D, D)

        _, final_state = gdn_prefill_pytorch(
            q, k, v, gate, beta, state, chunk_size=16)

        assert not torch.allclose(final_state, state), "State unchanged after prefill!"


class TestActivations:
    def test_silu_and_mul(self):
        from enginex.ops.activations import silu_and_mul_pytorch

        d = 128
        x = torch.randn(4, d * 2)
        out = torch.empty(4, d)
        silu_and_mul_pytorch(x, out)

        expected = F.silu(x[..., :d]) * x[..., d:]
        assert torch.allclose(out, expected, atol=1e-5)

    def test_gelu_and_mul(self):
        from enginex.ops.activations import gelu_and_mul_pytorch

        d = 128
        x = torch.randn(4, d * 2)
        out = torch.empty(4, d)
        gelu_and_mul_pytorch(x, out)

        expected = F.gelu(x[..., :d]) * x[..., d:]
        assert torch.allclose(out, expected, atol=1e-5)


class TestNorm:
    def test_rms_norm(self):
        from enginex.ops.norm import rms_norm_pytorch

        hidden_size = 256
        x = torch.randn(4, hidden_size)
        w = torch.ones(hidden_size)
        out = torch.empty_like(x)

        rms_norm_pytorch(x, w, out, epsilon=1e-6)

        # Manual check
        variance = x.float().pow(2).mean(-1, keepdim=True)
        expected = (x * torch.rsqrt(variance + 1e-6)) * w
        assert torch.allclose(out, expected, atol=1e-4)


class TestRegistry:
    def test_registry_creates(self):
        from enginex.dispatch.registry import get_registry
        reg = get_registry()
        assert reg is not None

    def test_probe_runs(self):
        from enginex.dispatch.registry import OperatorRegistry
        reg = OperatorRegistry()
        reg.probe()
        # Should have registered operators
        assert len(reg.ops) > 0

    def test_pytorch_fallbacks_always_available(self):
        from enginex.dispatch.registry import OperatorRegistry, Backend
        reg = OperatorRegistry()
        reg.probe()

        # These must ALWAYS have a fallback
        critical_ops = [
            "moe_topk_softmax",
            "gdn_decode",
            "gdn_prefill",
        ]
        for op_name in critical_ops:
            entry = reg.ops.get(op_name)
            assert entry is not None, f"{op_name} not registered"
            assert entry.active is not None, f"{op_name} has no active impl"
            assert entry.active.available, f"{op_name} impl not available"

    def test_summary(self):
        from enginex.dispatch.registry import OperatorRegistry
        reg = OperatorRegistry()
        reg.probe()
        summary = reg.summary()
        assert "EngineX Operator Registry Summary" in summary
        assert "moe_topk_softmax" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
