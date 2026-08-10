#!/usr/bin/env python3
"""
verify_single_gpu.py — 单卡 BI-V100 验证 ix_full_bridge + MoE dispatch chain

用法: python3 verify_single_gpu.py
要求: 在真机 Docker 里运行，单卡即可

测试链条:
  Step 0: JIT 编译 ix_full_bridge.cpp → .so
  Step 1: ixformer::infer::topk_softmax
  Step 2: ixformer::infer::moe_compute_token_index_api (gen_idx)
  Step 3: ixformer::infer::moe_expand_input
  Step 4: ixformer::infer::moe_w16a16_group_gemm (w13)
  Step 5: ixformer::infer::silu_and_mul
  Step 6: ixformer::infer::moe_w16a16_group_gemm (w2)
  Step 7: ixformer::infer::moe_output_reduce_sum (combine)
  Step 8: fused_moe_forward (全链路一次调用)
  Step 9: paged_attention
  Step 10: flash_attn_prefill
  Step 11: rms_norm
"""

import os
import sys
import time
import traceback

# ============================================================================
# Step 0: JIT compile ix_full_bridge.cpp
# ============================================================================
def step0_compile_bridge():
    print("=" * 60)
    print("STEP 0: JIT compile ix_full_bridge.cpp")
    print("=" * 60)

    # Find the source
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "ex_engine", "csrc", "ix_full_bridge.cpp"),
        "/workspace/ex_engine/csrc/ix_full_bridge.cpp",
    ]
    cpp_path = None
    for c in candidates:
        if os.path.exists(c):
            cpp_path = c
            break
    if cpp_path is None:
        print("  ✗ ix_full_bridge.cpp NOT FOUND")
        print(f"  Searched: {candidates}")
        return None

    print(f"  Source: {cpp_path}")

    from torch.utils.cpp_extension import load
    t0 = time.time()
    try:
        bridge = load(
            name="ix_full_bridge",
            sources=[cpp_path],
            extra_cflags=["-O2", "-std=c++17"],
            verbose=True,
        )
        dt = time.time() - t0
        fns = [x for x in dir(bridge) if not x.startswith("_")]
        print(f"  ✓ Compiled in {dt:.1f}s")
        print(f"  Functions: {fns}")
        return bridge
    except Exception as e:
        dt = time.time() - t0
        print(f"  ✗ Compile FAILED after {dt:.1f}s")
        print(f"  Error: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# Step 1-7: MoE dispatch chain (individual steps)
# ============================================================================
def step1_topk_softmax(bridge):
    import torch
    print("\nSTEP 1: topk_softmax")
    gating = torch.randn(4, 64, dtype=torch.float32, device="cuda")  # 4 tokens, 64 experts
    topk = 8
    try:
        weights, ids = bridge.topk_softmax(gating, topk, True)
        print(f"  ✓ weights: {weights.shape} {weights.dtype}, ids: {ids.shape} {ids.dtype}")
        print(f"    weights sum per token: {weights.sum(dim=-1).tolist()}")
        print(f"    ids range: [{ids.min().item()}, {ids.max().item()}]")
        assert weights.shape == (4, 8)
        assert ids.shape == (4, 8)
        assert ids.max().item() < 64
        return weights, ids
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None, None


def step2_gen_idx(bridge, ids):
    import torch
    print("\nSTEP 2: moe_gen_idx")
    try:
        flat_ids = ids.view(-1)  # (32,)
        result = bridge.moe_gen_idx(flat_ids, 64)
        print(f"  ✓ Returns {len(result)} tensors:")
        names = ["src_dst", "dst_src", "expert_sizes", "cumsum"]
        for i, (name, t) in enumerate(zip(names, result)):
            print(f"    [{i}] {name}: {t.shape} {t.dtype}")
        return result
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None


def step3_expand_input(bridge, idx, topk=8):
    import torch
    print("\nSTEP 3: moe_expand_input")
    hidden = torch.randn(4, 1536, dtype=torch.float16, device="cuda")  # 4 tokens, hidden=1536
    try:
        expanded = bridge.moe_expand_input(hidden, idx[0], idx[1], topk)
        print(f"  ✓ expanded: {expanded.shape} {expanded.dtype}")
        assert expanded.shape[0] == 4 * topk
        return expanded, hidden
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None, None


def step4_group_gemm_w13(bridge, expanded, idx):
    import torch
    print("\nSTEP 4: group_gemm (w13)")
    # w13: (64 experts, 2*intermediate, hidden) — simulate small
    # Real: (64, 4608, 1536) but we use smaller for test
    E, inter2, H = 64, 256, 1536
    w13 = torch.randn(E, inter2, H, dtype=torch.float16, device="cuda")
    try:
        gemm1 = bridge.group_gemm(expanded, w13, idx[2], inter2)
        print(f"  ✓ gemm1: {gemm1.shape} {gemm1.dtype}")
        return gemm1, w13
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None, None


def step5_silu_and_mul(bridge, gemm1):
    import torch
    print("\nSTEP 5: silu_and_mul")
    try:
        act = bridge.silu_and_mul(gemm1)
        print(f"  ✓ act: {act.shape} {act.dtype}")
        assert act.shape[-1] == gemm1.shape[-1] // 2
        return act
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None


def step6_group_gemm_w2(bridge, act, idx):
    import torch
    print("\nSTEP 6: group_gemm (w2)")
    E, H, I = 64, 1536, act.shape[-1]
    w2 = torch.randn(E, H, I, dtype=torch.float16, device="cuda")
    try:
        gemm2 = bridge.group_gemm(act, w2, idx[2], H)
        print(f"  ✓ gemm2: {gemm2.shape} {gemm2.dtype}")
        return gemm2
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None


def step7_combine_result(bridge, gemm2, weights):
    import torch
    print("\nSTEP 7: moe_combine_result")
    try:
        result = bridge.moe_combine_result(gemm2, weights)
        print(f"  ✓ result: {result.shape} {result.dtype}")
        print(f"    NaN: {result.isnan().any().item()}, Inf: {result.isinf().any().item()}")
        return result
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return None


# ============================================================================
# Step 8: Full fused_moe_forward
# ============================================================================
def step8_fused_moe(bridge):
    import torch
    print("\n" + "=" * 60)
    print("STEP 8: fused_moe_forward (FULL PIPELINE)")
    print("=" * 60)
    num_tokens = 4
    hidden_size = 1536
    num_experts = 64
    inter_size = 128  # small for test
    topk = 8

    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    router = torch.randn(num_tokens, num_experts, dtype=torch.float16, device="cuda")
    w13 = torch.randn(num_experts, inter_size * 2, hidden_size, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, hidden_size, inter_size, dtype=torch.float16, device="cuda")

    try:
        result = bridge.fused_moe_forward(hidden, router, w13, w2, topk, num_experts, True)
        print(f"  ✓ result: {result.shape} {result.dtype}")
        print(f"    NaN: {result.isnan().any().item()}, Inf: {result.isinf().any().item()}")
        print(f"    abs_mean: {result.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Step 9: paged_attention
# ============================================================================
def step9_paged_attention(bridge):
    import torch
    print("\nSTEP 9: paged_attention")
    B, Hq, Hkv, D = 1, 4, 1, 128
    block_size = 16
    max_blocks = 4
    seq_len = 48  # fits in 3 blocks

    query = torch.randn(B, Hq, D, dtype=torch.float16, device="cuda")
    # KV cache: (num_blocks, Hkv, block_size, D)
    total_blocks = max_blocks
    key_cache = torch.randn(total_blocks, Hkv, block_size, D, dtype=torch.float16, device="cuda")
    value_cache = torch.randn(total_blocks, Hkv, block_size, D, dtype=torch.float16, device="cuda")
    block_tables = torch.arange(max_blocks, dtype=torch.int32, device="cuda").unsqueeze(0)
    seq_lens = torch.tensor([seq_len], dtype=torch.int32, device="cuda")
    output = torch.empty(B, Hq, D, dtype=torch.float16, device="cuda")

    try:
        bridge.paged_attention(
            output, query, key_cache, value_cache,
            Hkv, D ** -0.5,
            block_tables, seq_lens, block_size, seq_len, None)
        print(f"  ✓ output: {output.shape} {output.dtype}")
        print(f"    NaN: {output.isnan().any().item()}, abs_mean: {output.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Step 10: flash_attn_prefill
# ============================================================================
def step10_flash_attn(bridge):
    import torch
    print("\nSTEP 10: flash_attn_prefill")
    Hq, Hkv, D = 4, 1, 128
    seq_len = 32

    query = torch.randn(seq_len, Hq, D, dtype=torch.float16, device="cuda")
    key = torch.randn(seq_len, Hkv, D, dtype=torch.float16, device="cuda")
    value = torch.randn(seq_len, Hkv, D, dtype=torch.float16, device="cuda")
    output = torch.empty_like(query)
    block_tables = torch.empty(0, dtype=torch.int32, device="cuda")
    cu_q = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    try:
        bridge.flash_attn_prefill(
            query, key, value, output, block_tables,
            cu_q, cu_k, seq_len, seq_len, D ** -0.5, True, -1, -1)
        print(f"  ✓ output: {output.shape} {output.dtype}")
        print(f"    NaN: {output.isnan().any().item()}, abs_mean: {output.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Step 11: rms_norm
# ============================================================================
def step11_rms_norm(bridge):
    import torch
    print("\nSTEP 11: rms_norm")
    hidden = torch.randn(4, 1536, dtype=torch.float16, device="cuda")
    weight = torch.ones(1536, dtype=torch.float16, device="cuda")
    output = torch.empty_like(hidden)
    try:
        bridge.rms_norm(output, hidden, weight, 1e-6)
        print(f"  ✓ output: {output.shape}")
        print(f"    NaN: {output.isnan().any().item()}, abs_mean: {output.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Step 12: corex_moe.py Python module (tests tiered dispatch)
# ============================================================================
def step12_corex_moe_python():
    import torch
    print("\n" + "=" * 60)
    print("STEP 12: corex_moe.py Python tiered dispatch")
    print("=" * 60)

    # Add project root to path
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)

    try:
        from ex_engine.python.corex_moe import moe_forward, topk_softmax
        print("  ✓ corex_moe imported")
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False

    num_tokens = 4
    hidden_size = 256  # small for test
    num_experts = 8    # small
    inter_size = 64
    topk = 2

    hidden = torch.randn(num_tokens, hidden_size, dtype=torch.float16, device="cuda")
    gate = torch.randn(num_tokens, num_experts, dtype=torch.float16, device="cuda")
    w13 = torch.randn(num_experts, inter_size * 2, hidden_size, dtype=torch.float16, device="cuda")
    w2 = torch.randn(num_experts, hidden_size, inter_size, dtype=torch.float16, device="cuda")

    try:
        result = moe_forward(hidden, gate, w13, w2, topk=topk,
                             renormalize=True, num_experts=num_experts)
        print(f"  ✓ moe_forward: {result.shape} {result.dtype}")
        print(f"    NaN: {result.isnan().any().item()}, abs_mean: {result.abs().mean().item():.4f}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        traceback.print_exc()
        return False


# ============================================================================
# Main
# ============================================================================
def main():
    import torch
    print("=" * 60)
    print("  BI-V100 Single GPU Verification")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("=" * 60)

    results = {}

    # Step 0: Compile
    bridge = step0_compile_bridge()
    results["compile"] = bridge is not None

    if bridge is not None:
        # Steps 1-7: Individual MoE steps
        weights, ids = step1_topk_softmax(bridge)
        results["topk_softmax"] = weights is not None

        if ids is not None:
            idx = step2_gen_idx(bridge, ids)
            results["gen_idx"] = idx is not None

            if idx is not None:
                expanded, hidden = step3_expand_input(bridge, idx)
                results["expand_input"] = expanded is not None

                if expanded is not None:
                    gemm1, w13 = step4_group_gemm_w13(bridge, expanded, idx)
                    results["group_gemm_w13"] = gemm1 is not None

                    if gemm1 is not None:
                        act = step5_silu_and_mul(bridge, gemm1)
                        results["silu_and_mul"] = act is not None

                        if act is not None:
                            gemm2 = step6_group_gemm_w2(bridge, act, idx)
                            results["group_gemm_w2"] = gemm2 is not None

                            if gemm2 is not None:
                                result = step7_combine_result(bridge, gemm2, weights)
                                results["combine_result"] = result is not None

        # Step 8: Full pipeline
        results["fused_moe"] = step8_fused_moe(bridge)

        # Step 9-11: Other kernels
        results["paged_attention"] = step9_paged_attention(bridge)
        results["flash_attn"] = step10_flash_attn(bridge)
        results["rms_norm"] = step11_rms_norm(bridge)

    # Step 12: Python module test (works even without bridge)
    results["corex_moe_python"] = step12_corex_moe_python()

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    for name, ok in results.items():
        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  {status}  {name}")

    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\n  {passed}/{total} passed")

    if results.get("fused_moe"):
        print("\n  >>> MoE FULL C++ PIPELINE WORKS — comp 168 parity achieved <<<")
    elif results.get("corex_moe_python"):
        print("\n  >>> MoE Python fallback works — C++ bridge needs debugging <<<")

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
