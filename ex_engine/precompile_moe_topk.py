"""
Precompile moe_topk_softmax_v3.cu → .so during Docker build.
Same pattern as precompile_gdn.py.

Run: python3 ex_engine/precompile_moe_topk.py
"""
import os, sys

def main():
    cu_path = os.path.join(os.path.dirname(__file__), "csrc", "moe_topk_softmax_v3.cu")
    if not os.path.isfile(cu_path):
        print(f"[MOE] ERROR: {cu_path} not found")
        sys.exit(1)

    print(f"[MOE] Compiling {cu_path} ...")
    from torch.utils.cpp_extension import load
    ext = load(
        name="moe_topk_softmax_v3",
        sources=[cu_path],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    )
    print("[MOE] ✓ moe_topk_softmax_v3.so compiled successfully")

    # Verify
    import torch
    gating = torch.randn(4, 64, device='cuda', dtype=torch.float16)
    w, ids, _ = ext.moe_topk_softmax(gating, 8, True)
    assert not w.isnan().any(), "NaN in topk weights!"
    assert torch.allclose(w.sum(dim=-1), torch.ones(4, device='cuda'), atol=1e-3)
    print("[MOE] ✓ Runtime verification passed")

if __name__ == "__main__":
    main()
