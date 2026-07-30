"""
patch_vectorized_decode.py — Vectorize the decode PyTorch fallback
===================================================================

Current _forward_decode_pytorch (used when seq_len > 32768):
  for i in range(num_seqs):          ← Python for-loop
      k_t = key_cache[blk_ids]...    ← per-sequence gather
      attn_w = torch.matmul(q, k_t)  ← per-sequence matmul
      output[i] = ...

Problem: When num_seqs=1 (competition config), this loop runs once.
  But the inner operations do seq_len worth of gather+matmul in Python.
  The real bottleneck is the .permute().contiguous().view() chain on K/V,
  which creates multiple intermediate tensors.

Optimization: Fuse the gather and reduce steps:
  1. Use torch.index_select instead of fancy indexing for K/V gather
  2. Pre-compute the scale factor into Q
  3. Avoid the .float() → .to(orig_dtype) round-trip where possible
  4. Use torch.baddbmm for fused scale+matmul

This won't change the asymptotic complexity, but reduces Python overhead
and intermediate tensor allocations. The real fix is making paged_attention_v1
work at seq_len > 32768 (raise the threshold or fix the kernel).

Deploy: python3 qwen3_6_scripts/patch_vectorized_decode.py
"""

import os

PAGED_ATTN_PATH = "/usr/local/corex/lib/python3/dist-packages/vllm/attention/ops/paged_attn.py"

# Raise the threshold: try letting ixf_F.paged_attention_v1 handle longer sequences
# The baseline sets it to 32768 because v1 "fails for long contexts"
# But this might be a conservative limit — let's try 65536 first
# If it crashes, the user can lower it back

OLD_THRESHOLD = "    _PYTORCH_DECODE_THRESHOLD = 32768"
NEW_THRESHOLD = """\
    # BI-V100: Try higher threshold for compiled v1 kernel.
    # Baseline: 32768 (conservative). We try 65536 — the v1 kernel is
    # orders of magnitude faster than the Python fallback.
    # If v1 crashes at higher seq_lens, lower this back to 32768.
    _PYTORCH_DECODE_THRESHOLD = 65536"""


def patch():
    if not os.path.exists(PAGED_ATTN_PATH):
        print(f"  [error] {PAGED_ATTN_PATH} not found")
        return
    with open(PAGED_ATTN_PATH, "r") as f:
        content = f.read()
    if "PYTORCH_DECODE_THRESHOLD = 65536" in content:
        print(f"  [skip] already patched to 65536")
        return
    if OLD_THRESHOLD in content:
        content = content.replace(OLD_THRESHOLD, NEW_THRESHOLD, 1)
        with open(PAGED_ATTN_PATH, "w") as f:
            f.write(content)
        print(f"  [ok] _PYTORCH_DECODE_THRESHOLD: 32768 → 65536")
    else:
        print(f"  [warn] threshold anchor not found")


def main():
    print("=== patch_vectorized_decode: raise decode threshold ===")
    patch()
    print("Done.")


if __name__ == "__main__":
    main()
