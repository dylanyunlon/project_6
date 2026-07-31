"""
patch_ixformer_native.py — Enable ixformer's native V1/V2 paged attention kernels
===================================================================================

CRITICAL FIXES discovered from hardware diagnostics:

1. V1 head_mapping: paged_attn.py passes num_kv_heads (int=4), but
   ixformer_torch_ops.vllm_single_query_cached_kv_attention requires a Tensor.
   The Tensor is: torch.repeat_interleave(torch.arange(num_kv_heads, device), num_queries_per_kv)
   For Qwen3.6: [0,0,0,0,0,0, 1,1,1,1,1,1, 2,2,2,2,2,2, 3,3,3,3,3,3]

2. V2 native kernel EXISTS in ixformer (vllm_single_query_cached_kv_attention_v2)
   but _custom_ops.py has raise NotImplementedError(). The V2 signature:
     (output, partition, exp_sums, max_logits, temp_output, query, 
      key_cache, value_cache, head_mapping, scale, block_tables, 
      context_lens, block_size, max_context_len, alibi_slopes, use_sqrt_alibi)
   Note: V2 has an extra 'partition' (int) parameter that V1 doesn't.

3. Triton 2.3.1 is installed at /usr/local/lib/python3.10/site-packages/triton
   but vllm (at /usr/local/corex/lib64/python3/dist-packages/vllm) says
   "Triton not installed" — path mismatch.

This patch fixes all three by modifying _custom_ops.py and the Triton import.
"""

import os
import sys


VLLM_ROOT = "/usr/local/corex/lib64/python3/dist-packages/vllm"
CUSTOM_OPS_PATH = os.path.join(VLLM_ROOT, "_custom_ops.py")


def patch_custom_ops():
    """Fix V1 head_mapping and replace V2 NotImplementedError with native ixformer kernel."""
    with open(CUSTOM_OPS_PATH, "r") as f:
        content = f.read()

    changes = []

    # --- Fix 1: V1 head_mapping must be a Tensor ---
    # Current code passes head_mapping directly through.
    # paged_attn.py passes num_kv_heads (int).
    # We need to convert int → Tensor inside _custom_ops.py.
    old_v1 = '''def paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes=None,
        kv_cache_dtype=None,
):
    return ixf_F.vllm_single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )'''

    new_v1 = '''def paged_attention_v1(
        output,
        query,
        key_cache,
        value_cache,
        head_mapping,
        scale,
        block_tables,
        context_lens,
        block_size,
        max_context_len,
        alibi_slopes=None,
        kv_cache_dtype=None,
):
    # BI-V100 fix: ixformer requires head_mapping as Tensor, not int.
    # paged_attn.py passes num_kv_heads (int). Convert here.
    if isinstance(head_mapping, int):
        num_kv_heads = head_mapping
        num_heads = query.shape[1]
        num_queries_per_kv = num_heads // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device=query.device),
            num_queries_per_kv)
    return ixf_F.vllm_single_query_cached_kv_attention(
            output,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            context_lens,
            block_size,
            max_context_len,
            alibi_slopes,
        )'''

    if old_v1 in content:
        content = content.replace(old_v1, new_v1, 1)
        changes.append("V1: Added int→Tensor conversion for head_mapping")
    else:
        print("  [warn] V1 function body not found as expected — may already be patched")

    # --- Fix 2: V2 replace NotImplementedError with native ixformer kernel ---
    # ixformer signature:
    #   vllm_single_query_cached_kv_attention_v2(
    #       output, partition, exp_sums, max_logits, temp_output,
    #       query, key_cache, value_cache, head_mapping, scale,
    #       block_tables, context_lens, block_size, max_context_len,
    #       alibi_slopes, use_sqrt_alibi)
    # _PARTITION_SIZE = 512 (from paged_attn.py)

    old_v2 = '''    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    raise NotImplementedError()'''

    new_v2 = '''    blocksparse_block_size: int = 64,
    blocksparse_head_sliding_step: int = 0,
) -> None:
    # BI-V100: Use ixformer native V2 kernel instead of NotImplementedError.
    # Convert num_kv_heads (int) → head_mapping (Tensor)
    if isinstance(num_kv_heads, int):
        num_heads = query.shape[1]
        num_queries_per_kv = num_heads // num_kv_heads
        head_mapping = torch.repeat_interleave(
            torch.arange(num_kv_heads, dtype=torch.int32, device=query.device),
            num_queries_per_kv)
    else:
        head_mapping = num_kv_heads

    _PARTITION_SIZE = 512
    max_num_partitions = (max_seq_len + _PARTITION_SIZE - 1) // _PARTITION_SIZE

    return ixf_F.vllm_single_query_cached_kv_attention_v2(
            out,
            max_num_partitions,
            exp_sum,
            max_logits,
            tmp_out,
            query,
            key_cache,
            value_cache,
            head_mapping,
            scale,
            block_tables,
            seq_lens,
            block_size,
            max_seq_len,
            alibi_slopes,
        )'''

    if old_v2 in content:
        content = content.replace(old_v2, new_v2, 1)
        changes.append("V2: Replaced NotImplementedError with ixformer native kernel")
    elif "raise NotImplementedError()" in content and "paged_attention_v2" in content:
        print("  [warn] V2 NotImplementedError found but exact match failed")
    else:
        print("  [warn] V2 may already be patched")

    with open(CUSTOM_OPS_PATH, "w") as f:
        f.write(content)

    for c in changes:
        print(f"  [ok] {c}")


def patch_triton_path():
    """Fix Triton import path so vllm can find it."""
    # Triton is at /usr/local/lib/python3.10/site-packages/triton
    # vllm checks for triton at import time in importing.py
    triton_path = "/usr/local/lib/python3.10/site-packages"
    importing_path = os.path.join(VLLM_ROOT, "importing.py")

    if not os.path.exists(importing_path):
        # Try to find it
        for root, dirs, files in os.walk(VLLM_ROOT):
            if "importing.py" in files:
                importing_path = os.path.join(root, "importing.py")
                break

    if os.path.exists(importing_path):
        with open(importing_path, "r") as f:
            content = f.read()
        
        if triton_path not in content and "Triton not installed" in content:
            # Add sys.path insertion before the triton import check
            old_import = "import triton"
            new_import = f"import sys; sys.path.insert(0, '{triton_path}'); import triton"
            if old_import in content:
                content = content.replace(old_import, new_import, 1)
                with open(importing_path, "w") as f:
                    f.write(content)
                print(f"  [ok] Triton path fix: added {triton_path} to sys.path")
            else:
                print("  [warn] Could not find 'import triton' in importing.py")
        else:
            print("  [skip] Triton path already fixed or not needed")
    else:
        print(f"  [warn] importing.py not found at {importing_path}")

    # Also create a symlink as backup
    triton_src = "/usr/local/lib/python3.10/site-packages/triton"
    triton_dst = "/usr/local/corex/lib64/python3/dist-packages/triton"
    if os.path.exists(triton_src) and not os.path.exists(triton_dst):
        try:
            os.symlink(triton_src, triton_dst)
            print(f"  [ok] Symlinked triton → corex dist-packages")
        except Exception as e:
            print(f"  [warn] Symlink failed: {e}")


def main():
    print("=== patch_ixformer_native: Enable native V1/V2 kernels ===")
    print()

    print("--- Fix 1+2: _custom_ops.py V1 head_mapping + V2 native kernel ---")
    patch_custom_ops()

    print("\n--- Fix 3: Triton path ---")
    patch_triton_path()

    print("\n=== Summary ===")
    print("V1: head_mapping int→Tensor conversion (fixes RuntimeError)")
    print("V2: ixformer native kernel (replaces NotImplementedError)")
    print("     ixf_F.vllm_single_query_cached_kv_attention_v2()")
    print("     partition = max_num_partitions (PARTITION_SIZE=512)")
    print("Triton: sys.path fix + symlink for vllm import")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
