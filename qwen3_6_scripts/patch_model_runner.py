"""
Fix: prefix_cache_hit stays True for chunked-prefill chunk 2+ even when past cache.

Root cause:
  model_runner.py _compute_for_prefix_cache_hit has three cases:
    Case 1: prefix_cache_len <= context_len  → "already past cache, do normal"
    Case 2: context_len < prefix_cache_len < seq_len  → partial hit, correct
    Case 3: seq_len <= prefix_cache_len  → full hit, reduce to 1 token

  Case 1 does nothing (leaves prefix_cache_hit = True). Then in utils.py:
    if inter_data.prefix_cache_hit:
        block_table = computed_block_nums   ← ONLY the original prefix blocks!

  But context_len > prefix_cache_len means chunk 1 tokens (between prefix_cache_len
  and context_len) are ALSO in KV cache and need to be in block_table.
  block_table = computed_block_nums misses all chunk-1 blocks.

  In _forward_prefix_pytorch:
    num_ctx_blocks = ceil(context_len / block_size)  # e.g. 268
    block_tables.shape[1] = len(computed_block_nums)  # e.g. 12  <-- too small!
    At tile_blk >= 12: blk_ids is empty → k_t shape [..., 0] → amax crash.

Fix:
  Set prefix_cache_hit = False for Case 1, so utils.py falls through to:
    elif chunked_prefill_enabled:
        block_table = block_tables[seq_id]   ← full block table (prefix + chunk1)
"""

import re
import sys

CANDIDATE_PATHS = [
    "/usr/local/corex/lib64/python3/dist-packages/vllm/worker/model_runner.py",
    "/usr/local/corex/lib/python3/dist-packages/vllm/worker/model_runner.py",
]

OLD_BLOCK = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            pass"""

NEW_BLOCK = """\
        if prefix_cache_len <= context_len:
            # We already passed the cache hit region,
            # so do normal computation.
            # Must clear prefix_cache_hit so _add_seq_group uses the full
            # block_tables (prefix + previous-chunk blocks) instead of only
            # computed_block_nums (prefix only).  Without this, block_tables
            # passed to _forward_prefix_pytorch is too narrow for context_len,
            # causing an empty blk_ids slice and a zero-dim amax() crash.
            inter_data.prefix_cache_hit = False"""

import os

patched = False
for path in CANDIDATE_PATHS:
    if not os.path.exists(path):
        continue
    with open(path, "r") as f:
        src = f.read()
    if OLD_BLOCK not in src:
        if NEW_BLOCK in src:
            print(f"[patch_model_runner] already patched: {path}")
            patched = True
            break
        print(f"[patch_model_runner] WARNING: expected block not found in {path}, skipping")
        continue
    patched_src = src.replace(OLD_BLOCK, NEW_BLOCK, 1)
    with open(path, "w") as f:
        f.write(patched_src)
    print(f"[patch_model_runner] patched Case-1 prefix_cache_hit fix in: {path}")
    patched = True
    break

if not patched:
    print("[patch_model_runner] ERROR: could not find model_runner.py at any known path", file=sys.stderr)
    sys.exit(1)
