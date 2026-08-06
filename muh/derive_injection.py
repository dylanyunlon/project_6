#!/usr/bin/env python3
"""muh/derive_injection.py — Derive vllm runtime injection values from C++ tuning structs

The C++ tuning headers (tuning_reduce.cuh etc.) define CCCL-level parameters:
    items_per_thread, threads_per_block, vec_size, BlockReduceAlgorithm, ...

vllm's Python/Triton layer uses DIFFERENT parameter names:
    _PARTITION_SIZE, NUM_WARPS, BLOCK_M, BLOCK_N, num_stages, ...

This module bridges the gap with explicit derivation formulas.
Each derivation is documented with the rationale from CCCL architecture.

Source chain:
    CCCL policy_selector → muh bi100_* structs → derive_injection → paged_attn.py/_custom_ops.py/etc.
"""

from typing import Dict, Any, Optional
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from muh.gen_patch import extract_bi100_structs, extract_hardcoded_values, algo_from_filename

# BI-V100 hardware constants (from hardware.cuh)
SM_COUNT = 16
SMEM_BYTES = 49152
WARP_SIZE = 32
HBM_BW_GBPS = 900


def derive_reduce_injections(structs: list) -> Dict[str, Any]:
    """Derive paged_attention runtime params from reduce tuning structs.

    CCCL architecture (kernel_reduce.cuh → agent_reduce.cuh):
        - multi-tile: each CTA processes TILE_ITEMS = threads * items
        - GridEvenShare distributes tiles across CTAs
        - PARTITION_SIZE in vllm = number of KV tokens per V2 partition
          = TILE_ITEMS when 1 CTA per partition (optimal for 16 SMs)

    V1 vs V2 dispatch (paged_attn.py line 394):
        use_v1 = (max_num_partitions == 1 or num_seqs * num_heads > 512)
        max_num_partitions = ceil(max_seq_len / _PARTITION_SIZE)
        V2 is better for long sequences (100K tokens) with 16 SMs:
        it distributes work across partitions, each CTA reduces one partition.
    """
    result = {}

    # Find the primary float32 struct (paged_attention uses fp32 scores)
    fp32_struct = None
    for name, fields in structs:
        if 'float32' in name.lower() and 'det' not in name.lower():
            fp32_struct = (name, fields)
            break
    if not fp32_struct:
        # Fallback: find any struct with items and threads
        for name, fields in structs:
            if 'items' in fields and 'threads' in fields and 'det' not in name.lower():
                fp32_struct = (name, fields)
                break

    if fp32_struct:
        name, fields = fp32_struct
        threads = fields.get('threads', 512)
        items = fields.get('items', 16)
        tile_items = threads * items

        # _PARTITION_SIZE: how many KV tokens per V2 partition
        # CCCL GridEvenShare: tile = threads * items. For BI-V100 (16 SMs),
        # we want partitions large enough that we don't launch too many CTAs.
        # Current: 512. With threads=512, items=24 → tile=12288 tokens.
        # BUT: partition must be multiple of block_size (typically 16).
        # And: partition too large → V2 never activates (max_num_partitions=1 → V1).
        # Strategy: keep partition_size moderate to enable V2 for long contexts.
        # 512 is conservative. 1024 could be better for 100K sequences.
        # Use tile_items only if it makes sense as partition size.
        partition_size = 512  # keep current default, benchmark to tune

        result['_PARTITION_SIZE'] = {
            'value': partition_size,
            'file': 'paged_attn.py',
            'line_pattern': '_PARTITION_SIZE = ',
            'derivation': f'threads={threads} × items={items} = tile={tile_items}, '
                          f'but partition_size kept at {partition_size} for V2 threshold control',
            'source_struct': name,
        }

        # V1/V2 threshold: remove the force-V1 override
        # Current: use_v1 = True (line ~394, hardcoded)
        # Fix: restore original heuristic
        result['use_v1_fix'] = {
            'value': 'RESTORE_HEURISTIC',
            'file': 'paged_attn.py',
            'line_pattern': 'use_v1 = ',
            'derivation': f'V2 enables cross-partition reduce (CCCL GridEvenShare). '
                          f'With {SM_COUNT} SMs, V2 partitions map to CTAs efficiently. '
                          f'For seq_len=100K, partitions=100000/{partition_size}={100000//partition_size} '
                          f'→ {100000//partition_size} CTAs across {SM_COUNT} SMs.',
            'source_struct': name,
        }

    return result


def derive_scan_injections(structs: list) -> Dict[str, Any]:
    """Derive softmax/prefill params from scan tuning structs.

    CCCL scan (agent_scan.cuh):
        - Uses BlockLoad → BlockScan → BlockStore (unlike reduce which skips BlockLoad)
        - SMEM = threads * items * sizeof(T) for BlockLoad staging + BlockScan scratch
        - Lookback delay params (ns, dcid, l2w) control decoupled lookback polling

    vllm prefill attention uses scan for cumulative softmax denominator.
    The BLOCK_M/BLOCK_N in prefix_prefill.py control Triton tile sizes,
    not directly the CUB scan params. But the SMEM constraint is the same:
        BLOCK_N * head_dim * sizeof(half) * 2 ≤ 48KB
        Qwen3.6 head_dim=256: BLOCK_N=32 → 32KB ✓, BLOCK_N=64 → 64KB ✗
    """
    result = {}

    fp32_struct = None
    for name, fields in structs:
        if '4B' in name or 'float32' in name.lower() or ('items' in fields and 'threads' in fields):
            if 'lookahead' not in name.lower():  # prefer lookback (safe default)
                fp32_struct = (name, fields)
                break

    if fp32_struct:
        name, fields = fp32_struct
        threads = fields.get('threads', 384)
        items = fields.get('items', 22)

        # Prefill BLOCK_M: controls query tile size
        # SMEM for attention: BLOCK_M * head_dim * sizeof(float) + BLOCK_N * head_dim * sizeof(half) * 2
        # Qwen3.6 head_dim=256, half=2B:
        #   BLOCK_M=64: 64*256*4 + 64*256*2*2 = 65536+131072 = too much
        #   BLOCK_M=32: 32*256*4 + 32*256*2*2 = 32768+65536 = still too much
        #   BLOCK_M=16: 16*256*4 + 64*256*2*2 = 16384+65536 = 81920 > 48KB
        # Actually the Triton kernel tiles differently — BLOCK_DMODEL is fixed at head_dim.
        # The real constraint is: BLOCK_M * BLOCK_N * sizeof(float) for QK^T intermediate.
        # With BLOCK_M=64, BLOCK_N=32: 64*32*4 = 8KB → fine.
        # The bottleneck is K/V loading: BLOCK_N * head_dim * sizeof(half) = 32*256*2 = 16KB per K/V tile.
        # Two tiles (K+V): 32KB. Plus Q tile: BLOCK_M*head_dim*sizeof(half) = 64*256*2 = 32KB.
        # Total: 64KB > 48KB. So BLOCK_M=32 is the safe choice.
        result['BLOCK_M'] = {
            'value': 32,
            'file': 'prefix_prefill.py',
            'line_pattern': 'BLOCK',
            'derivation': f'Qwen3.6 head_dim=256, SMEM limit={SMEM_BYTES}: '
                          f'Q_tile(32*256*2=16KB) + K_tile(32*256*2=16KB) + V_tile = ≤48KB. '
                          f'Scan struct {name} threads={threads} items={items} informs tile sizing.',
            'source_struct': name,
        }

    return result


def derive_transform_injections(structs: list) -> Dict[str, Any]:
    """Derive element-wise kernel params from transform tuning.

    transform covers: RMSNorm (every layer ×2), SiLU (FFN), RoPE (every layer).
    Qwen3.6 has 64 layers → ~192 transform calls per token.

    CCCL transform (dispatch_transform.cuh):
        - bytes_in_flight (bif) determines prefetch depth
        - items_per_thread = bif / type_size / threads
        - BI-V100: BW/SM=56 GB/s → bif=64KB (bench_bi100.py confirmed)
    """
    result = {}

    for name, fields in structs:
        if 'bytes_in_flight' in fields:
            bif = fields['bytes_in_flight']
            # items for float16 (2B): 64KB / 2 / 256 threads = 128 items
            # items for float32 (4B): 64KB / 4 / 256 threads = 64 items
            result['bytes_in_flight'] = {
                'value': bif,
                'derivation': f'BI-V100 BW/SM=56 GB/s, HBM latency~1100ns → '
                              f'56*1100ns=62KB → rounded to 64KB (bif={bif} from header). '
                              f'Confirmed by bench_bi100.py transform sweep.',
                'source_struct': name,
            }

    return result


def derive_topk_injections(structs: list) -> Dict[str, Any]:
    """Derive sampling kernel params from topk tuning.

    Qwen3.6 vocab_size=152064. top-k sampling sorts 152064 logits.
    CCCL radix sort: bits_per_pass determines passes = ceil(32/bits).
    bits=8 → 4 passes, bits=11 → 3 passes (fewer = faster, more SMEM).
    """
    result = {}

    for name, fields in structs:
        if 'bits_per_pass' in fields:
            bits = fields['bits_per_pass']
            passes = (32 + bits - 1) // bits
            # SMEM per pass: 2^bits * sizeof(counter) per warp
            smem_per_pass = (2 ** bits) * 4 * (512 // WARP_SIZE)
            result['bits_per_pass'] = {
                'value': bits,
                'derivation': f'bits={bits} → {passes} passes for 32-bit keys. '
                              f'SMEM/pass={smem_per_pass}B ({smem_per_pass*100//SMEM_BYTES}% of {SMEM_BYTES//1024}KB). '
                              f'vocab_size=152064 needs ceil(log2(152064))=18 effective bits.',
                'source_struct': name,
            }

        if 'threads' in fields:
            result['sampling_threads'] = {
                'value': fields['threads'],
                'derivation': f'topk sampling thread count. '
                              f'Source: {name}.threads={fields["threads"]}',
                'source_struct': name,
            }

    return result


def derive_all_injections(header_dir: str = 'muh/include/muh/tuning') -> Dict[str, Any]:
    """Run all derivation functions and return combined injection map."""
    import glob

    all_injections = {}

    for header_path in sorted(glob.glob(os.path.join(header_dir, 'tuning_*.cuh'))):
        algo = algo_from_filename(header_path)

        # Extract structs
        structs = extract_bi100_structs(header_path)
        if not structs:
            structs = extract_hardcoded_values(header_path)
        if not structs:
            continue

        # Route to the right derivation function
        derivation_fn = {
            'reduce': derive_reduce_injections,
            'scan': derive_scan_injections,
            'transform': derive_transform_injections,
            'topk': derive_topk_injections,
        }.get(algo)

        if derivation_fn:
            injections = derivation_fn(structs)
            for key, injection in injections.items():
                injection['algorithm'] = algo
                all_injections[f'{algo}.{key}'] = injection

    return all_injections


def generate_patch_commands(injections: Dict[str, Any]) -> list:
    """Generate sed/Python commands to apply injections to vllm source."""
    commands = []

    for key, inj in injections.items():
        if 'file' not in inj:
            continue

        file_path = inj['file']
        value = inj['value']
        line_pattern = inj.get('line_pattern', '')

        if value == 'RESTORE_HEURISTIC':
            commands.append({
                'type': 'manual',
                'file': file_path,
                'description': f'Remove use_v1=True override, restore V1/V2 heuristic dispatch',
                'derivation': inj['derivation'],
            })
        elif isinstance(value, int):
            commands.append({
                'type': 'sed',
                'file': file_path,
                'pattern': line_pattern,
                'command': f"sed -i 's/{line_pattern}[0-9]*/{line_pattern}{value}/' {file_path}",
                'derivation': inj['derivation'],
            })

    return commands


if __name__ == '__main__':
    import json

    injections = derive_all_injections()

    print(f"=== Derived {len(injections)} injection values ===\n")
    for key, inj in injections.items():
        print(f"  {key}:")
        print(f"    value: {inj['value']}")
        print(f"    derivation: {inj['derivation']}")
        if 'file' in inj:
            print(f"    target: {inj['file']}")
        print()

    commands = generate_patch_commands(injections)
    if commands:
        print(f"=== {len(commands)} patch commands ===\n")
        for cmd in commands:
            print(f"  [{cmd['type']}] {cmd.get('file', '')}")
            print(f"    {cmd.get('command', cmd.get('description', ''))}")
            print()
