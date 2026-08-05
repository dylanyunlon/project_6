#!/usr/bin/env python3
"""muh/gen_patch.py — Generate vllm kernel patches from C++ tuning headers

Reads muh/include/muh/tuning/tuning_*.cuh, extracts bi100_* struct values,
and generates unified diff patches for the vllm source tree.

The previous version read from .muh YAML files. This version reads directly
from C++ headers — single source of truth, no YAML middleman.

Usage:
    python3 muh/gen_patch.py [--header-dir muh/include/muh/tuning] [-o patches/]
"""

import re
import os
import sys
import glob
import argparse
from datetime import datetime


def extract_bi100_structs(filepath):
    """Extract all bi100_* struct constexpr values from a C++ header.

    Returns list of (struct_name, {field: value, ...}) tuples.
    """
    with open(filepath, 'r') as f:
        content = f.read()

    structs = []
    # Split on struct definitions
    # Pattern: struct bi100_xxx {  ...  };
    pattern = re.compile(
        r'struct\s+(bi100_\w+)\s*\{(.*?)\};',
        re.DOTALL
    )

    for m in pattern.finditer(content):
        name = m.group(1)
        body = m.group(2)
        fields = {}

        # Extract: static constexpr int threads = 512;
        for fm in re.finditer(
            r'static\s+constexpr\s+int\s+(\w+)\s*=\s*(\d+)',
            body
        ):
            fields[fm.group(1)] = int(fm.group(2))

        # Extract: static constexpr BlockLoadAlgorithm load_algo = BLOCK_LOAD_DIRECT;
        for fm in re.finditer(
            r'static\s+constexpr\s+\w+\s+(\w+)\s*=\s*(\w+)',
            body
        ):
            if fm.group(1) not in fields:  # don't overwrite int extractions
                fields[fm.group(1)] = fm.group(2)

        # Extract LookbackDelayPolicy: {LookbackDelayAlgorithm::xxx, N, M}
        delay_m = re.search(
            r'LookbackDelayPolicy\s+\w+\s*=\s*\{\s*'
            r'LookbackDelayAlgorithm::(\w+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\}',
            body
        )
        if delay_m:
            fields['delay_algo'] = delay_m.group(1)
            fields['delay_ns'] = int(delay_m.group(2))
            fields['delay_l2w'] = int(delay_m.group(3))

        if fields:
            structs.append((name, fields))

    return structs


def algo_from_filename(filepath):
    """tuning_reduce.cuh → reduce"""
    base = os.path.basename(filepath)
    return base.replace('tuning_', '').replace('.cuh', '')


# --- vllm kernel mapping ---
# Maps (algorithm, struct_field) → (vllm_file, define/variable, context)
# This must be updated when we have access to actual vllm-bi100 source tree.
# For now, these are the known injection points from enginex-vllm-bi100-qwen36.

VLLM_INJECTION_POINTS = {
    # ═══════════════════════════════════════════════════════════════════
    # WARNING: ALL csrc/*.cu targets are DEAD — files do not exist.
    # enginex-vllm-bi100-qwen36 ships: Python + precompiled .so + Triton.
    # No .cu source files. gen_patch patches have zero effect.
    # (Confirmed: commit 41ecb8c, enginex zip analysis)
    # ═══════════════════════════════════════════════════════════════════
    #
    # DEAD injection points (kept for documentation):
    # ('reduce', 'threads'): [('csrc/attention/attention_kernels.cu', 'NUM_THREADS')],
    # ('topk', 'threads'): [('csrc/sampling/sampling_kernels.cu', 'SAMPLING_BLOCK_SIZE')],
    # ('scan', 'threads'): [('csrc/attention/paged_attention_v1.cu', 'SCAN_BLOCK_SIZE')],
    # ('transform', 'threads'): [('csrc/activation_kernels.cu', 'ACTIVATION_BLOCK_SIZE')],
    # ('batch_memcpy', 'threads'): [('csrc/cache_kernels.cu', 'COPY_BLOCK_SIZE')],
    # ('for', 'threads'): [('csrc/pos_encoding_kernels.cu', 'ROPE_BLOCK_SIZE')],
    #
    # ═══════════════════════════════════════════════════════════════════
    # REAL injection points (confirmed working):
    # ═══════════════════════════════════════════════════════════════════
    ('prefill', 'BLOCK_M'): [
        ('prefix_prefill.py', 'BLOCK'),           # Triton JIT tl.constexpr
    ],
    ('prefill', 'NUM_WARPS'): [
        ('prefix_prefill.py', 'NUM_WARPS'),       # Triton JIT
    ],
    ('flash_attn', 'BLOCK_M'): [
        ('vllm/attention/ops/triton_flash_attention.py', 'BLOCK_M'),  # Triton autotune
    ],
    ('flash_attn', 'BLOCK_N'): [
        ('vllm/attention/ops/triton_flash_attention.py', 'BLOCK_N'),  # Triton autotune
    ],
    ('moe', 'BLOCK_SIZE_M'): [
        ('vllm/model_executor/layers/fused_moe/fused_moe.py', 'BLOCK_SIZE_M'),  # → ixformer
    ],
    ('runtime', 'SMEM'): [
        ('vllm/_custom_ops.py', 'get_max_shared_memory'),  # 32KB→48KB fix
    ],
    ('scheduler', 'num_steps'): [
        ('computility-run.yaml', 'num-scheduler-steps'),  # Python dispatch overhead
    ],
}


# --- Complete tuning algorithm registry ---
# All 26 algorithms with muh tuning headers.
# 'injection': algorithms with known vllm kernel injection points
# 'library': algorithms used via CCCL library calls (no direct vllm injection)
# 'struct_mode': 'named' = has bi100_* structs, 'inline' = computes in policy_selector

TUNING_REGISTRY = {
    # === 6 algorithms with vllm injection points (struct_mode='named') ===
    'reduce':       {'mode': 'injection', 'struct_mode': 'named',  'vllm_files': ['csrc/attention/attention_kernels.cu', 'csrc/attention/paged_attention_v2.cu']},
    'scan':         {'mode': 'injection', 'struct_mode': 'named',  'vllm_files': ['csrc/attention/paged_attention_v1.cu']},
    'topk':         {'mode': 'injection', 'struct_mode': 'named',  'vllm_files': ['csrc/sampling/sampling_kernels.cu']},
    'transform':    {'mode': 'injection', 'struct_mode': 'named',  'vllm_files': ['csrc/activation_kernels.cu', 'csrc/layernorm_kernels.cu']},
    'batch_memcpy': {'mode': 'injection', 'struct_mode': 'named',  'vllm_files': ['csrc/cache_kernels.cu']},
    'for':          {'mode': 'injection', 'struct_mode': 'named',  'vllm_files': ['csrc/pos_encoding_kernels.cu']},

    # === 20 algorithms without direct vllm injection (struct_mode='inline') ===
    # These are used via CCCL device-level APIs, not via #define injection.
    # Their tuning values affect performance when vllm calls CUB functions.
    'adjacent_difference':       {'mode': 'library', 'struct_mode': 'inline'},
    'batched_topk':              {'mode': 'library', 'struct_mode': 'inline'},
    'find':                      {'mode': 'library', 'struct_mode': 'inline'},
    'find_bound_sorted_values':  {'mode': 'library', 'struct_mode': 'inline'},
    'histogram':                 {'mode': 'library', 'struct_mode': 'inline'},
    'merge':                     {'mode': 'library', 'struct_mode': 'inline'},
    'merge_sort':                {'mode': 'library', 'struct_mode': 'inline'},
    'radix_sort':                {'mode': 'library', 'struct_mode': 'inline'},
    'reduce_by_key':             {'mode': 'library', 'struct_mode': 'inline'},
    'rle_encode':                {'mode': 'library', 'struct_mode': 'inline'},
    'rle_non_trivial_runs':      {'mode': 'library', 'struct_mode': 'inline'},
    'scan_by_key':               {'mode': 'library', 'struct_mode': 'inline'},
    'segmented_radix_sort':      {'mode': 'library', 'struct_mode': 'inline'},
    'segmented_reduce':          {'mode': 'library', 'struct_mode': 'inline'},
    'segmented_scan':            {'mode': 'library', 'struct_mode': 'inline'},
    'segmented_sort':            {'mode': 'library', 'struct_mode': 'inline'},
    'select_if':                 {'mode': 'library', 'struct_mode': 'inline'},
    'three_way_partition':       {'mode': 'library', 'struct_mode': 'inline'},
    'transform_tile':            {'mode': 'library', 'struct_mode': 'inline'},
    'unique_by_key':             {'mode': 'library', 'struct_mode': 'inline'},
}



def extract_hardcoded_values(filepath):
    """Fallback: extract key values from policy_selector return statements.
    
    For algorithms where bi100_* structs don't exist (values computed inline).
    Handles multiple patterns:
      - topk: return {threads, items, load_algo, scan_algo, bits}
      - transform: constexpr int bi100_bytes_in_flight = N;
      - batch_memcpy: return {threads, items, ...}
      - generic: first integer in return {} is threads_per_block
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    algo = algo_from_filename(filepath)
    
    # --- topk special case: extract bits_per_pass from calc_bits_per_pass ---
    if algo == 'topk':
        # Extract the return statement: return {threads, items, ..., bits};
        iluvatar_match = re.search(
            r'hw\.at_least\(.*iluvatar.*?\)\s*\{(.*?)return\s*\{([^}]+)\}',
            content, re.DOTALL
        )
        if iluvatar_match:
            return_args = iluvatar_match.group(2).strip()
            # Pattern: {512, items, BLOCK_LOAD_VECTORIZE, BLOCK_SCAN_WARP_SCANS, calc_bits_per_pass(key_size)}
            parts = [p.strip() for p in return_args.split(',')]
            fields = {}
            if len(parts) >= 1 and parts[0].isdigit():
                fields['threads'] = int(parts[0])
            # calc_bits_per_pass for float32 (key_size=4) = 11
            bits_match = re.search(r'calc_bits_per_pass', return_args)
            if bits_match:
                fields['bits_per_pass'] = 11  # key_size=4 for float32 logits
            return [('__inline_topk__', fields)]
    
    # --- transform special case: extract bytes_in_flight + thread config ---
    if algo == 'transform':
        fields = {}
        bif_match = re.search(r'bi100_bytes_in_flight\s*=\s*(\d+)', content)
        if bif_match:
            fields['bytes_in_flight'] = int(bif_match.group(1))
        # Look for thread count in vectorized policy or return statement
        vec_threads = re.search(
            r'VectorizedPolicy\s*\{?\s*(\d+)\s*,\s*(\d+)',
            content
        )
        if vec_threads:
            fields['threads'] = int(vec_threads.group(1))
            fields['items'] = int(vec_threads.group(2))
        elif not fields:
            # Fallback: find any constexpr threads
            t_match = re.search(r'threads_per_block\s*=?\s*(\d+)', content)
            if t_match:
                fields['threads'] = int(t_match.group(1))
        if fields:
            return [('__inline_transform__', fields)]
    
    # --- batch_memcpy special case ---
    if algo == 'batch_memcpy':
        fields = {}
        t_match = re.search(r'threads_per_block\s*[=:]\s*(\d+)', content)
        if t_match:
            fields['threads'] = int(t_match.group(1))
        if not fields:
            t_match = re.search(r'return\s*\{?\s*(\d+)', content)
            if t_match:
                fields['threads'] = int(t_match.group(1))
        if fields:
            return [('__inline_batch_memcpy__', fields)]
    
    # --- Generic fallback: find iluvatar branch return value ---
    iluvatar_match = re.search(
        r'hw\.at_least\(.*iluvatar.*?\)\s*\{(.*?)(?=\n\s{2,4}\})',
        content, re.DOTALL
    )
    if not iluvatar_match:
        return []
    
    branch = iluvatar_match.group(1)
    
    # Find return {N, ...} — first integer is typically threads_per_block
    return_match = re.search(r'return\s*\{(\d+)', branch)
    if not return_match:
        return []
    
    threads = int(return_match.group(1))
    return [('__inline__', {'threads': threads})]


def generate_patches(header_dir):
    """Read all tuning headers, extract bi100 values, generate patches."""
    patches = []
    summary = []

    headers = sorted(glob.glob(os.path.join(header_dir, 'tuning_*.cuh')))
    if not headers:
        print(f"ERROR: No tuning_*.cuh found in {header_dir}", file=sys.stderr)
        return [], []

    for hpath in headers:
        algo = algo_from_filename(hpath)
        structs = extract_bi100_structs(hpath)

        if not structs:
            # Fallback: try extracting inline values from policy_selector
            structs = extract_hardcoded_values(hpath)
            if not structs:
                summary.append(f"SKIP {algo}: no bi100_* structs and no inline values found")
                continue

        # Select the struct that matches each vllm kernel's data type.
        #
        # CCCL's policy_selector dispatches by (accum_size, type_t, offset_size).
        # gen_patch must do the same: when injecting into paged_attention
        # (float32 scores), use bi100_plus_float32_o4, not bi100_plus_accum1_o4.
        #
        # The VLLM_KERNEL_MAP in muh_kernel_map.py defines each kernel's
        # data_types. This mapping encodes the primary data type per algorithm:
        ALGO_PRIMARY_TYPE = {
            'reduce':       ('float32', 4),   # paged_attention scores
            'scan':         ('float32', 4),   # softmax denominator
            'topk':         ('float32', 4),   # logits
            'transform':    ('float16', 2),   # activations (SiLU, RMSNorm input)
            'batch_memcpy': ('float16', 2),   # KV cache blocks
            'for':          ('float16', 2),   # RoPE
        }

        target_type, target_size = ALGO_PRIMARY_TYPE.get(algo, ('float32', 4))

        # Score each struct by match quality
        def struct_score(name, fields):
            score = 0
            name_lower = name.lower()
            # Exact type name match (best)
            if target_type.replace('float', 'f') in name_lower or target_type in name_lower:
                score += 100
            # Accum/type size match in name (e.g. "_4B_", "_accum4_", "float32")
            size_tags = [f'_{target_size}B', f'_accum{target_size}', f'float{target_size*8}']
            for tag in size_tags:
                if tag.lower() in name_lower:
                    score += 50
            # Offset size 4 preferred (most common in vllm)
            if '_o4' in name_lower:
                score += 10
            # Penalize 'default' and 'det' (deterministic) structs
            if 'default' in name_lower:
                score -= 200
            if 'det' in name_lower:
                score -= 50
            # Penalize 1-byte type structs for float32 targets
            if target_size >= 4 and ('_1B' in name or 'accum1' in name_lower):
                score -= 100
            return score

        scored = [(struct_score(n, f), n, f) for n, f in structs]
        scored.sort(key=lambda x: -x[0])
        _, pname, pfields = scored[0]

        summary.append(f"READ {algo}: {pname} → {pfields} (target: {target_type})")

        for field_name, value in pfields.items():
            key = (algo, field_name)
            if key not in VLLM_INJECTION_POINTS:
                continue

            for vllm_file, define_name in VLLM_INJECTION_POINTS[key]:
                patch_text = (
                    f"--- a/{vllm_file}\n"
                    f"+++ b/{vllm_file}\n"
                    f"@@ muh tuning injection @@\n"
                    f"-// {define_name}: default\n"
                    f"+#define {define_name} {value}  "
                    f"// muh: from {pname}.{field_name} (tuning_{algo}.cuh)\n"
                )
                patches.append({
                    'algo': algo,
                    'struct': pname,
                    'field': field_name,
                    'value': value,
                    'vllm_file': vllm_file,
                    'define': define_name,
                    'diff': patch_text,
                })
                summary.append(
                    f"  PATCH {vllm_file}: {define_name} = {value} "
                    f"(from {pname}.{field_name})"
                )

    return patches, summary


def write_patches(patches, out_dir):
    """Write combined patch file."""
    os.makedirs(out_dir, exist_ok=True)

    combined = os.path.join(out_dir, 'muh_bi100_tuning.patch')
    with open(combined, 'w') as f:
        f.write(f"# muh kernel tuning patch for Iluvatar BI-V100\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Source: muh/include/muh/tuning/tuning_*.cuh bi100_* structs\n")
        f.write(f"# Patches: {len(patches)}\n\n")
        for p in patches:
            f.write(p['diff'])
            f.write('\n')

    return combined


def main():
    p = argparse.ArgumentParser(description='Generate vllm patches from muh C++ headers')
    p.add_argument('--header-dir', default='muh/include/muh/tuning',
                   help='Directory containing tuning_*.cuh headers')
    p.add_argument('-o', '--output-dir', default='patches',
                   help='Output directory for patches')
    p.add_argument('--dry-run', action='store_true',
                   help='Print to stdout instead of writing')
    args = p.parse_args()

    patches, summary = generate_patches(args.header_dir)

    print(f"muh gen_patch: scanned {args.header_dir}\n")
    for s in summary:
        print(f"  {s}")

    if not patches:
        print("\nNo patches generated.")
        return

    if args.dry_run:
        print(f"\n--- {len(patches)} patches ---\n")
        for p in patches:
            print(p['diff'])
    else:
        combined = write_patches(patches, args.output_dir)
        print(f"\nWritten: {combined}")


if __name__ == '__main__':
    main()
