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
    ('reduce', 'threads'): [
        ('csrc/attention/attention_kernels.cu', 'NUM_THREADS'),
        ('csrc/attention/paged_attention_v2.cu', 'NUM_THREADS'),
    ],
    ('reduce', 'items'): [
        ('csrc/attention/attention_kernels.cu', 'NUM_ITEMS_PER_THREAD'),
    ],
    ('reduce', 'items_per_vec_load'): [
        ('csrc/attention/attention_kernels.cu', 'VEC_SIZE'),
    ],
    ('topk', 'threads'): [
        ('csrc/sampling/sampling_kernels.cu', 'SAMPLING_BLOCK_SIZE'),
    ],
    ('topk', 'bits_per_pass'): [
        ('csrc/sampling/sampling_kernels.cu', 'RADIX_BITS'),
    ],
    ('scan', 'threads'): [
        ('csrc/attention/paged_attention_v1.cu', 'SCAN_BLOCK_SIZE'),
    ],
    ('transform', 'threads'): [
        ('csrc/activation_kernels.cu', 'ACTIVATION_BLOCK_SIZE'),
        ('csrc/layernorm_kernels.cu', 'LAYERNORM_BLOCK_SIZE'),
    ],
    ('batch_memcpy', 'threads'): [
        ('csrc/cache_kernels.cu', 'COPY_BLOCK_SIZE'),
    ],
    ('for', 'threads'): [
        ('csrc/pos_encoding_kernels.cu', 'ROPE_BLOCK_SIZE'),
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
    Extracts the threads_per_block from the first return in the iluvatar branch.
    """
    with open(filepath, 'r') as f:
        content = f.read()
    
    algo = algo_from_filename(filepath)
    
    # Find iluvatar branch
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

        # Use the first non-default struct as the primary tuning
        # (default is fallback; prefer the type-specific ones)
        primary = None
        for name, fields in structs:
            if 'default' not in name:
                primary = (name, fields)
                break
        if primary is None:
            primary = structs[0]

        pname, pfields = primary
        summary.append(f"READ {algo}: {pname} → {pfields}")

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
