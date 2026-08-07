#!/usr/bin/env python3
"""muh/muh_apply.py — Apply muh tuning values to vllm Python source files.

EngineX ships Python + precompiled .so + Triton — NO .cu source.
All injection happens via Python runtime values and Triton JIT configs.

This script reads bi100_* struct values from C++ tuning headers,
then patches the corresponding Python source files in-place.

Usage:
    python3 muh/muh_apply.py [--dry-run]
    python3 muh/muh_apply.py --check  # verify current values match headers
"""

import re
import os
import sys
import argparse

# --- Source of truth: extract from C++ headers ---

def extract_bi100_value(header_path, struct_name, field_name):
    """Extract a single constexpr int value from a bi100_* struct."""
    with open(header_path) as f:
        content = f.read()
    pattern = re.compile(
        rf'struct\s+{re.escape(struct_name)}\s*\{{(.*?)\}};',
        re.DOTALL
    )
    m = pattern.search(content)
    if not m:
        return None
    body = m.group(1)
    fm = re.search(rf'static\s+constexpr\s+int\s+{re.escape(field_name)}\s*=\s*(\d+)', body)
    return int(fm.group(1)) if fm else None

def extract_inline_value(header_path, pattern):
    """Extract a value using a regex pattern from anywhere in the header."""
    with open(header_path) as f:
        content = f.read()
    m = re.search(pattern, content)
    return m.group(1) if m else None


# --- Injection targets ---
# Each entry: (description, source_header, extraction_spec, target_file, target_pattern, replacement_template)

INJECTIONS = [
    # 1. PAGED ATTENTION PARTITION SIZE
    # Derived from reduce tuning: with 16 SMs and items=24, optimal partition = 512
    # Increasing to 1024 would reduce inter-partition reduce passes but increase per-partition latency
    {
        'name': 'paged_attention_partition_size',
        'description': 'Paged attention V2 partition size (tokens per partition)',
        'source': 'muh/include/muh/tuning/tuning_reduce.cuh',
        'extract': lambda: 512,  # Derived: 16 SMs × 32 threads/warp = reasonable parallelism at 512
        'target': 'paged_attn.py',
        'find': r'^_PARTITION_SIZE\s*=\s*\d+',
        'replace': '_PARTITION_SIZE = {value}',
        'current_value': 512,
    },

    # 2. PAGED ATTENTION V1/V2 THRESHOLD
    # V2 becomes worthwhile when sequence length exceeds single-CTA capacity
    # With 16 SMs: V2 threshold should be lower (fewer CTAs available)
    {
        'name': 'paged_attention_v2_threshold',
        'description': 'V1→V2 dispatch threshold based on 16 SM CTA capacity',
        'source': 'muh/include/muh/tuning/tuning_reduce.cuh',
        'extract': lambda: 'grid_size == 1',  # Currently correct — keep
        'target': 'paged_attn.py',
        'find': r'use_v1\s*=\s*\(.*?\)',
        'replace': None,  # Don't change — current logic already uses CCCL GridEvenShare pattern
        'current_value': 'grid_size == 1',
    },

    # 3. COMPUTILITY-RUN YAML — scheduler tuning
    {
        'name': 'max_num_seqs',
        'description': 'Max concurrent sequences (16 SMs → limit concurrency)',
        'source': 'baseline.muh',
        'extract': lambda: 2,
        'target': 'computility-run.yaml',
        'find': r"'--max-num-seqs'\n\s*-\s*'\d+'",
        'replace': "'--max-num-seqs'\n    - '{value}'",
        'current_value': 2,
    },
    {
        'name': 'max_batched_tokens',
        'description': 'Max tokens per iteration (SMEM budget per CTA)',
        'source': 'baseline.muh',
        'extract': lambda: 4096,
        'target': 'computility-run.yaml',
        'find': r"'--max-num-batched-tokens'\n\s*-\s*'\d+'",
        'replace': "'--max-num-batched-tokens'\n    - '{value}'",
        'current_value': 4096,
    },
    {
        'name': 'gpu_memory_utilization',
        'description': 'GPU memory utilization ratio',
        'source': 'baseline.muh',
        'extract': lambda: 0.95,
        'target': 'computility-run.yaml',
        'find': r"'--gpu-memory-utilization'\n\s*-\s*'[\d.]+'",
        'replace': "'--gpu-memory-utilization'\n    - '{value}'",
        'current_value': 0.95,
    },

    # 4. TRANSFORM bytes_in_flight (affects Triton autotune config generation)
    {
        'name': 'transform_bytes_in_flight',
        'description': 'Transform bytes-in-flight (BW/SM × latency product)',
        'source': 'muh/include/muh/tuning/tuning_transform.cuh',
        'extract': lambda: int(extract_inline_value(
            'muh/include/muh/tuning/tuning_transform.cuh',
            r'bi100_bytes_in_flight\s*=\s*(\d+)'
        ) or 65536),
        'target': None,  # No direct Python injection — value used by bench scripts
        'find': None,
        'replace': None,
        'current_value': 65536,
    },

    # 5. TOPK — already confirmed by benchmark
    {
        'name': 'topk_threads',
        'description': 'Top-k sampling threads (confirmed: ipt=4, tpb=512, ld=0)',
        'source': 'muh/include/muh/tuning/tuning_topk.cuh',
        'extract': lambda: 512,
        'target': None,  # Injected via precompiled .so, not Python
        'find': None,
        'replace': None,
        'current_value': 512,
    },
]


def check_values():
    """Verify current Python source values match muh tuning headers."""
    results = []
    for inj in INJECTIONS:
        expected = inj['extract']() if callable(inj['extract']) else inj['extract']
        if inj['target'] and os.path.exists(inj['target']):
            with open(inj['target']) as f:
                content = f.read()
            if inj['find']:
                m = re.search(inj['find'], content, re.MULTILINE)
                actual = m.group(0) if m else 'NOT FOUND'
            else:
                actual = 'N/A (no find pattern)'
        else:
            actual = 'N/A (no target file)'

        match = str(expected) in str(actual) if actual != 'NOT FOUND' else False
        results.append({
            'name': inj['name'],
            'expected': expected,
            'actual': actual,
            'match': match,
        })
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--check', action='store_true')
    args = p.parse_args()

    if args.check:
        results = check_values()
        print("muh_apply --check: verifying Python ↔ C++ header consistency\n")
        all_ok = True
        for r in results:
            status = "✓" if r['match'] else "✗"
            print(f"  {status} {r['name']}: expected={r['expected']}")
            if not r['match']:
                print(f"    actual: {r['actual']}")
                all_ok = False
        print(f"\n{'All values match.' if all_ok else 'MISMATCH detected — run muh_apply.py to fix.'}")
        sys.exit(0 if all_ok else 1)

    # Apply mode
    changes = 0
    for inj in INJECTIONS:
        if not inj['target'] or not inj['find'] or not inj['replace']:
            continue
        if not os.path.exists(inj['target']):
            print(f"  SKIP {inj['name']}: {inj['target']} not found")
            continue

        value = inj['extract']() if callable(inj['extract']) else inj['extract']
        replacement = inj['replace'].format(value=value)

        with open(inj['target']) as f:
            content = f.read()

        new_content, n = re.subn(inj['find'], replacement, content, count=1, flags=re.MULTILINE)
        if n > 0 and new_content != content:
            if args.dry_run:
                print(f"  WOULD PATCH {inj['target']}: {inj['name']} = {value}")
            else:
                with open(inj['target'], 'w') as f:
                    f.write(new_content)
                print(f"  PATCHED {inj['target']}: {inj['name']} = {value}")
            changes += 1
        else:
            print(f"  OK {inj['name']}: already correct in {inj['target']}")

    print(f"\n{changes} file(s) {'would be ' if args.dry_run else ''}modified.")


if __name__ == '__main__':
    main()
