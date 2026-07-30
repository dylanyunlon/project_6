#!/usr/bin/env python3
"""muh/extract.py — Extract CCCL tuning parameter spaces into muh/schema/*.yaml

Reads cccl_upstream/cub/cub/device/dispatch/tuning/tuning_*.cuh,
parses policy struct fields and SM-specific tuning values,
outputs one YAML file per algorithm under muh/schema/.

Usage:
    python3 muh/extract.py [--cccl-root cccl_upstream] [--out-dir muh/schema]
"""

import re
import os
import sys
import glob
import argparse
from pathlib import Path
from collections import OrderedDict

# --- Enum value sets (from CCCL headers) ---

BLOCK_LOAD_ALGORITHMS = [
    "BLOCK_LOAD_DIRECT",
    "BLOCK_LOAD_VECTORIZE",
    "BLOCK_LOAD_TRANSPOSE",
    "BLOCK_LOAD_WARP_TRANSPOSE",
    "BLOCK_LOAD_WARP_TRANSPOSE_TIMESLICED",
    "BLOCK_LOAD_STRIPED",
]

BLOCK_STORE_ALGORITHMS = [
    "BLOCK_STORE_DIRECT",
    "BLOCK_STORE_WARP_TRANSPOSE",
    "BLOCK_STORE_WARP_TRANSPOSE_TIMESLICED",
    "BLOCK_STORE_STRIPED",
]

BLOCK_REDUCE_ALGORITHMS = [
    "BLOCK_REDUCE_RAKING",
    "BLOCK_REDUCE_RAKING_COMMUTATIVE_ONLY",
    "BLOCK_REDUCE_WARP_REDUCTIONS",
]

BLOCK_SCAN_ALGORITHMS = [
    "BLOCK_SCAN_RAKING",
    "BLOCK_SCAN_RAKING_MEMOIZE",
    "BLOCK_SCAN_WARP_SCANS",
]

CACHE_LOAD_MODIFIERS = [
    "LOAD_DEFAULT",
    "LOAD_CA",
    "LOAD_CG",
    "LOAD_CS",
    "LOAD_CV",
    "LOAD_LDG",
]

LOOKBACK_DELAY_ALGORITHMS = [
    "no_delay",
    "fixed_delay",
    "exponential_backoff",
    "exponential_backoff_jitter",
    "exponential_backoff_jitter_window",
    "exponential_backon_jitter_window",
    "exponential_backon_jitter",
    "exponential_backon",
]

# --- Field type → range/enum mapping ---

FIELD_TYPES = {
    "threads_per_block": {"type": "int", "range": [32, 1024], "step": 32},
    "items_per_thread": {"type": "int", "range": [1, 32], "step": 1},
    "vec_size": {"type": "int", "range": [1, 8], "step": 1},
    "bits_per_pass": {"type": "int", "range": [4, 11], "step": 1},
    "radix_bits": {"type": "int", "range": [4, 8], "step": 1},
    "load_algorithm": {"type": "enum", "values": BLOCK_LOAD_ALGORITHMS},
    "store_algorithm": {"type": "enum", "values": BLOCK_STORE_ALGORITHMS},
    "reduce_algorithm": {"type": "enum", "values": BLOCK_REDUCE_ALGORITHMS},
    "scan_algorithm": {"type": "enum", "values": BLOCK_SCAN_ALGORITHMS},
    "load_modifier": {"type": "enum", "values": CACHE_LOAD_MODIFIERS},
    "lookback_delay.kind": {"type": "enum", "values": LOOKBACK_DELAY_ALGORITHMS},
    "lookback_delay.delay": {"type": "int", "range": [0, 2000], "step": 50},
    "lookback_delay.l2_write_latency": {"type": "int", "range": [0, 2000], "step": 50},
    "reduce_and_scan_warps": {"type": "int", "range": [1, 8], "step": 1},
    "lookahead_items_per_thread": {"type": "int", "range": [1, 16], "step": 1},
}


def extract_policy_fields(content, filename):
    """Extract policy struct field names from a tuning file."""
    fields = []
    # Match lines like: int threads_per_block; or BlockLoadAlgorithm load_algorithm;
    pattern = re.compile(
        r'^\s+(?:int|BlockLoadAlgorithm|BlockStoreAlgorithm|BlockReduceAlgorithm|'
        r'BlockScanAlgorithm|CacheLoadModifier|LookbackDelayPolicy)\s+'
        r'(\w+)\s*[;=]',
        re.MULTILINE
    )
    for m in pattern.finditer(content):
        field = m.group(1)
        if field not in fields:
            fields.append(field)
    return fields


def extract_sm_tunings(content):
    """Extract SM-specific tuning values from static constexpr definitions."""
    tunings = {}
    # Match patterns like: sm80_tuning, sm90_tuning, sm100_tuning
    sm_pattern = re.compile(r'struct\s+sm(\d+)_tuning')
    for m in sm_pattern.finditer(content):
        sm = int(m.group(1))
        if sm not in tunings:
            tunings[sm] = []

    # Extract actual parameter values from constexpr definitions
    # Pattern: static constexpr int threads = 512;
    blocks = re.split(r'(?=struct\s+sm\d+_tuning)', content)
    for block in blocks:
        sm_m = re.match(r'struct\s+sm(\d+)_tuning', block)
        if not sm_m:
            continue
        sm = int(sm_m.group(1))
        vals = {}
        for line in block.split('\n'):
            # int values
            m = re.search(r'static\s+constexpr\s+int\s+(\w+)\s*=\s*(\d+)', line)
            if m:
                vals[m.group(1)] = int(m.group(2))
            # enum values
            m = re.search(r'static\s+constexpr\s+(?:BlockLoadAlgorithm|BlockStoreAlgorithm|CacheLoadModifier)\s+(\w+)\s*=\s*(\w+)', line)
            if m:
                vals[m.group(1)] = m.group(2)
        if vals:
            tunings.setdefault(sm, []).append(vals)

    return tunings


def extract_inline_tunings(content):
    """Extract inline tuning values from make_mem_scaled_lookback_scan_policy calls and similar."""
    inline = []
    # Pattern: threads_per_block, items_per_thread in constructor-style calls
    pattern = re.compile(
        r'(?:topk_policy|ReducePassPolicy|ScanLookbackPolicy)\s*\{'
        r'\s*(\d+)\s*,\s*(\d+)',
        re.MULTILINE
    )
    for m in pattern.finditer(content):
        inline.append({
            "threads_per_block": int(m.group(1)),
            "items_per_thread": int(m.group(2)),
        })
    return inline


def algo_name_from_filename(filename):
    """tuning_topk.cuh → topk"""
    base = os.path.basename(filename)
    return base.replace("tuning_", "").replace(".cuh", "")


def build_schema(algo, fields, sm_tunings, inline_tunings):
    """Build YAML-serializable schema dict for one algorithm."""
    schema = OrderedDict()
    schema["algorithm"] = algo
    schema["source"] = f"cub/cub/device/dispatch/tuning/tuning_{algo}.cuh"

    # Parameter space
    params = OrderedDict()
    for field in fields:
        if field in FIELD_TYPES:
            params[field] = dict(FIELD_TYPES[field])
        elif field == "lookback_delay":
            # Expand to sub-fields
            for sub in ["lookback_delay.kind", "lookback_delay.delay", "lookback_delay.l2_write_latency"]:
                params[sub] = dict(FIELD_TYPES[sub])
        else:
            params[field] = {"type": "int", "range": [1, 1024], "step": 1, "note": "unknown_range"}
    schema["parameters"] = dict(params)

    # Known SM tunings (for reference when tuning BI-V100)
    if sm_tunings:
        ref = OrderedDict()
        for sm, vals_list in sorted(sm_tunings.items()):
            ref[f"sm{sm}"] = vals_list
        schema["reference_tunings"] = dict(ref)

    # BI-V100 placeholder
    schema["bi_v100"] = {
        "status": "pending_benchmark",
        "note": "Run muh benchmark on Iluvatar BI-V100 to fill these values",
        "threads_per_block": "TBD",
        "items_per_thread": "TBD",
    }

    return schema


def yaml_dump(data, indent=0):
    """Simple YAML serializer (no dependency on pyyaml)."""
    lines = []
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(yaml_dump(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {v}")
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                lines.append(f"{prefix}-")
                lines.append(yaml_dump(item, indent + 1))
            else:
                lines.append(f"{prefix}- {item}")
    else:
        lines.append(f"{prefix}{data}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Extract CCCL tuning params to muh schema")
    parser.add_argument("--cccl-root", default="cccl_upstream",
                        help="Path to CCCL root (default: cccl_upstream)")
    parser.add_argument("--out-dir", default="muh/schema",
                        help="Output directory for YAML schemas (default: muh/schema)")
    args = parser.parse_args()

    tuning_dir = os.path.join(args.cccl_root, "cub", "cub", "device", "dispatch", "tuning")
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    tuning_files = sorted(glob.glob(os.path.join(tuning_dir, "tuning_*.cuh")))
    if not tuning_files:
        print(f"ERROR: No tuning_*.cuh files found in {tuning_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(tuning_files)} tuning files in {tuning_dir}")

    all_algos = []
    for filepath in tuning_files:
        if os.path.basename(filepath) == "common.cuh":
            continue

        with open(filepath, "r") as f:
            content = f.read()

        algo = algo_name_from_filename(filepath)
        fields = extract_policy_fields(content, filepath)
        sm_tunings = extract_sm_tunings(content)
        inline_tunings = extract_inline_tunings(content)

        schema = build_schema(algo, fields, sm_tunings, inline_tunings)

        out_path = os.path.join(out_dir, f"{algo}.yaml")
        with open(out_path, "w") as f:
            f.write(f"# muh schema for {algo}\n")
            f.write(f"# Auto-extracted from {schema['source']}\n")
            f.write(f"# Generated by muh/extract.py\n\n")
            f.write(yaml_dump(dict(schema)))
            f.write("\n")

        all_algos.append(algo)
        print(f"  {algo}: {len(fields)} params, {len(sm_tunings)} SM tunings → {out_path}")

    # Write index
    index_path = os.path.join(out_dir, "_index.yaml")
    with open(index_path, "w") as f:
        f.write("# muh schema index — all extracted CCCL tuning algorithms\n\n")
        f.write("algorithms:\n")
        for algo in all_algos:
            f.write(f"  - {algo}\n")
        f.write(f"\ntotal: {len(all_algos)}\n")
        f.write(f"source: cccl_upstream/cub/cub/device/dispatch/tuning/\n")

    print(f"\nDone: {len(all_algos)} schemas → {out_dir}/")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
