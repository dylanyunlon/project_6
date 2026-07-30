#!/usr/bin/env python3
"""muh/gen_patch.py — Generate vllm kernel patches from .muh tuning configuration

Given a .muh file with tuning overrides for BI-V100, generates unified diff
patches that can be applied to the vllm source tree to inject optimized
kernel parameters.

The key insight: vllm's CUDA kernels (attention, sampling, layernorm) have
hardcoded launch configs. This script generates patches that replace those
hardcodes with values tuned for Iluvatar BI-V100 via CCCL benchmark data.

Usage:
    python3 muh/gen_patch.py baseline.muh [-o patches/] [--vllm-root /path/to/vllm]
"""

import os
import sys
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))
from parse import load_muh


# --- Kernel location mapping ---
# Maps CCCL algorithm names to vllm source files and the specific
# constants/defines that control kernel launch parameters.

VLLM_KERNEL_MAP = {
    "reduce": {
        "description": "Attention score reduction in multi-head attention",
        "files": [
            "csrc/attention/attention_kernels.cu",
            "csrc/attention/paged_attention_v2.cu",
        ],
        "params": {
            "threads_per_block": {
                "pattern": "NUM_THREADS",
                "default": 128,
                "locations": ["#define NUM_THREADS 128"],
            },
            "items_per_thread": {
                "pattern": "NUM_ITEMS_PER_THREAD",
                "default": 8,
            },
            "vec_size": {
                "pattern": "VEC_SIZE",
                "default": 4,
            },
        },
    },
    "topk": {
        "description": "Top-k / top-p sampling in decode stage",
        "files": [
            "csrc/sampling/sampling_kernels.cu",
        ],
        "params": {
            "threads_per_block": {
                "pattern": "SAMPLING_BLOCK_SIZE",
                "default": 256,
            },
            "bits_per_pass": {
                "pattern": "RADIX_BITS",
                "default": 8,
            },
        },
    },
    "scan": {
        "description": "Prefix scan in paged attention block table lookup",
        "files": [
            "csrc/attention/paged_attention_v1.cu",
        ],
        "params": {
            "threads_per_block": {
                "pattern": "SCAN_BLOCK_SIZE",
                "default": 128,
            },
        },
    },
    "transform": {
        "description": "Elementwise activation kernels (SiLU, GELU, RMSNorm)",
        "files": [
            "csrc/activation_kernels.cu",
            "csrc/layernorm_kernels.cu",
        ],
        "params": {
            "threads_per_block": {
                "pattern": "ACTIVATION_BLOCK_SIZE",
                "default": 512,
            },
        },
    },
    "batch_memcpy": {
        "description": "KV cache block copy between GPU memory regions",
        "files": [
            "csrc/cache_kernels.cu",
        ],
        "params": {
            "threads_per_block": {
                "pattern": "COPY_BLOCK_SIZE",
                "default": 256,
            },
        },
    },
    "for": {
        "description": "Elementwise for-each kernels (position embeddings, rope)",
        "files": [
            "csrc/pos_encoding_kernels.cu",
        ],
        "params": {
            "threads_per_block": {
                "pattern": "ROPE_BLOCK_SIZE",
                "default": 512,
            },
        },
    },
}


def generate_define_patch(algo, param_name, old_value, new_value, define_name, filepath):
    """Generate a unified diff snippet for a #define change."""
    lines = []
    lines.append(f"--- a/{filepath}")
    lines.append(f"+++ b/{filepath}")
    lines.append(f"@@ -1,1 +1,1 @@")
    lines.append(f"-#define {define_name} {old_value}")
    lines.append(f"+#define {define_name} {new_value}  // muh: tuned for BI-V100 ({algo}.{param_name})")
    return "\n".join(lines)


def generate_patches(config, vllm_root=None):
    """Generate all patches from tuning config."""
    tuning = config.get("tuning", {})
    patches = []
    summary = []

    for algo, algo_params in tuning.items():
        if not isinstance(algo_params, dict):
            continue

        mapping = VLLM_KERNEL_MAP.get(algo)
        if mapping is None:
            summary.append(f"SKIP {algo}: no vllm kernel mapping defined")
            continue

        for param_name, new_value in algo_params.items():
            if param_name.startswith("_"):
                continue
            if new_value is None:
                continue

            param_spec = mapping.get("params", {}).get(param_name)
            if param_spec is None:
                continue

            old_value = param_spec.get("default")
            define_name = param_spec.get("pattern", param_name.upper())

            for filepath in mapping.get("files", []):
                patch = generate_define_patch(
                    algo, param_name, old_value, new_value, define_name, filepath
                )
                patches.append({
                    "algo": algo,
                    "param": param_name,
                    "file": filepath,
                    "old": old_value,
                    "new": new_value,
                    "diff": patch,
                })
                summary.append(
                    f"PATCH {filepath}: {define_name} {old_value} → {new_value} "
                    f"(from {algo}.{param_name})"
                )

    return patches, summary


def write_patches(patches, out_dir):
    """Write patches to individual .patch files."""
    os.makedirs(out_dir, exist_ok=True)

    # Combined patch
    combined_path = os.path.join(out_dir, "muh_bi100_tuning.patch")
    with open(combined_path, 'w') as f:
        f.write(f"# muh kernel tuning patch for Iluvatar BI-V100\n")
        f.write(f"# Generated: {datetime.now().isoformat()}\n")
        f.write(f"# Algorithms patched: {len(set(p['algo'] for p in patches))}\n")
        f.write(f"# Total changes: {len(patches)}\n\n")
        for p in patches:
            f.write(p["diff"])
            f.write("\n\n")

    # Per-algorithm patches
    by_algo = {}
    for p in patches:
        by_algo.setdefault(p["algo"], []).append(p)

    for algo, algo_patches in by_algo.items():
        algo_path = os.path.join(out_dir, f"{algo}.patch")
        with open(algo_path, 'w') as f:
            f.write(f"# muh tuning patch: {algo} for BI-V100\n\n")
            for p in algo_patches:
                f.write(p["diff"])
                f.write("\n\n")

    return combined_path


def main():
    parser = argparse.ArgumentParser(description="Generate vllm kernel patches from .muh")
    parser.add_argument("muh_file", help="Path to .muh file")
    parser.add_argument("-o", "--output-dir", default="patches",
                        help="Output directory for patches (default: patches)")
    parser.add_argument("--vllm-root", default=None,
                        help="Path to vllm source tree (for verification)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print patches to stdout instead of writing files")
    args = parser.parse_args()

    config = load_muh(args.muh_file)
    patches, summary = generate_patches(config, args.vllm_root)

    print(f"muh gen_patch: {len(patches)} patches from {args.muh_file}\n")
    for s in summary:
        print(f"  {s}")

    if not patches:
        print("\nNo patches generated. Add tuning overrides to your .muh file.")
        return

    if args.dry_run:
        print("\n--- Patches ---\n")
        for p in patches:
            print(p["diff"])
            print()
    else:
        combined = write_patches(patches, args.output_dir)
        print(f"\nWritten to {args.output_dir}/")
        print(f"Combined: {combined}")


if __name__ == "__main__":
    main()
