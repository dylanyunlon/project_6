#!/usr/bin/env python3
"""muh/gen_yaml.py — Generate computility-run.yaml from a .muh configuration

Reads a .muh file (via parse.py), extracts the 'vllm' and 'env' sections,
and outputs a computility-run.yaml compatible with ModelHub XC platform.

Usage:
    python3 muh/gen_yaml.py baseline.muh [-o computility-run.yaml]
"""

import os
import sys
import argparse

# Import from sibling module
sys.path.insert(0, os.path.dirname(__file__))
from parse import load_muh


# Mapping from .muh vllm config keys to CLI arguments
VLLM_ARG_MAP = {
    "model_path": "--model",
    "served_model_name": "--served-model-name",
    "max_model_len": "--max-model-len",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "tensor_parallel": "-tp",
    "max_num_seqs": "--max-num-seqs",
    "max_num_batched_tokens": "--max-num-batched-tokens",
    "max_seq_len_to_capture": "--max-seq-len-to-capture",
    "tool_call_parser": "--tool-call-parser",
    "reasoning_parser": "--reasoning-parser",
}

# Boolean flags (presence = enabled, no value needed)
VLLM_FLAG_MAP = {
    "trust_remote_code": "--trust-remote-code",
    "disable_log_requests": "--disable-log-requests",
    "disable_frontend_multiprocessing": "--disable-frontend-multiprocessing",
    "enable_chunked_prefill": "--enable-chunked-prefill",
    "enable_auto_tool_choice": "--enable-auto-tool-choice",
    "enable_prefix_caching": "--enable-prefix-caching",
}


def build_command(vllm_config):
    """Build the command list for computility-run.yaml from vllm config."""
    cmd = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
    ]

    # Model path (required)
    model_path = vllm_config.get("model_path", "/model")
    cmd.extend(["--model", model_path])

    # Named arguments
    for muh_key, cli_arg in VLLM_ARG_MAP.items():
        if muh_key == "model_path":
            continue  # already handled
        val = vllm_config.get(muh_key)
        if val is not None:
            cmd.extend([cli_arg, str(val)])

    # Boolean flags
    for muh_key, cli_flag in VLLM_FLAG_MAP.items():
        if vllm_config.get(muh_key, False):
            cmd.append(cli_flag)

    # Extra raw args (pass-through)
    extra = vllm_config.get("extra_args", [])
    if isinstance(extra, list):
        cmd.extend([str(a) for a in extra])

    return cmd


def build_env(env_config):
    """Build environment variable list."""
    env_list = []
    for name, value in env_config.items():
        env_list.append({"name": name, "value": str(value)})
    return env_list


def yaml_serialize_computility(concurrency, command, env_list):
    """Serialize to computility-run.yaml format (no pyyaml dependency)."""
    lines = []
    lines.append(f"concurrency: {concurrency}")
    lines.append("command:")
    for item in command:
        lines.append(f"    - '{item}'" if ' ' in str(item) else f"    - {item}")
    if env_list:
        lines.append("env:")
        for e in env_list:
            lines.append(f"    - name: {e['name']}")
            lines.append(f"      value: {e['value']}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Generate computility-run.yaml from .muh")
    parser.add_argument("muh_file", help="Path to .muh file")
    parser.add_argument("-o", "--output", default="computility-run.yaml",
                        help="Output file path (default: computility-run.yaml)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print to stdout instead of writing file")
    args = parser.parse_args()

    config = load_muh(args.muh_file)

    vllm_config = config.get("vllm", {})
    env_config = config.get("env", {})
    concurrency = config.get("concurrency", 1)

    command = build_command(vllm_config)
    env_list = build_env(env_config)

    output = yaml_serialize_computility(concurrency, command, env_list)

    if args.dry_run:
        print(output)
    else:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Generated {args.output} ({len(command)} command args, {len(env_list)} env vars)")


if __name__ == "__main__":
    main()
