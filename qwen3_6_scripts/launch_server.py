#!/usr/bin/env python3
"""launch_server.py — Ensure our patched api_server.py runs, not the base image's.

Patches the RUNNING vllm install's api_server.py/cli_args.py in-place before
importing, then delegates to the standard vllm api_server main().
"""
import os, sys, shutil

def _force_patch():
    """Copy our files over ALL vllm installs found on sys.path."""
    src_dir = os.path.dirname(os.path.abspath(__file__))
    patched = set()
    
    for p in sys.path:
        api = os.path.join(p, "vllm", "entrypoints", "openai", "api_server.py")
        if os.path.isfile(api) and api not in patched:
            for f in ["api_server.py", "cli_args.py", "serving_chat.py", 
                       "protocol.py", "serving_tokenization.py"]:
                src = os.path.join(src_dir, f)
                dst = os.path.join(p, "vllm", "entrypoints", "openai", f)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
            # chat_utils
            cu_src = os.path.join(src_dir, "chat_utils.py")
            cu_dst = os.path.join(p, "vllm", "entrypoints", "chat_utils.py")
            if os.path.isfile(cu_src):
                shutil.copy2(cu_src, cu_dst)
            # reasoning
            reason_src = os.path.join(src_dir, "reasoning")
            reason_dst = os.path.join(p, "vllm", "reasoning")
            if os.path.isdir(reason_src):
                shutil.copytree(reason_src, reason_dst, dirs_exist_ok=True)
            # tool parser
            tp_src = os.path.join(src_dir, "qwen3coder_tool_parser.py")
            tp_dst = os.path.join(p, "vllm", "entrypoints", "openai", 
                                  "tool_parsers", "qwen3coder_tool_parser.py")
            if os.path.isfile(tp_src) and os.path.isdir(os.path.dirname(tp_dst)):
                shutil.copy2(tp_src, tp_dst)
            patched.add(api)
    
    if patched:
        print(f"[launch] Force-patched {len(patched)} vllm installs", file=sys.stderr)
    else:
        print("[launch] WARNING: no vllm installs found to patch", file=sys.stderr)

_force_patch()

# Now run the standard vllm api_server with our patched version
from vllm.entrypoints.openai.api_server import *
