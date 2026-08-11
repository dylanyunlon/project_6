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
        vllm_root = os.path.join(p, "vllm")
        api = os.path.join(vllm_root, "entrypoints", "openai", "api_server.py")
        if not os.path.isfile(api) or api in patched:
            continue
        
        # --- Entrypoints ---
        for f in ["api_server.py", "cli_args.py", "serving_chat.py", 
                   "protocol.py", "serving_tokenization.py"]:
            src = os.path.join(src_dir, f)
            dst = os.path.join(vllm_root, "entrypoints", "openai", f)
            if os.path.isfile(src):
                shutil.copy2(src, dst)
        # chat_utils
        cu_src = os.path.join(src_dir, "chat_utils.py")
        cu_dst = os.path.join(vllm_root, "entrypoints", "chat_utils.py")
        if os.path.isfile(cu_src):
            shutil.copy2(cu_src, cu_dst)
        # reasoning
        reason_src = os.path.join(src_dir, "reasoning")
        reason_dst = os.path.join(vllm_root, "reasoning")
        if os.path.isdir(reason_src):
            shutil.copytree(reason_src, reason_dst, dirs_exist_ok=True)
        # tool parser
        tp_src = os.path.join(src_dir, "qwen3coder_tool_parser.py")
        tp_dst = os.path.join(vllm_root, "entrypoints", "openai", 
                              "tool_parsers", "qwen3coder_tool_parser.py")
        if os.path.isfile(tp_src) and os.path.isdir(os.path.dirname(tp_dst)):
            shutil.copy2(tp_src, tp_dst)
        
        # --- Model, attention, engine files (critical for runtime) ---
        model_dir = os.path.join(vllm_root, "model_executor", "models")
        attn_dir = os.path.join(vllm_root, "attention", "ops")
        core_dir = os.path.join(vllm_root, "core")
        for fname, dst_dir in [
            ("qwen3_5.py", model_dir),
            ("mamba_cache.py", model_dir),
            ("registry.py", model_dir),
            ("_custom_ops.py", vllm_root),
            ("paged_attn.py", attn_dir),
            ("sequence.py", vllm_root),
            ("scheduler.py", core_dir),
            ("model_runner.py", os.path.join(vllm_root, "worker")),
            ("bi100_env.py", vllm_root),
            ("bi100_profile.py", vllm_root),
            ("block_major_kv_cache.py", vllm_root),
            ("gdn_prefix.py", vllm_root),
            ("logits_processor.py", os.path.join(vllm_root, "model_executor", "layers")),
            ("sampler.py", os.path.join(vllm_root, "model_executor", "layers")),
        ]:
            src = os.path.join(src_dir, fname)
            if os.path.isfile(src) and os.path.isdir(dst_dir):
                shutil.copy2(src, os.path.join(dst_dir, fname))
        
        # --- Prebuilt .so files ---
        prebuilt_dir = os.path.join(src_dir, "prebuilt", "corex-3.2.3-ivcore10")
        if os.path.isdir(prebuilt_dir):
            for so_file in os.listdir(prebuilt_dir):
                if so_file.endswith(".so"):
                    src_so = os.path.join(prebuilt_dir, so_file)
                    dst_so = os.path.join(vllm_root, so_file)
                    if not os.path.isfile(dst_so):
                        shutil.copy2(src_so, dst_so)
        
        # --- Run Python source patches on this vllm install ---
        for patch_script in [
            "patch_xformers_sdpa_seq.py",
            "patch_xformers_profile.py",
            "patch_model_runner.py",
            "patch_vllm_qwen3_5.py",
            "patch_corex_swap_blocks.py",
        ]:
            script_path = os.path.join(src_dir, patch_script)
            if os.path.isfile(script_path):
                try:
                    import subprocess
                    subprocess.run([sys.executable, script_path],
                                   cwd=src_dir, timeout=30,
                                   capture_output=True)
                except Exception:
                    pass
        
        patched.add(api)
    
    if patched:
        print(f"[launch] Force-patched {len(patched)} vllm installs", file=sys.stderr)
    else:
        print("[launch] WARNING: no vllm installs found to patch", file=sys.stderr)

_force_patch()

# execvp replaces this process with vllm api_server, passing all CLI args through.
# This is the safest approach: no import issues, our patched files are already on disk.
print("[launch] Starting vllm api_server with args:", sys.argv[1:], file=sys.stderr)
os.execvp(sys.executable, [
    sys.executable, "-m", "vllm.entrypoints.openai.api_server"
] + sys.argv[1:])
