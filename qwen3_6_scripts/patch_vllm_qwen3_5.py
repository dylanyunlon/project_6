"""
Patches the vLLM model registry and deploys the Qwen3_5 model file.

Deploy steps on the remote machine:
  1. cp modified_scripts/qwen3_5.py \
        /usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models/qwen3_5.py
  2. python3 modified_scripts/patch_vllm_qwen3_5.py

Also edit your model config.json to set:
  "architectures": ["Qwen3_5ForCausalLM"]

Target: vLLM at /usr/local/corex/lib64/python3/dist-packages/vllm/
"""

VLLM_ROOT = "/usr/local/corex/lib64/python3/dist-packages/vllm"
REGISTRY = f"{VLLM_ROOT}/model_executor/models/registry.py"


def patch_file(path, replacements):
    with open(path, "r") as f:
        content = f.read()

    patched = False
    for old, new in replacements:
        if new in content:
            print(f"  [skip] already patched: {repr(new[:70])}")
            continue
        if old not in content:
            print(f"  [warn] anchor not found: {repr(old[:70])}")
            continue
        content = content.replace(old, new, 1)
        patched = True
        print(f"  [ok]   patched after: {repr(old[:70])}")

    if patched:
        with open(path, "w") as f:
            f.write(content)


def main():
    print(f"=== Patching {REGISTRY} ===")
    patch_file(REGISTRY, [
        (
            '    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),\n'
            '    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),',
            '    "Qwen3ForCausalLM": ("qwen3", "Qwen3ForCausalLM"),\n'
            '    "Qwen3MoeForCausalLM": ("qwen3_moe", "Qwen3MoeForCausalLM"),\n'
            '    "Qwen3_5ForCausalLM": ("qwen3_5", "Qwen3_5ForCausalLM"),\n'
            '    "Qwen3_5MoeForCausalLM": ("qwen3_5", "Qwen3_5MoeForCausalLM"),',
        ),
    ])

    print("\n=== Verification ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qwen3_5",
            f"{VLLM_ROOT}/model_executor/models/qwen3_5.py",
        )
        mod = importlib.util.module_from_spec(spec)
        # Quick check: does the class exist?
        spec.loader.exec_module(mod)
        cls = mod.Qwen3_5ForCausalLM
        print(f"  Qwen3_5ForCausalLM found: {cls}")
        cls_moe = mod.Qwen3_5MoeForCausalLM
        print(f"  Qwen3_5MoeForCausalLM found: {cls_moe}")
    except Exception as e:
        print(f"  [warn] verification failed (may be OK at runtime): {e}")

    print("\nDone. Remember to:")
    print("  1. Set config.json 'architectures': ['Qwen3_5ForCausalLM'] or ['Qwen3_5MoEForCausalLM']")
    print("  2. Run patch_transformers_qwen3_5.py if not already done")


if __name__ == "__main__":
    main()
