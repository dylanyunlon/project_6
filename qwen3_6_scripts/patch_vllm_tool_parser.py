"""
Patches vLLM 0.6.3 to register Qwen3CoderToolParser under the name "qwen3_coder".

Deploy steps on the remote machine (already called by patch_ops.sh):
  1. cp qwen3coder_tool_parser.py \
        /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/openai/tool_parsers/
  2. python3 patch_vllm_tool_parser.py

Usage after patching:
  --tool-call-parser qwen3_coder --enable-auto-tool-choice
"""

import os

VLLM_ROOT = "/usr/local/corex/lib/python3/dist-packages/vllm"
TOOL_PARSERS_DIR = f"{VLLM_ROOT}/entrypoints/openai/tool_parsers"
INIT_FILE = f"{TOOL_PARSERS_DIR}/__init__.py"


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
        print(f"  [ok]   patched: {repr(old[:50])} -> {repr(new[:50])}")

    if patched:
        with open(path, "w") as f:
            f.write(content)


def main():
    if not os.path.isdir(TOOL_PARSERS_DIR):
        raise FileNotFoundError(
            f"Tool parsers directory not found: {TOOL_PARSERS_DIR}\n"
            "Verify the vLLM installation path.")

    print(f"=== Patching {INIT_FILE} ===")
    patch_file(INIT_FILE, [
        (
            "from .mistral_tool_parser import MistralToolParser",
            "from .mistral_tool_parser import MistralToolParser\n"
            "from .qwen3coder_tool_parser import Qwen3CoderToolParser",
        ),
        (
            '"MistralToolParser", "Internlm2ToolParser", "Llama3JsonToolParser"\n]',
            '"MistralToolParser", "Internlm2ToolParser", "Llama3JsonToolParser",\n'
            '    "Qwen3CoderToolParser"\n]',
        ),
    ])

    print("\n=== Verification ===")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "qwen3coder_tool_parser",
            f"{TOOL_PARSERS_DIR}/qwen3coder_tool_parser.py",
        )
        mod = importlib.util.module_from_spec(spec)
        print(f"  Module spec loaded: {spec.name}")
        print("  (full import requires torch/vllm runtime — skipping exec)")
    except Exception as e:
        print(f"  [warn] spec check failed: {e}")

    print("\nDone. Start vLLM server with:")
    print("  --tool-call-parser qwen3_coder --enable-auto-tool-choice")


if __name__ == "__main__":
    main()
