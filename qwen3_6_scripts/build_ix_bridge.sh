#!/bin/bash
# Build ix_full_bridge.so — bridges ixformer::infer C++ symbols to Python
# Compiled via torch.utils.cpp_extension at docker build time
# Runtime: dlopen links against libixformer.so in base image

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VLLM_ROOT="${1:?usage: build_ix_bridge.sh VLLM_ROOT}"
BRIDGE_SRC="${SCRIPT_DIR}/ix_full_bridge.cpp"
MOE_SRC="${SCRIPT_DIR}/ix_moe_bridge.cpp"

if [ ! -f "$BRIDGE_SRC" ]; then
    echo "[bridge] SKIP: $BRIDGE_SRC not found"
    exit 0
fi

# Find ixformer .so directory
IX_LIB=""
for d in /usr/local/corex/lib/python3/dist-packages/ixformer \
         /usr/local/corex/lib64 \
         /usr/lib/python3/dist-packages/ixformer; do
    if [ -d "$d" ]; then
        IX_LIB="$d"
        break
    fi
done

# Find torch library path
TORCH_LIB=$(python3 -c "import torch; print(torch.__path__[0] + '/lib')" 2>/dev/null)

echo "[bridge] Building ix_full_bridge via torch.utils.cpp_extension..."
echo "[bridge] ixformer lib: ${IX_LIB:-not found}"
echo "[bridge] torch lib: ${TORCH_LIB:-not found}"

python3 << PYEOF
import torch
from torch.utils.cpp_extension import load
import os, shutil

# Extra link flags: find libixformer.so and link
extra_ldflags = []
ix_lib = "${IX_LIB}"
torch_lib = "${TORCH_LIB}"

# Search for libixformer.so
for d in [ix_lib, "/usr/local/corex/lib64", "/usr/local/corex/lib"]:
    if d and os.path.exists(os.path.join(d, "libixformer.so")):
        extra_ldflags.extend([f"-L{d}", "-lixformer"])
        break
else:
    # No explicit libixformer.so — symbols may be in already-loaded .so
    # (ixformer is imported as Python module which loads the .so)
    try:
        import ixformer
        print("[bridge] ixformer Python module available — symbols in process")
    except:
        print("[bridge] WARNING: no libixformer.so found, link may fail at runtime")

# Add torch lib to rpath
if torch_lib and os.path.isdir(torch_lib):
    extra_ldflags.append(f"-Wl,-rpath,{torch_lib}")

try:
    mod = load(
        name="ix_full_bridge",
        sources=["${BRIDGE_SRC}"],
        extra_cflags=["-O2", "-std=c++17"],
        extra_ldflags=extra_ldflags,
        verbose=True,
    )
    # Copy compiled .so to VLLM_ROOT for import
    import importlib
    spec = importlib.util.find_spec("ix_full_bridge")
    if spec and spec.origin:
        dest = os.path.join("${VLLM_ROOT}", "ix_full_bridge.so")
        shutil.copy2(spec.origin, dest)
        print(f"[bridge] SUCCESS: {dest}")
    else:
        # Try finding it in torch extension build dir
        build_dir = os.path.expanduser("~/.cache/torch_extensions")
        for root, dirs, files in os.walk(build_dir):
            for f in files:
                if f.startswith("ix_full_bridge") and f.endswith(".so"):
                    src = os.path.join(root, f)
                    dest = os.path.join("${VLLM_ROOT}", "ix_full_bridge.so")
                    shutil.copy2(src, dest)
                    print(f"[bridge] SUCCESS: {src} -> {dest}")
                    break
except Exception as e:
    print(f"[bridge] FAILED: {e}")
    raise SystemExit(1)
PYEOF

echo "[bridge] Build complete"
