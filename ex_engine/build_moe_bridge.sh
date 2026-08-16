#!/usr/bin/env bash
# build_moe_bridge.sh — Compile MoE ops + bridge into ix_moe_bridge.so
#
# Links against:
#   libcuinfer.so   (cuinferCustomGemm, cuinferTopK — confirmed in symbol dump)
#   libixformer.so  (silu_and_mul, rms_norm, flash_attn, etc — confirmed)
#
# Real device compiler: corex clang/16, NOT nvcc
# Reference: ex_engine/build_ix_bridge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VLLM_ROOT="${1:-}"

echo "[moe_bridge] Building ix_moe_bridge.so"
echo "[moe_bridge] Script dir: ${SCRIPT_DIR}"

# --- Locate sources ---
MOE_CU="${SCRIPT_DIR}/csrc/moe_ops_impl.cu"
BRIDGE_CPP="${SCRIPT_DIR}/csrc/ix_full_bridge_v2.cpp"

if [[ ! -f "$MOE_CU" ]]; then
    echo "[moe_bridge] ERROR: $MOE_CU not found" >&2
    exit 1
fi
if [[ ! -f "$BRIDGE_CPP" ]]; then
    echo "[moe_bridge] ERROR: $BRIDGE_CPP not found" >&2
    exit 1
fi

# --- Locate libraries ---
COREX_ROOT="${COREX_ROOT:-/usr/local/corex}"

# Find libcuinfer.so
CUINFER_SO=""
for d in "${COREX_ROOT}/lib64" "${COREX_ROOT}/lib" "/usr/lib64" "/usr/lib"; do
    if [[ -f "${d}/libcuinfer.so" ]]; then
        CUINFER_SO="${d}/libcuinfer.so"
        break
    fi
done

# Find libixformer.so and ixformer Python package
IX_LIB_DIR=""
IX_SO_FILES=()
for d in \
    "${COREX_ROOT}/lib/python3/dist-packages/ixformer" \
    "${COREX_ROOT}/lib64/python3/dist-packages/ixformer" \
    "$(python3 -c 'import ixformer, os; print(os.path.dirname(ixformer.__file__))' 2>/dev/null || echo '')"; do
    if [[ -d "$d" ]]; then
        IX_LIB_DIR="$d"
        while IFS= read -r so; do
            IX_SO_FILES+=("$so")
        done < <(find "$d" -name "*.so" -type f 2>/dev/null)
        break
    fi
done

echo "[moe_bridge] COREX_ROOT: ${COREX_ROOT}"
echo "[moe_bridge] cuinfer: ${CUINFER_SO:-NOT FOUND}"
echo "[moe_bridge] ixformer dir: ${IX_LIB_DIR:-NOT FOUND}"
echo "[moe_bridge] ixformer .so count: ${#IX_SO_FILES[@]}"

# --- Build via torch.utils.cpp_extension ---
mkdir -p "${SCRIPT_DIR}/prebuilt"

python3 << 'PYEOF'
import os, sys, glob, shutil

script_dir = os.environ.get("SCRIPT_DIR", ".")
vllm_root = os.environ.get("VLLM_ROOT", "")

moe_cu = os.path.join(script_dir, "csrc", "moe_ops_impl.cu")
bridge_cpp = os.path.join(script_dir, "csrc", "ix_full_bridge_v2.cpp")

# Collect linker flags
extra_ldflags = []
rpath_dirs = set()

corex_root = os.environ.get("COREX_ROOT", "/usr/local/corex")
for search_dir in [
    os.path.join(corex_root, "lib64"),
    os.path.join(corex_root, "lib"),
]:
    if os.path.isdir(search_dir):
        rpath_dirs.add(search_dir)
        for so in glob.glob(os.path.join(search_dir, "libcuinfer*.so*")):
            extra_ldflags.append(so)

# ixformer .so files
try:
    import ixformer
    ix_dir = os.path.dirname(ixformer.__file__)
    rpath_dirs.add(ix_dir)
    for so in glob.glob(os.path.join(ix_dir, "*.so")):
        extra_ldflags.append(so)
    for so in glob.glob(os.path.join(ix_dir, "lib*.so")):
        if so not in extra_ldflags:
            extra_ldflags.append(so)
except ImportError:
    # Search common paths
    for d in [
        os.path.join(corex_root, "lib", "python3", "dist-packages", "ixformer"),
        os.path.join(corex_root, "lib64", "python3", "dist-packages", "ixformer"),
    ]:
        if os.path.isdir(d):
            rpath_dirs.add(d)
            for so in glob.glob(os.path.join(d, "*.so")):
                extra_ldflags.append(so)

for d in rpath_dirs:
    extra_ldflags.append(f"-Wl,-rpath,{d}")

print(f"[moe_bridge] Linking against {len(extra_ldflags)} items")
for f in extra_ldflags[:10]:
    print(f"  {f}")

try:
    from torch.utils.cpp_extension import load

    mod = load(
        name="ix_moe_bridge",
        sources=[moe_cu, bridge_cpp],
        extra_include_paths=[os.path.join(script_dir, "csrc")],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=["-O2", "--extended-lambda"],
        extra_ldflags=extra_ldflags,
        verbose=True,
    )
    print("[moe_bridge] ✓ Compilation successful")

    # Find and copy the built .so
    import importlib
    spec = importlib.util.find_spec("ix_moe_bridge")
    if spec and spec.origin:
        dst = os.path.join(script_dir, "prebuilt", "ix_moe_bridge.so")
        shutil.copy2(spec.origin, dst)
        print(f"[moe_bridge] ✓ Saved to {dst}")

        if vllm_root:
            vllm_dst = os.path.join(vllm_root, "ex_engine", "ix_moe_bridge.so")
            os.makedirs(os.path.dirname(vllm_dst), exist_ok=True)
            shutil.copy2(spec.origin, vllm_dst)
            print(f"[moe_bridge] ✓ Deployed to {vllm_dst}")
    else:
        print("[moe_bridge] ⚠ Could not locate compiled .so via importlib")

except Exception as e:
    print(f"[moe_bridge] ERROR: {e}", file=sys.stderr)
    import traceback; traceback.print_exc()
    sys.exit(1)
PYEOF

echo "[moe_bridge] Done"
