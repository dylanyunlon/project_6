#!/usr/bin/env bash
# build_all.sh — Compile all EX Engine algorithm factors
#
# Output:
#   ix_full_bridge.so    — 8 ops from _ixformer_torch.so (silu, norm, rope, cache, attn, linear)
#   ix_moe_bridge.so     — 5 MoE ops (topk via cuinferTopK, index, expand, combine + group_gemm)
#   gemm_grouped.so      — CUTLASS Cu10 per-expert GEMM (✓ verified 1.97x on BI-V100)
#
# Usage: bash build_all.sh
# Requires: corex clang/16 + torch + cutlass headers

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSRC="${SCRIPT_DIR}/csrc"

# Detect paths
COREX_ROOT="${COREX_ROOT:-/usr/local/corex-3.2.3}"
COREX_LIB=""
for d in "${COREX_ROOT}/lib64" "/usr/local/corex/lib64"; do
    [[ -d "$d" ]] && COREX_LIB="$d" && break
done

IXFORMER_TORCH_SO=""
for f in "${COREX_LIB}"/python3/dist-packages/ixformer/_ixformer_torch*.so \
         "${COREX_LIB}"/python3/dist-packages/ixformer/_ixformer_torch.cpython-310-x86_64-linux-gnu.so; do
    [[ -f "$f" ]] && IXFORMER_TORCH_SO="$f" && break
done

CUINFER_LIB=""
for d in "${COREX_LIB}" "/usr/local/corex/lib64"; do
    [[ -f "${d}/libcuinfer.so" ]] && CUINFER_LIB="$d" && break
done

CUTLASS_INC=""
for d in /usr/local/corex-samples-*/samples/cutlass/include \
         /usr/local/corex/include/cutlass \
         /usr/include/cutlass; do
    [[ -d "$d" ]] && CUTLASS_INC="$d" && break
done

echo "[build] COREX_LIB: ${COREX_LIB}"
echo "[build] IXFORMER_TORCH_SO: ${IXFORMER_TORCH_SO}"
echo "[build] CUINFER_LIB: ${CUINFER_LIB}"
echo "[build] CUTLASS_INC: ${CUTLASS_INC}"

OUTPUT="${SCRIPT_DIR}/prebuilt"
mkdir -p "${OUTPUT}"

# ============================================================
# 1. ix_full_bridge.so — links _ixformer_torch.so
# ============================================================
echo ""
echo "=== Building ix_full_bridge.so ==="
python3 << PYEOF
import os, sys, shutil
try:
    from torch.utils.cpp_extension import load
    mod = load(
        name="ix_full_bridge",
        sources=["${CSRC}/ix_full_bridge_v2.cpp"],
        extra_ldflags=[
            "-L$(dirname ${IXFORMER_TORCH_SO})",
            "-l:$(basename ${IXFORMER_TORCH_SO})",
            "-Wl,-rpath,$(dirname ${IXFORMER_TORCH_SO})",
        ] if "${IXFORMER_TORCH_SO}" else [],
        extra_cflags=["-O2", "-std=c++17"],
        verbose=True,
    )
    import importlib
    spec = importlib.util.find_spec("ix_full_bridge")
    if spec and spec.origin:
        shutil.copy2(spec.origin, "${OUTPUT}/ix_full_bridge.so")
        print("[build] ✓ ix_full_bridge.so")
except Exception as e:
    print(f"[build] ✗ ix_full_bridge.so: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ============================================================
# 2. ix_moe_bridge.so — MoE ops + cuinfer topk
# ============================================================
echo ""
echo "=== Building ix_moe_bridge.so ==="
python3 << PYEOF
import os, sys, shutil
ldflags = ["-O2"]
if "${CUINFER_LIB}":
    ldflags += ["-L${CUINFER_LIB}", "-lcuinfer", "-Wl,-rpath,${CUINFER_LIB}"]

sources = ["${CSRC}/ix_moe_bridge.cpp"]
if os.path.isfile("${CSRC}/moe_ops_impl.cu"):
    sources.append("${CSRC}/moe_ops_impl.cu")

try:
    from torch.utils.cpp_extension import load
    mod = load(
        name="ix_moe_bridge",
        sources=sources,
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=["-O2"],
        extra_ldflags=ldflags,
        verbose=True,
    )
    import importlib
    spec = importlib.util.find_spec("ix_moe_bridge")
    if spec and spec.origin:
        shutil.copy2(spec.origin, "${OUTPUT}/ix_moe_bridge.so")
        print("[build] ✓ ix_moe_bridge.so")
except Exception as e:
    print(f"[build] ✗ ix_moe_bridge.so: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF

# ============================================================
# 3. gemm_grouped.so — CUTLASS Cu10 grouped GEMM
# ============================================================
echo ""
echo "=== Building gemm_grouped.so ==="
if [[ -z "${CUTLASS_INC}" ]]; then
    echo "[build] ✗ gemm_grouped.so: cutlass include not found"
else
    python3 << PYEOF
import os, sys, shutil
sources = [
    "${CSRC}/gemm_grouped.cu",
    "${CSRC}/gemm_grouped_bind.cpp",
]
sources = [s for s in sources if os.path.isfile(s)]
ldflags = []
if "${CUINFER_LIB}":
    ldflags += ["-L${CUINFER_LIB}", "-lcuinfer", "-Wl,-rpath,${CUINFER_LIB}"]

try:
    from torch.utils.cpp_extension import load
    mod = load(
        name="gemm_grouped",
        sources=sources,
        extra_include_paths=["${CUTLASS_INC}", "${CSRC}"],
        extra_cflags=["-O2", "-std=c++17"],
        extra_cuda_cflags=["-O2", "--extended-lambda", "-I${CUTLASS_INC}"],
        extra_ldflags=ldflags,
        verbose=True,
    )
    import importlib
    spec = importlib.util.find_spec("gemm_grouped")
    if spec and spec.origin:
        shutil.copy2(spec.origin, "${OUTPUT}/gemm_grouped.so")
        print("[build] ✓ gemm_grouped.so")
except Exception as e:
    print(f"[build] ✗ gemm_grouped.so: {e}", file=sys.stderr)
    sys.exit(1)
PYEOF
fi

echo ""
echo "=== Build Results ==="
ls -la "${OUTPUT}"/*.so 2>/dev/null || echo "No .so files built"
