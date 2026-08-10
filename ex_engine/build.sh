#!/bin/bash
# ex_engine/build.sh — Compile EX Engine factor .so libraries
#
# CCCL parallel: ci/build_cub.sh selects compiler, arch, std
# We select compiler (corex clang or nvcc), arch (SM70), build .so
#
# Usage:
#   ./ex_engine/build.sh              # auto-detect toolchain
#   ./ex_engine/build.sh --nvcc       # force nvcc
#   ./ex_engine/build.sh --corex      # force corex clang
#
# Output: ex_engine/build/ex_factor_N.so for each factor

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"
CSRC_DIR="${SCRIPT_DIR}/csrc"
INCLUDE_DIR="${SCRIPT_DIR}/include"

mkdir -p "$BUILD_DIR"

# ============================================================================
# Toolchain detection (CCCL pattern: .devcontainer/launch.sh --host)
# ============================================================================

COREX_ROOT="/usr/local/corex"
COREX_CLANG="${COREX_ROOT}/lib64/clang/16"
NVCC="nvcc"
COMPILER=""

detect_toolchain() {
    if [[ "${1:-auto}" == "--corex" ]] || [[ -d "$COREX_CLANG" && "${1:-auto}" != "--nvcc" ]]; then
        # BI-V100 corex SDK — use clang/16 as CUDA compiler
        COMPILER="corex"
        echo "[EX] Using corex clang/16 toolchain at ${COREX_ROOT}"
    elif command -v nvcc &>/dev/null; then
        COMPILER="nvcc"
        echo "[EX] Using nvcc toolchain"
    else
        echo "[EX] ERROR: No CUDA compiler found"
        exit 1
    fi
}

# ============================================================================
# Compile a single factor .cu → .so
# ============================================================================

compile_factor() {
    local factor_id=$1
    local cu_file=$2
    local so_name="ex_factor_${factor_id}.so"
    local so_path="${BUILD_DIR}/${so_name}"

    echo "[EX] Compiling factor ${factor_id}: $(basename ${cu_file}) → ${so_name}"

    if [[ "$COMPILER" == "corex" ]]; then
        # CoreX/Iluvatar: clang-based CUDA compilation
        # From real machine GDN compile log (dockerrizhi.txt):
        #   /usr/local/corex/bin/clang++ ... --cuda-gpu-arch=ivcore10
        #   --cuda-path=/usr/local/corex -std=c++17
        #   -D__ILUVATAR__ -D__ILUVATAR_WORKAROUND__
        local OBJ="${BUILD_DIR}/$(basename ${cu_file} .cu).cuda.o"
        "${COREX_ROOT}/bin/clang++" \
            -D__ILUVATAR__ \
            -D__ILUVATAR_WORKAROUND__ \
            -D__ILUVATAR_DIAG__ \
            -fPIC \
            -O2 \
            --cuda-gpu-arch=ivcore10 \
            --cuda-path="${COREX_ROOT}" \
            -std=c++17 \
            -I"${INCLUDE_DIR}" \
            -isystem "${COREX_ROOT}/include" \
            -c "${cu_file}" \
            -o "${OBJ}"

        # Link .o → .so (match real machine: c++ ... -shared -L ... -lcudart)
        c++ "${OBJ}" -shared \
            -L"${COREX_ROOT}/lib64" \
            -lcudart \
            -o "${so_path}"

        rm -f "${OBJ}"
    else
        # Standard nvcc
        nvcc \
            -arch=sm_70 \
            -std=c++17 \
            -O2 \
            --compiler-options '-fPIC' \
            -shared \
            -I"${INCLUDE_DIR}" \
            -o "${so_path}" \
            "${cu_file}"
    fi

    if [[ -f "${so_path}" ]]; then
        local size=$(stat -c%s "${so_path}" 2>/dev/null || stat -f%z "${so_path}" 2>/dev/null)
        echo "[EX] ✓ ${so_name} (${size} bytes)"
    else
        echo "[EX] ✗ FAILED: ${so_name}"
        return 1
    fi
}

# ============================================================================
# Compile the registry shared library
# ============================================================================

compile_registry() {
    local so_path="${BUILD_DIR}/libex_registry.so"
    echo "[EX] Compiling registry → libex_registry.so"

    gcc -O2 -shared -fPIC \
        -I"${INCLUDE_DIR}" \
        -o "${so_path}" \
        "${CSRC_DIR}/ex_registry.c" \
        -ldl

    if [[ -f "${so_path}" ]]; then
        echo "[EX] ✓ libex_registry.so"
    else
        echo "[EX] ✗ FAILED: libex_registry.so"
        return 1
    fi
}

# ============================================================================
# Main
# ============================================================================

detect_toolchain "${1:-auto}"

echo ""
echo "========================================"
echo "  EX Engine Build"
echo "  Toolchain: ${COMPILER}"
echo "  Output: ${BUILD_DIR}/"
echo "========================================"
echo ""

# Build registry first
compile_registry

# Factor mapping (must match ex_engine.h factor IDs)
FACTORS=(
    "0:factor_moe_topk_softmax.cu"
    "2:factor_moe_fused_gemm.cu"
    "5:factor_gdn_chunk_fwd.cu"
)

TOTAL=0
SUCCESS=0
for entry in "${FACTORS[@]}"; do
    fid="${entry%%:*}"
    cu_file="${CSRC_DIR}/${entry##*:}"
    TOTAL=$((TOTAL + 1))
    if [[ -f "$cu_file" ]]; then
        if compile_factor "$fid" "$cu_file"; then
            SUCCESS=$((SUCCESS + 1))
        fi
    else
        echo "[EX] SKIP factor ${fid}: ${cu_file} not found"
    fi
done

echo ""
echo "========================================"
echo "  Build complete: ${SUCCESS}/${TOTAL} factors"
echo "  Output: ${BUILD_DIR}/"
echo "========================================"
ls -la "${BUILD_DIR}/"
