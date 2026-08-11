#!/usr/bin/env bash
# build_unified_bridge.sh — Compile ix_unified_bridge.so on BI-V100
# Uses manual compiler flags since torch.utils.cpp_extension is stripped from corex torch.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/csrc/ilu"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "$BUILD_DIR"

PYTHON=${PYTHON:-python3}
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Torch paths — manual discovery (no cpp_extension)
TORCH_ROOT=$($PYTHON -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INC="${TORCH_ROOT}/include"
TORCH_INC2="${TORCH_ROOT}/include/torch/csrc/api/include"
TORCH_LIB="${TORCH_ROOT}/lib"

# Compiler: corex clang or system g++
for _CXX in /usr/local/corex/bin/clang++ /usr/local/corex-3.2.3/bin/clang++ g++; do
    [ -x "$_CXX" ] && CXX="$_CXX" && break
done
echo "[build] CXX=$CXX"
echo "[build] TORCH_ROOT=$TORCH_ROOT"
echo "[build] PY_INC=$PY_INC"

OUT="${BUILD_DIR}/ix_unified_bridge${PY_SUFFIX}"

$CXX -shared -fPIC -O2 -std=c++17 \
    -I"$SRC_DIR" \
    -I"$PY_INC" \
    -I"$TORCH_INC" \
    -I"$TORCH_INC2" \
    -L"$TORCH_LIB" \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Wl,--no-as-needed,-rpath,"$TORCH_LIB" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=ix_unified_bridge \
    "$SRC_DIR/ix_unified_bridge.cpp" \
    -o "$OUT" 2>&1

if [ -f "$OUT" ]; then
    echo "[build] SUCCESS: $OUT ($(du -h "$OUT" | cut -f1))"
    $PYTHON -c "
import importlib.util
spec = importlib.util.spec_from_file_location('ix_unified_bridge', '$OUT')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
funcs = [x for x in dir(mod) if not x.startswith('_')]
print(f'[verify] {len(funcs)} functions: {funcs}')
" 2>&1 || echo "[verify] import test needs ixformer runtime symbols"
else
    echo "[build] FAILED"
fi
