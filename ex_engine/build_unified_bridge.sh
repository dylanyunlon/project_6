#!/usr/bin/env bash
# build_unified_bridge.sh — Compile ix_unified_bridge.so on BI-V100 real hardware
#
# This builds a single .so that exposes all 14 ixformer::infer functions
# to Python via pybind11.  It links against the base image's existing
# ixformer .so files at runtime (no static linking needed).
#
# Usage:
#   cd /tmp/gdn_test/project_6 && bash ex_engine/build_unified_bridge.sh
#
# Output:
#   ex_engine/build/ix_unified_bridge.cpython-310-x86_64-linux-gnu.so

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${SCRIPT_DIR}/csrc/ilu"
BUILD_DIR="${SCRIPT_DIR}/build"
mkdir -p "$BUILD_DIR"

# Detect Python
PYTHON=${PYTHON:-python3}
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# Detect PyTorch
TORCH_DIR=$($PYTHON -c "import torch; print(torch.utils.cmake_prefix_path)")
TORCH_INC=$($PYTHON -c "import torch; print(torch.utils.cpp_extension.include_paths()[0])")
TORCH_LIB=$($PYTHON -c "import torch; print(torch.utils.cpp_extension.library_paths()[0])")

# Detect corex compiler (prefer) or system g++
if [ -f /usr/local/corex/bin/clang++ ]; then
    CXX=/usr/local/corex/bin/clang++
    echo "[build] Using CoreX clang++: $CXX"
elif [ -f /usr/local/corex/lib64/clang/16/bin/clang++ ]; then
    CXX=/usr/local/corex/lib64/clang/16/bin/clang++
    echo "[build] Using CoreX clang/16: $CXX"
else
    CXX=g++
    echo "[build] Using system g++: $CXX"
fi

echo "[build] Python include: $PY_INC"
echo "[build] Torch include:  $TORCH_INC"
echo "[build] Torch lib:      $TORCH_LIB"
echo "[build] Output suffix:  $PY_SUFFIX"

# Compile
OUT="${BUILD_DIR}/ix_unified_bridge${PY_SUFFIX}"

$CXX -shared -fPIC -O2 -std=c++17 \
    -I"$SRC_DIR" \
    -I"$PY_INC" \
    -I"$TORCH_INC" \
    -I"$TORCH_INC/torch/csrc/api/include" \
    -L"$TORCH_LIB" \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Wl,--no-as-needed \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=ix_unified_bridge \
    "$SRC_DIR/ix_unified_bridge.cpp" \
    -o "$OUT"

echo "[build] SUCCESS: $OUT"
ls -lh "$OUT"

# Verify
$PYTHON -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('ix_unified_bridge', '$OUT')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
funcs = [x for x in dir(mod) if not x.startswith('_')]
print(f'[verify] {len(funcs)} functions exported: {funcs}')
" || echo "[verify] Import test requires ixformer runtime (expected on non-BI-V100)"
