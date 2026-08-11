#!/usr/bin/env bash
# build_moe_topk.sh — Compile moe_topk_softmax_v3.cu into importable .so
set -euo pipefail
cd "$(dirname "$0")"

PYTHON=${PYTHON:-python3}
TORCH_ROOT=$($PYTHON -c "import torch; import os; print(os.path.dirname(torch.__file__))")
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TORCH_INC="${TORCH_ROOT}/include"
TORCH_INC2="${TORCH_ROOT}/include/torch/csrc/api/include"
TORCH_LIB="${TORCH_ROOT}/lib"

for _CXX in /usr/local/corex/bin/clang++ g++; do
    [ -x "$_CXX" ] && CXX="$_CXX" && break
done

mkdir -p build
OUT="build/moe_topk_softmax_v3${PY_SUFFIX}"

echo "[build] CXX=$CXX"
echo "[build] Output: $OUT"

$CXX -shared -fPIC -O2 -std=c++17 \
    --cuda-gpu-arch=ivcore10 \
    -I"$PY_INC" \
    -I"$TORCH_INC" \
    -I"$TORCH_INC2" \
    -L"$TORCH_LIB" \
    -ltorch -ltorch_cpu -ltorch_cuda -lc10 -lc10_cuda \
    -Wl,--no-as-needed,-rpath,"$TORCH_LIB" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=moe_topk_softmax_v3 \
    csrc/moe_topk_softmax_v3.cu \
    -o "$OUT" 2>&1

echo "[build] Size: $(du -h "$OUT" | cut -f1)"

# Verify import + GPU test
$PYTHON << PY
import importlib.util, torch
spec = importlib.util.spec_from_file_location("moe_topk_softmax_v3", "$OUT")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
g = torch.randn(4, 64, device="cuda", dtype=torch.float16)
w, ids, src = mod.moe_topk_softmax(g, 8, True)
print(f"[verify] ✓ weights={w.shape} ids={ids.shape} sum={w.sum(-1).tolist()}")
PY
