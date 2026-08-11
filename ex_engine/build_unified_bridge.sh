#!/usr/bin/env bash
# build_unified_bridge.sh — Compile ix_unified_bridge.so
# Strategy: try torch.utils.cpp_extension.load() first (proven on BI-V100),
# fall back to manual clang++ if torch extension not available.
set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/csrc/ilu/ix_unified_bridge.cpp"
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

if [ ! -f "$SRC" ]; then
    echo "[build_bridge] ERROR: $SRC not found"
    exit 1
fi

PYTHON=${PYTHON:-python3}

# Method 1: torch.utils.cpp_extension.load() — same method that works for moe_topk, _moe_C, gdn
echo "[build_bridge] Trying torch.utils.cpp_extension.load()..."
$PYTHON << PYEOF
import os, sys, glob

src = "$SRC"
build_dir = "$BUILD_DIR"

try:
    from torch.utils.cpp_extension import load
    
    extra_include = ["$SCRIPT_DIR/csrc/ilu"]
    extra_ldflags = []
    
    for p in ["/usr/local/corex/lib64/python3/dist-packages/ixformer",
              "/usr/local/corex/lib64"]:
        if os.path.isdir(p):
            sos = glob.glob(os.path.join(p, "*.so"))
            if sos:
                extra_ldflags.append(f"-L{p}")
                extra_ldflags.append(f"-Wl,-rpath,{p}")

    # Use load() for compilation only. It may fail on import because
    # ixformer::infer symbols need RTLD_GLOBAL preload at runtime.
    # That's OK — we just need the .so file to exist.
    try:
        ext = load(
            name="ix_unified_bridge",
            sources=[src],
            extra_include_paths=extra_include,
            extra_ldflags=extra_ldflags,
            verbose=True,
            build_directory=build_dir,
        )
        funcs = [x for x in dir(ext) if not x.startswith('_')]
        print(f"[build_bridge] SUCCESS via cpp_extension: {len(funcs)} functions: {funcs}")
        sys.exit(0)
    except ImportError as ie:
        # Compilation succeeded but import failed (expected: ixformer symbols unresolved)
        # Check if .so was actually produced
        built = glob.glob(os.path.join(build_dir, "ix_unified_bridge*.so"))
        if built:
            print(f"[build_bridge] COMPILED OK: {built[0]}")
            print(f"[build_bridge] Import deferred to runtime (ixformer preload needed): {ie}")
            sys.exit(0)
        else:
            print(f"[build_bridge] No .so produced: {ie}")
            sys.exit(1)

except Exception as e:
    # Check if .so exists from compilation before the exception
    built = glob.glob(os.path.join(build_dir, "ix_unified_bridge*.so"))
    if built:
        print(f"[build_bridge] COMPILED OK (exception during import): {built[0]}")
        sys.exit(0)
    print(f"[build_bridge] cpp_extension failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -eq 0 ]; then
    echo "[build_bridge] torch.utils.cpp_extension succeeded"
    ls -la "$BUILD_DIR"/ix_unified_bridge*.so 2>/dev/null
    exit 0
fi

# Method 2: Manual clang++ (fallback)
echo "[build_bridge] Falling back to manual clang++..."
PY_INC=$($PYTHON -c "import sysconfig; print(sysconfig.get_path('include'))")
PY_SUFFIX=$($PYTHON -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")
TORCH_ROOT=$($PYTHON -c "import torch; import os; print(os.path.dirname(torch.__file__))")
TORCH_INC="${TORCH_ROOT}/include"
TORCH_INC2="${TORCH_ROOT}/include/torch/csrc/api/include"
TORCH_LIB="${TORCH_ROOT}/lib"

CXX=""
for _CXX in /usr/local/corex/bin/clang++ g++; do
    [ -x "$_CXX" ] && CXX="$_CXX" && break
done

OUT="${BUILD_DIR}/ix_unified_bridge${PY_SUFFIX}"

$CXX -shared -fPIC -O2 -std=c++17 \
    -I"$SCRIPT_DIR/csrc/ilu" \
    -I"$PY_INC" \
    -I"$TORCH_INC" \
    -I"$TORCH_INC2" \
    -L"$TORCH_LIB" \
    -ltorch -ltorch_cpu -ltorch_python -lc10 \
    -Wl,--no-as-needed,-rpath,"$TORCH_LIB" \
    -Wl,--unresolved-symbols=ignore-in-shared-libs \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    -DTORCH_EXTENSION_NAME=ix_unified_bridge \
    "$SRC" \
    -o "$OUT" 2>&1

if [ -f "$OUT" ]; then
    echo "[build_bridge] SUCCESS via manual clang: $OUT ($(du -h "$OUT" | cut -f1))"
else
    echo "[build_bridge] FAILED"
    exit 1
fi
