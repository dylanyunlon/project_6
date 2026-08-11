#!/bin/bash
# build.sh — Compile all .so libraries for ex_engine
#
# Produces:
#   build/ix_moe_bridge.*.so — dlopen bridge to libixformer.so (12 functions)
#
# Run inside Docker where libixformer.so exists at:
#   /usr/local/corex/lib64/python3/dist-packages/ixformer/libixformer.so

set -e
cd "$(dirname "$0")"
echo "[build.sh] START"

mkdir -p build

# ============================================================================
# 1. ix_moe_bridge.so — THE KEY DELIVERABLE
#    Links to libixformer.so → exposes topk_softmax etc to Python
# ============================================================================
echo "[build.sh] Compiling ix_moe_bridge..."
python3 precompile_ix_bridge.py 2>&1 || {
    echo "[build.sh] WARNING: ix_moe_bridge compile failed (expected outside Docker)"
}

# Check result
if ls build/ix_moe_bridge*.so 1>/dev/null 2>&1; then
    echo "[build.sh] SUCCESS: $(ls build/ix_moe_bridge*.so)"
else
    echo "[build.sh] WARNING: no ix_moe_bridge.so produced"
fi

echo "[build.sh] DONE"
ls -la build/*.so 2>/dev/null || echo "[build.sh] No .so files in build/"
