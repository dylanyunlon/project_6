#!/usr/bin/env bash
# deploy_unified_bridge.sh — Deploy ix_unified_bridge + gdn_fp32 to vllm
#
# Called from patch_ops.sh after build_unified_bridge.sh
# Puts .so and .py into the vllm install path so `from vllm import ...` works.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT=${1:?usage: deploy_unified_bridge.sh VLLM_ROOT}

echo "[deploy] Target: $VLLM_ROOT"

# 1. Deploy ix_unified_bridge.so
BRIDGE_SO=$(find "$SCRIPT_DIR/build" -name "ix_unified_bridge*.so" -print -quit 2>/dev/null || true)
if [ -n "$BRIDGE_SO" ] && [ -f "$BRIDGE_SO" ]; then
    install -m 0755 "$BRIDGE_SO" "$VLLM_ROOT/ix_unified_bridge.so"
    echo "[deploy] ✓ ix_unified_bridge.so → $VLLM_ROOT/"
else
    echo "[deploy] ⚠ ix_unified_bridge.so not built yet (will use Tier1/2 fallback)"
fi

# 2. Deploy Python modules
install -m 0644 "$SCRIPT_DIR/python/ix_unified.py" "$VLLM_ROOT/ix_unified.py"
echo "[deploy] ✓ ix_unified.py → $VLLM_ROOT/"

install -m 0644 "$SCRIPT_DIR/python/gdn_fp32.py" "$VLLM_ROOT/gdn_fp32.py"
echo "[deploy] ✓ gdn_fp32.py → $VLLM_ROOT/"

# 3. Deploy corex_moe.py (updated to use ix_unified)
if [ -f "$SCRIPT_DIR/python/corex_moe.py" ]; then
    install -m 0644 "$SCRIPT_DIR/python/corex_moe.py" "$VLLM_ROOT/model_executor/models/corex_moe.py"
    echo "[deploy] ✓ corex_moe.py → models/"
fi

# 4. Create __init__ stubs so `from vllm import ix_unified` works
for mod in ix_unified gdn_fp32; do
    if [ -f "$VLLM_ROOT/${mod}.py" ]; then
        # Verify it's importable
        python3 -c "import sys; sys.path.insert(0,'$VLLM_ROOT'); import ${mod}; print('[deploy] ✓ ${mod} importable')" || \
            echo "[deploy] ⚠ ${mod}.py deployed but import test failed (may need runtime deps)"
    fi
done

echo "[deploy] Done."
