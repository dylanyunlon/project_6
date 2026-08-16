#!/usr/bin/env bash
# probe_moe_symbols.sh — Verify ix_moe_bridge.so has all 5 MoE symbols
#
# Run on real device after build_moe_bridge.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find the .so
SO_FILE=""
for p in \
    "${SCRIPT_DIR}/prebuilt/ix_moe_bridge.so" \
    "${SCRIPT_DIR}/ix_moe_bridge.so" \
    "$(python3 -c 'import ix_moe_bridge; print(ix_moe_bridge.__file__)' 2>/dev/null || echo '')"; do
    if [[ -f "$p" ]]; then
        SO_FILE="$p"
        break
    fi
done

if [[ -z "$SO_FILE" ]]; then
    echo "[probe] ERROR: ix_moe_bridge.so not found"
    exit 1
fi

echo "[probe] Checking: $SO_FILE"
echo "[probe] Size: $(du -h "$SO_FILE" | cut -f1)"
echo ""

# Required MoE symbols (must be in ixformer::infer namespace)
REQUIRED=(
    "topk_softmax"
    "moe_compute_token_index_api"
    "moe_expand_input"
    "moe_w16a16_group_gemm"
    "moe_output_reduce_sum"
)

# Required bridge symbols (pybind11 Python bindings)
BRIDGE_REQUIRED=(
    "topk_softmax"
    "moe_gen_idx"
    "moe_expand_input"
    "group_gemm"
    "moe_combine_result"
    "fused_moe_forward"
    "silu_and_mul"
    "rms_norm"
    "linear"
    "paged_attention"
    "flash_attn_prefill"
)

echo "=== MoE implementation symbols (ixformer::infer) ==="
PASS=0
FAIL=0
ALL_SYMS=$(nm -D "$SO_FILE" 2>/dev/null || nm "$SO_FILE" 2>/dev/null || echo "")

for sym in "${REQUIRED[@]}"; do
    count=$(echo "$ALL_SYMS" | grep -c "$sym" || true)
    if [[ $count -gt 0 ]]; then
        echo "  ✓ $sym ($count matches)"
        PASS=$((PASS + 1))
    else
        echo "  ✗ $sym — MISSING"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== pybind11 bridge symbols ==="
for sym in "${BRIDGE_REQUIRED[@]}"; do
    count=$(echo "$ALL_SYMS" | grep -c "$sym" || true)
    if [[ $count -gt 0 ]]; then
        echo "  ✓ $sym"
    else
        echo "  ✗ $sym — MISSING"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "=== Python import test ==="
python3 -c "
import sys
sys.path.insert(0, '$(dirname "$SO_FILE")')
try:
    import ix_moe_bridge as m
    funcs = [f for f in dir(m) if not f.startswith('_')]
    print(f'  ✓ Import OK, {len(funcs)} functions: {funcs}')
except Exception as e:
    print(f'  ✗ Import failed: {e}')
" 2>&1

echo ""
if [[ $FAIL -eq 0 ]]; then
    echo "[probe] ✓ ALL SYMBOLS PRESENT ($PASS MoE + bridge OK)"
else
    echo "[probe] ✗ $FAIL SYMBOLS MISSING"
    exit 1
fi
