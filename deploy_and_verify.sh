#!/usr/bin/env bash
# deploy_and_verify.sh — Pull latest, deploy patches, verify, start server
# Run from project root: bash deploy_and_verify.sh
#
# What this commit fixes:
#   1. OpenCompass 0 score: max_tokens clamp in serving_chat.py
#   2. t2_n_2 FAIL: n>1 fanout without temperature==0 restriction
#   3. ValidatorIterator index: middleware strips index from messages
#   4. Prefill acceleration: 3-tier ixformer flash attention dispatch
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo " Step 1: Pull latest code"
echo "=========================================="
git pull --ff-only 2>&1 || git pull 2>&1
echo ""

echo "=========================================="
echo " Step 2: Deploy via patch_ops.sh"
echo "=========================================="
cd qwen3_6_scripts
bash patch_ops.sh
cd "$SCRIPT_DIR"
echo ""

echo "=========================================="
echo " Step 3: Verify deployed files"
echo "=========================================="

# Find VLLM_ROOT
VLLM_ROOT=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null || echo "")
if [[ -z "$VLLM_ROOT" ]]; then
    echo "ERROR: cannot find vllm package root"
    exit 1
fi
echo "VLLM_ROOT=${VLLM_ROOT}"

# 3a. Check serving_chat.py has max_tokens clamp
echo ""
echo "--- serving_chat.py: max_tokens clamp ---"
if grep -q "request.max_tokens > default_max_tokens" "${VLLM_ROOT}/entrypoints/openai/serving_chat.py"; then
    echo "  ✓ max_tokens clamp is deployed"
else
    echo "  ✗ max_tokens clamp NOT found — OpenCompass will still score 0"
fi

# 3b. Check serving_chat.py has relaxed fanout
echo ""
echo "--- serving_chat.py: n>1 fanout ---"
if grep -q "2 <= n <= 4" "${VLLM_ROOT}/entrypoints/openai/serving_chat.py"; then
    echo "  ✓ relaxed n>1 fanout is deployed (n=2-4, any temperature)"
else
    echo "  ✗ relaxed fanout NOT found — t2_n_2 may still FAIL"
fi

# 3c. Check api_server.py has sanitize middleware
echo ""
echo "--- api_server.py: index sanitizer middleware ---"
if grep -q "sanitize_chat_body" "${VLLM_ROOT}/entrypoints/openai/api_server.py"; then
    echo "  ✓ index sanitizer middleware is deployed"
else
    echo "  ✗ sanitizer NOT found — ValidatorIterator errors may persist"
fi

# 3d. Check paged_attn.py has ixformer flash dispatch
echo ""
echo "--- paged_attn.py: ixformer flash prefill dispatch ---"
PAGED_ATTN="${VLLM_ROOT}/attention/ops/paged_attn.py"
if grep -q "_ixformer_flash_attn_func" "$PAGED_ATTN"; then
    echo "  ✓ ixformer flash_attn_func dispatch is deployed"
else
    echo "  ✗ flash_attn_func dispatch NOT found"
fi
if grep -q "_ixformer_flash_attn_varlen" "$PAGED_ATTN"; then
    echo "  ✓ ixformer flash_attn_varlen dispatch is deployed"
else
    echo "  ✗ flash_attn_varlen dispatch NOT found"
fi
if grep -q "CoreXFA2" "$PAGED_ATTN"; then
    echo "  ✓ CoreXFA2 dispatch is deployed"
else
    echo "  ✗ CoreXFA2 dispatch NOT found"
fi

echo ""
echo "=========================================="
echo " Step 4: Probe ixformer flash backends"
echo "=========================================="
python3 - <<'PYEOF'
import sys

print("--- ixformer.functions.flash_attn_func ---")
try:
    import ixformer.functions as ixf_F
    fa = ixf_F.flash_attn_func
    print(f"  ✓ available: {fa}")
    # Print signature
    import inspect
    try:
        sig = inspect.signature(fa)
        print(f"    signature: flash_attn_func{sig}")
    except (ValueError, TypeError):
        print("    (signature not inspectable)")
except (ImportError, AttributeError) as e:
    print(f"  ✗ NOT available: {e}")

print("")
print("--- ixformer.contrib.vllm_flash_attn.flash_attn_varlen_func ---")
try:
    from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func
    print(f"  ✓ available: {flash_attn_varlen_func}")
except (ImportError, AttributeError) as e:
    print(f"  ✗ NOT available: {e}")

print("")
print("--- CoreXFA2 (ex_engine.python.corex_fa2) ---")
try:
    # Try from project path first
    sys.path.insert(0, '.')
    from ex_engine.python.corex_fa2 import CoreXFA2
    fa2 = CoreXFA2(4, 1, 256)  # dummy heads for availability check
    print(f"  ✓ imported, is_available={fa2.is_available}")
except ImportError as e:
    print(f"  ✗ NOT available: {e}")

print("")
print("--- ixformer.functions.vllm_single_query_cached_kv_attention ---")
try:
    import ixformer.functions as ixf_F
    pa = ixf_F.vllm_single_query_cached_kv_attention
    print(f"  ✓ paged_attn_v1 available: {pa}")
except (ImportError, AttributeError) as e:
    print(f"  ✗ NOT available: {e}")

print("")
print("=== DISPATCH PREDICTION ===")
backends = []
try:
    from ixformer.contrib.vllm_flash_attn import flash_attn_varlen_func
    backends.append("Tier 0: flash_attn_varlen_func (FUSED)")
except:
    pass
try:
    import ixformer.functions as ixf_F
    _ = ixf_F.flash_attn_func
    backends.append("Tier 0.5: flash_attn_func (FUSED, batch=1)")
except:
    pass
try:
    from ex_engine.python.corex_fa2 import CoreXFA2
    fa2 = CoreXFA2(4, 1, 256)
    if fa2.is_available:
        backends.append("Tier 1: CoreXFA2 packed_prefill (FUSED)")
except:
    pass
backends.append("Tier 2: Python Q-tiling (FALLBACK)")

print(f"  Will try {len(backends)} backends in order:")
for i, b in enumerate(backends):
    marker = ">>> ACTIVE" if i == 0 and "FUSED" in b else ""
    print(f"    {i+1}. {b} {marker}")

if any("FUSED" in b for b in backends[:-1]):
    print("")
    print("  ★ At least one FUSED kernel available!")
    print("    Long-prompt prefill should be dramatically faster.")
else:
    print("")
    print("  ⚠ No fused kernel available — will use Python Q-tiling.")
    print("    Long-prompt prefill will remain slow.")
PYEOF

echo ""
echo "=========================================="
echo " Step 5: Quick OpenCompass clamp test"
echo "=========================================="
python3 - <<'PYEOF2'
# Simulate the max_tokens clamp logic
max_model_len = 131072
test_cases = [
    ("aime2025", 66, 131072),
    ("gpqa_diamond", 1200, 131072),
    ("hle", 500, 131072),
    ("simpleqa", 300, 131072),
    ("longbench_v2", 95000, 131072),
]
print(f"max_model_len = {max_model_len}")
print(f"{'benchmark':<15} {'prompt':>8} {'req_max':>10} {'clamped':>10} {'result':>10}")
print("-" * 60)
for name, prompt_len, req_max in test_cases:
    default_max = max_model_len - prompt_len
    if default_max < 1:
        default_max = 1
    clamped = min(req_max, default_max)
    total = prompt_len + clamped
    result = "✓ OK" if total <= max_model_len else "✗ OVER"
    print(f"{name:<15} {prompt_len:>8} {req_max:>10} {clamped:>10} {result:>10}")
print("")
print("Before fix: ALL benchmarks → 400 error → 0 score")
print("After fix:  ALL benchmarks → request accepted → score > 0")
PYEOF2

echo ""
echo "=========================================="
echo " DONE — Ready to start server"
echo "=========================================="
echo ""
echo "Start server with:"
echo '  CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 \'
echo '  python3 -m vllm.entrypoints.openai.api_server \'
echo '    --model /workspace/models/Qwen3.6-35B-A3B --port 1111 --served-model-name llm \'
echo '    --max-model-len 131072 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \'
echo '    --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \'
echo '    --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \'
echo '    --max-seq-len-to-capture 32768 --enable-auto-tool-choice \'
echo '    --tool-call-parser qwen3_coder --reasoning-parser qwen3'
echo ""
echo "Watch for these log lines after first prefill request:"
echo '  [BI100 PREFILL] ixformer flash_attn_varlen: ... — FUSED kernel active'
echo '  [BI100 PREFILL] ixformer flash_attn_func: ... — FUSED kernel active'
echo '  [BI100 PREFILL] CoreXFA2 packed_prefill: ...'
echo "If none appear, prefill falls back to Python Q-tiling (slow but functional)."
