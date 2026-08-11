#!/bin/bash
# ==========================================================================
# PATCH_OPS.SH v2 — Align with comp 168 strategy
#
# COMP 168 PROOF (dockerrizhi.txt 07-23 lines 310-397):
#   corex_gdn.py:56   → dlopen libcorex_gdn.so ✅
#   corex_gdn.py:228  → GDN prefill fused ✅  
#   corex_gdn.py:138  → GDN decode fused ✅
#   corex_moe.py:339  → MoE prefill: expert-grouped-wmma ✅
#   corex_moe.py:249  → MoE decode fused ✅
#   corex_fa2.py:333  → FA2 packed prefill ✅
#   corex_fa2.py:507  → FA2 paged chunked prefill ✅
#   corex_fa2.py:225  → FA2 paged decode ✅
#
# ALL 3 corex modules are IN THE BASE IMAGE and work correctly.
# Our Sub508 failed because we OVERWROTE qwen3_5.py, breaking the call chain.
#
# STRATEGY: DO NOT TOUCH model layer. Only deploy:
#   1. transformers config (Qwen3_5Config)
#   2. serving layer (protocol/serving_chat/api_server/chat_utils/tool_parser/reasoning)
#   3. ix_bridge.so (fills ixf_F.vllm_moe_topk_softmax gap if base _custom_ops hits it)
#   4. _custom_ops.py patch (make topk_softmax use ix_bridge instead of crashing)
# ==========================================================================

cd "$(dirname "$0")"
echo "[patch_ops.v2] START — comp 168 aligned strategy"

VLLM=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    [ -d "$P" ] && VLLM="$P" && echo "[patch_ops] Found vllm at: $VLLM" && break
done
[ -z "$VLLM" ] && echo "[patch_ops] ERROR: vllm not found" && exit 1

# ---- PROBE ----
echo "[probe] === Base image state ==="
_QW="$VLLM/model_executor/models/qwen3_5.py"
[ -f "$_QW" ] && echo "[probe] qwen3_5.py: $(wc -c < "$_QW") bytes, $(wc -l < "$_QW") lines" || echo "[probe] qwen3_5.py: MISSING"
for m in corex_gdn.py corex_moe.py corex_fa2.py; do
    _F="$VLLM/model_executor/models/$m"
    [ -f "$_F" ] && echo "[probe] $m: $(wc -c < "$_F") bytes" || echo "[probe] $m: MISSING"
done
ls -la /usr/local/corex/lib64/libcorex_*.so 2>/dev/null || echo "[probe] no libcorex_*.so"
echo "[probe] ==========================="

# Find secondary vllm path for mirroring
VLLM2=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    [ -d "$P" ] && [ "$P" != "$VLLM" ] && VLLM2="$P" && break
done

# Helper: deploy to both vllm paths
deploy_both() {
    local src="$1" dst="$2"
    cp "$src" "$VLLM/$dst" 2>/dev/null || true
    [ -n "$VLLM2" ] && cp "$src" "$VLLM2/$dst" 2>/dev/null || true
}

# ===========================================================
# 1. Transformers config (Qwen3_5Config support)
# ===========================================================
TMODELS=""
for P in /usr/local/lib/python3.10/site-packages/transformers/models \
         /usr/local/corex/lib/python3/dist-packages/transformers/models; do
    [ -d "$P" ] && TMODELS="$P" && break
done
if [ -n "$TMODELS" ]; then
    pip install transformers==4.55.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 30 2>&1 || true
    apt-get update -qq && apt-get install -y -qq ninja-build 2>&1 || true
    cp -r ./qwen3_5 "$TMODELS/" 2>/dev/null || true
    cp -r ./qwen3_5_moe "$TMODELS/" 2>/dev/null || true
    python3 ./patch_transformers_qwen3_5.py 2>&1 || true
    echo "[patch_ops] transformers config deployed"
fi

# ===========================================================
# 2. MODEL LAYER — CONDITIONAL deployment
#    If base has qwen3_5.py > 1000 bytes → DO NOT OVERWRITE
#    This is the comp 168 strategy.
# ===========================================================
_QW_SIZE=0
[ -f "$_QW" ] && _QW_SIZE=$(wc -c < "$_QW")

if [ "$_QW_SIZE" -gt 1000 ]; then
    echo "[patch_ops] *** BASE IMAGE HAS qwen3_5.py (${_QW_SIZE} bytes) — KEEPING IT ***"
    echo "[patch_ops] *** This is the comp 168 strategy: don't break corex_* call chain ***"
    
    # Only add registry entry if missing
    if ! grep -q "Qwen3_5ForCausalLM" "$VLLM/model_executor/models/registry.py" 2>/dev/null; then
        cp ./registry.py "$VLLM/model_executor/models/registry.py" 2>/dev/null && \
            echo "[patch_ops] registry.py deployed (was missing Qwen3_5)"
        [ -n "$VLLM2" ] && cp ./registry.py "$VLLM2/model_executor/models/registry.py" 2>/dev/null || true
    fi
else
    echo "[patch_ops] *** BASE IMAGE MISSING qwen3_5.py — deploying ours ***"
    deploy_both ./qwen3_5.py "model_executor/models/qwen3_5.py"
    deploy_both ./registry.py "model_executor/models/registry.py"
    deploy_both ./mamba_cache.py "model_executor/models/mamba_cache.py"
    
    # Only deploy corex modules if base doesn't have them
    for m in corex_gdn.py corex_moe.py corex_fa2.py; do
        if [ ! -f "$VLLM/model_executor/models/$m" ]; then
            deploy_both "/workspace/ex_engine/python/$m" "model_executor/models/$m"
            echo "[patch_ops] deployed $m (was MISSING)"
        fi
    done
    
    # flash_qla_sm70 (only if we deployed our qwen3_5.py)
    _FLASH_SRC="/workspace/qwen3_6_scripts/flash_qla_sm70"
    if [ -d "$_FLASH_SRC" ]; then
        for _VPATH in "$VLLM" "$VLLM2"; do
            [ -z "$_VPATH" ] && continue
            cp -r "$_FLASH_SRC" "$_VPATH/model_executor/models/flash_qla_sm70" 2>/dev/null || true
        done
        echo "[patch_ops] flash_qla_sm70 deployed"
    fi
fi

# ===========================================================
# 3. SERVING LAYER — always deploy (comp 168 also used custom serving)
# ===========================================================
mkdir -p "$VLLM/entrypoints/openai/tool_parsers" 2>/dev/null || true
[ -n "$VLLM2" ] && mkdir -p "$VLLM2/entrypoints/openai/tool_parsers" 2>/dev/null || true

deploy_both ./protocol.py "entrypoints/openai/protocol.py"
deploy_both ./cli_args.py "entrypoints/openai/cli_args.py"
deploy_both ./serving_chat.py "entrypoints/openai/serving_chat.py"
deploy_both ./api_server.py "entrypoints/openai/api_server.py"
deploy_both ./chat_utils.py "entrypoints/chat_utils.py"
deploy_both ./qwen3coder_tool_parser.py "entrypoints/openai/tool_parsers/qwen3coder_tool_parser.py"
deploy_both ./tool_parsers_init.py "entrypoints/openai/tool_parsers/__init__.py"
python3 ./patch_vllm_tool_parser.py 2>&1 || true
cp -r ./reasoning "$VLLM/" 2>/dev/null || true
[ -n "$VLLM2" ] && cp -r ./reasoning "$VLLM2/" 2>/dev/null || true
echo "[patch_ops] serving layer deployed"

# ===========================================================
# 4. ix_bridge.so — ONLY PURPOSE: fill ixf_F.vllm_moe_topk_softmax gap
#    Even comp 168 had this issue — the base _custom_ops.py tries to call
#    ixf_F.vllm_moe_topk_softmax which doesn't exist.
#    BUT comp 168's corex_moe.py bypasses _custom_ops entirely.
#    So ix_bridge is only needed if base qwen3_5.py path hits _custom_ops.
# ===========================================================
_SITE="/usr/local/corex/lib/python3/dist-packages"
if [ -d "$_SITE" ]; then
    _EX_DST="$_SITE/ex_engine"
    mkdir -p "$_EX_DST/python" "$_EX_DST/build" "$_EX_DST/csrc"
    cp /workspace/ex_engine/python/*.py "$_EX_DST/python/" 2>/dev/null || true
    touch "$_EX_DST/__init__.py" "$_EX_DST/python/__init__.py"
    
    # Deploy pre-built .so
    if [ -d "/workspace/ex_engine/build" ]; then
        cp /workspace/ex_engine/build/*.so "$_EX_DST/build/" 2>/dev/null || true
        cp /workspace/ex_engine/build/*.so "$_EX_DST/" 2>/dev/null || true
        echo "[patch_ops] ex_engine .so deployed: $(ls /workspace/ex_engine/build/*.so 2>/dev/null | wc -l) files"
    fi
    
    # C++ sources for JIT
    cp /workspace/ex_engine/csrc/ix_full_bridge.cpp "$_EX_DST/csrc/" 2>/dev/null || true
    cp /workspace/ex_engine/csrc/ix_moe_bridge.cpp "$_EX_DST/csrc/" 2>/dev/null || true
    
    echo "[patch_ops] ex_engine package deployed to $_SITE"
fi

# ===========================================================
# 5. XFormers patches — DISABLED
#    Sub 168 (base image) achieved output_tps=11.9 WITHOUT any xformers patches.
#    Sub 520 applied these patches → output_tps=2.6 (4.6x slower!)
#    The patches replace ixformer flash attention with pure PyTorch O(L²) matmul.
#    Base image ixformer attention works correctly — proven by Sub 168.
# ===========================================================
echo "[patch_ops] xformers patches SKIPPED — base ixformer attention works (Sub 168 proof)"

# ===========================================================
# 6. model_runner patch (prefix_cache_hit fix)
# ===========================================================
python3 ./patch_model_runner.py 2>&1 || true
echo "[patch_ops] model_runner patched"

# ===========================================================
# 7. Deploy precompiled .so files
# ===========================================================
for _SO in /workspace/ex_engine/moe_topk_softmax_v3*.so /tmp/torch_extensions/*/moe_topk_softmax_v3*.so; do
    [ -f "$_SO" ] && cp "$_SO" "$_SITE/" 2>/dev/null && echo "[patch_ops] MoE topk .so: $(basename $_SO)" && break
done
for _SO in /workspace/ex_engine/moe_ops_v055*.so /tmp/torch_extensions/*/moe_ops_v055*.so; do
    [ -f "$_SO" ] && cp "$_SO" "$_SITE/" 2>/dev/null && echo "[patch_ops] MoE v055 .so: $(basename $_SO)" && break
done

echo "[patch_ops.v2] DONE — comp 168 aligned"
echo "[patch_ops.v2] KEY: base qwen3_5.py $([ "$_QW_SIZE" -gt 1000 ] && echo "KEPT" || echo "REPLACED"), serving layer deployed"
