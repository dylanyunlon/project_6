#!/bin/bash
# ==========================================================================
# PATCH_OPS.SH — Deploy our engine fixes + serving layer
#
# BASE IMAGE HAS BUGS (proven by NaN when using base-only):
#   - GDN layers produce NaN (base corex_gdn.py interface mismatch)
#   - corex_fa2.py missing from model_executor/models/
#   - No multimodal support in model → engine death on image request
#
# COMP 168 DEPLOYED CUSTOM CODE on top of base image to fix these → 48/52 pass
# We must do the same.
# ==========================================================================

cd "$(dirname "$0")"
echo "[patch_ops] START"

VLLM=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    if [ -d "$P" ]; then
        VLLM="$P"
        echo "[patch_ops] Found vllm at: $VLLM"
        break
    fi
done
[ -z "$VLLM" ] && echo "[patch_ops] ERROR: vllm not found" && exit 1

# ---- PROBE ----
echo "[probe] === Base image state ==="
_QW="$VLLM/model_executor/models/qwen3_5.py"
[ -f "$_QW" ] && echo "[probe] qwen3_5.py: $(wc -c < "$_QW") bytes" || echo "[probe] qwen3_5.py: MISSING"
for m in corex_gdn.py corex_moe.py corex_fa2.py; do
    _F="$VLLM/model_executor/models/$m"
    [ -f "$_F" ] && echo "[probe] $m: $(wc -c < "$_F") bytes" || echo "[probe] $m: MISSING"
done
ls -la /usr/local/corex/lib64/libcorex_*.so 2>/dev/null || echo "[probe] no libcorex_*.so"
echo "[probe] ==========================="

# ---- 1. Transformers config ----
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

# ---- 2. Model layer — deploy OUR fixes over base image ----
# 2a. qwen3_5.py — ALWAYS deploy ours (base image has NaN + no multimodal)
cp ./qwen3_5.py "$VLLM/model_executor/models/qwen3_5.py" && \
    echo "[patch_ops] qwen3_5.py deployed (fixes NaN + adds multimodal handling)"

# 2b. corex modules — ALWAYS deploy ours (base interface mismatch causes fallback)
cp /workspace/ex_engine/python/corex_gdn.py "$VLLM/model_executor/models/corex_gdn.py" && \
    echo "[patch_ops] corex_gdn.py deployed (interface matches qwen3_5.py)"
cp /workspace/ex_engine/python/corex_moe.py "$VLLM/model_executor/models/corex_moe.py" && \
    echo "[patch_ops] corex_moe.py deployed"
cp /workspace/ex_engine/python/corex_fa2.py "$VLLM/model_executor/models/corex_fa2.py" && \
    echo "[patch_ops] corex_fa2.py deployed (was MISSING from base)"

# 2c. Registry
if grep -q "Qwen3_5ForCausalLM" "$VLLM/model_executor/models/registry.py" 2>/dev/null; then
    echo "[patch_ops] registry already has Qwen3_5"
else
    cp ./registry.py "$VLLM/model_executor/models/registry.py" 2>/dev/null && \
        echo "[patch_ops] registry.py deployed"
fi

# 2d. XFormers patches (head_dim=256 bypass)
python3 ./patch_xformers_sdpa_seq.py 2>&1 || true
python3 ./patch_xformers_sdpa_batch.py 2>&1 || true
echo "[patch_ops] xformers patches applied"

# 2e. paged_attn.py — CRITICAL: base image uses Triton context_attention_fwd which hangs BI-V100
cp ./paged_attn.py "$VLLM/attention/ops/paged_attn.py" && \
    echo "[patch_ops] paged_attn.py deployed (replaces Triton context_attention_fwd with PyTorch)"
[ -n "$VLLM2" ] && cp ./paged_attn.py "$VLLM2/attention/ops/paged_attn.py" 2>/dev/null || true

# 2f. prefix_prefill.py — provides context_attention_fwd if anything still imports it
if [ -f "./prefix_prefill.py" ]; then
    cp ./prefix_prefill.py "$VLLM/attention/ops/prefix_prefill.py" && \
        echo "[patch_ops] prefix_prefill.py deployed"
    [ -n "$VLLM2" ] && cp ./prefix_prefill.py "$VLLM2/attention/ops/prefix_prefill.py" 2>/dev/null || true
fi

# 2g. model_runner prefix_cache_hit fix
python3 ./patch_model_runner.py 2>&1 || true

# 2h. mamba_cache (GDN state management)
cp ./mamba_cache.py "$VLLM/model_executor/models/mamba_cache.py" 2>/dev/null && \
    echo "[patch_ops] mamba_cache.py deployed"

# 2i. sequence.py (token count fix)
cp ./sequence.py "$VLLM/sequence.py" 2>/dev/null && \
    echo "[patch_ops] sequence.py deployed"

# 2j. scheduler.py (cache metrics)
cp ./scheduler.py "$VLLM/core/scheduler.py" 2>/dev/null && \
    echo "[patch_ops] scheduler.py deployed"

# ---- 3. Serving layer ----
mkdir -p "$VLLM/entrypoints/openai/tool_parsers" 2>/dev/null || true
cp ./qwen3coder_tool_parser.py "$VLLM/entrypoints/openai/tool_parsers/" 2>/dev/null || true
cp ./tool_parsers_init.py "$VLLM/entrypoints/openai/tool_parsers/__init__.py" 2>/dev/null || true
python3 ./patch_vllm_tool_parser.py 2>&1 || true
echo "[patch_ops] tool parser deployed"

cp -r ./reasoning "$VLLM/" 2>/dev/null || true
echo "[patch_ops] reasoning parser deployed"

cp ./protocol.py "$VLLM/entrypoints/openai/protocol.py" 2>/dev/null || true
cp ./cli_args.py "$VLLM/entrypoints/openai/cli_args.py" 2>/dev/null || true
cp ./serving_chat.py "$VLLM/entrypoints/openai/serving_chat.py" 2>/dev/null || true
cp ./api_server.py "$VLLM/entrypoints/openai/api_server.py" 2>/dev/null || true
cp ./chat_utils.py "$VLLM/entrypoints/chat_utils.py" 2>/dev/null || true
echo "[patch_ops] serving layer deployed"

# ---- 4. Mirror to VLLM2 ----
VLLM2=""
for P in /usr/local/corex/lib/python3/dist-packages/vllm \
         /usr/local/corex/lib64/python3/dist-packages/vllm; do
    [ -d "$P" ] && [ "$P" != "$VLLM" ] && VLLM2="$P" && break
done
if [ -n "$VLLM2" ]; then
    echo "[patch_ops] Mirroring to $VLLM2"
    cp ./qwen3_5.py "$VLLM2/model_executor/models/qwen3_5.py" 2>/dev/null || true
    cp /workspace/ex_engine/python/corex_gdn.py "$VLLM2/model_executor/models/corex_gdn.py" 2>/dev/null || true
    cp /workspace/ex_engine/python/corex_moe.py "$VLLM2/model_executor/models/corex_moe.py" 2>/dev/null || true
    cp /workspace/ex_engine/python/corex_fa2.py "$VLLM2/model_executor/models/corex_fa2.py" 2>/dev/null || true
    if ! grep -q "Qwen3_5ForCausalLM" "$VLLM2/model_executor/models/registry.py" 2>/dev/null; then
        cp ./registry.py "$VLLM2/model_executor/models/registry.py" 2>/dev/null || true
    fi
    cp ./mamba_cache.py "$VLLM2/model_executor/models/mamba_cache.py" 2>/dev/null || true
    cp ./sequence.py "$VLLM2/sequence.py" 2>/dev/null || true
    cp ./scheduler.py "$VLLM2/core/scheduler.py" 2>/dev/null || true
    mkdir -p "$VLLM2/entrypoints/openai/tool_parsers" 2>/dev/null || true
    cp ./qwen3coder_tool_parser.py "$VLLM2/entrypoints/openai/tool_parsers/" 2>/dev/null || true
    cp ./tool_parsers_init.py "$VLLM2/entrypoints/openai/tool_parsers/__init__.py" 2>/dev/null || true
    cp -r ./reasoning "$VLLM2/" 2>/dev/null || true
    cp ./protocol.py "$VLLM2/entrypoints/openai/protocol.py" 2>/dev/null || true
    cp ./cli_args.py "$VLLM2/entrypoints/openai/cli_args.py" 2>/dev/null || true
    cp ./serving_chat.py "$VLLM2/entrypoints/openai/serving_chat.py" 2>/dev/null || true
    cp ./api_server.py "$VLLM2/entrypoints/openai/api_server.py" 2>/dev/null || true
    cp ./chat_utils.py "$VLLM2/entrypoints/chat_utils.py" 2>/dev/null || true
fi

# ---- 5. _custom_ops.py (topk_softmax fallback) ----
cp ./_custom_ops.py "$VLLM/_custom_ops.py" 2>/dev/null && \
    echo "[patch_ops] _custom_ops.py deployed" || true
[ -n "$VLLM2" ] && cp ./_custom_ops.py "$VLLM2/_custom_ops.py" 2>/dev/null || true

# ---- 6. ex_engine.python subpackage (qwen3_5.py does "from ex_engine.python.ix_bridge") ----
# The flat ex_engine package has ix_bridge.py at top level, but qwen3_5.py imports from .python subdir
_EX_PKG=$(python3 -c "import ex_engine; import os; print(os.path.dirname(ex_engine.__file__))" 2>/dev/null)
if [ -n "$_EX_PKG" ] && [ -d "$_EX_PKG" ]; then
    mkdir -p "$_EX_PKG/python"
    touch "$_EX_PKG/python/__init__.py"
    for f in ix_bridge.py corex_moe.py corex_gdn.py corex_fa2.py; do
        [ -f "$_EX_PKG/$f" ] && ln -sf "$_EX_PKG/$f" "$_EX_PKG/python/$f"
    done
    echo "[patch_ops] ex_engine.python subpackage linked"
fi

# ---- 7. flash_qla_sm70 deployment to BOTH vllm paths ----
_FLASH_SRC="/workspace/qwen3_6_scripts/flash_qla_sm70"
if [ -d "$_FLASH_SRC" ]; then
    for _VPATH in "$VLLM" "$VLLM2"; do
        [ -z "$_VPATH" ] && continue
        _FLASH_DST="$_VPATH/model_executor/models/flash_qla_sm70"
        cp -r "$_FLASH_SRC" "$_FLASH_DST" 2>/dev/null || true
    done
    echo "[patch_ops] flash_qla_sm70 deployed to vllm model dirs"
fi

echo "[patch_ops] DONE"

# ---- 8. Deploy ex_engine package + compiled .so to Python path ----
_SITE="/usr/local/corex/lib/python3/dist-packages"
if [ -d "$_SITE" ]; then
    # Deploy ex_engine as importable package
    _EX_DST="$_SITE/ex_engine"
    mkdir -p "$_EX_DST/python" "$_EX_DST/build" "$_EX_DST/csrc"
    
    # Python files
    cp /workspace/ex_engine/python/*.py "$_EX_DST/python/" 2>/dev/null || true
    touch "$_EX_DST/__init__.py"
    touch "$_EX_DST/python/__init__.py"
    
    # Compiled .so files from build.sh
    if [ -d "/workspace/ex_engine/build" ]; then
        cp /workspace/ex_engine/build/*.so "$_EX_DST/build/" 2>/dev/null || true
        # Also copy to package root for easy loading
        cp /workspace/ex_engine/build/*.so "$_EX_DST/" 2>/dev/null || true
        echo "[patch_ops] ex_engine .so files deployed: $(ls /workspace/ex_engine/build/*.so 2>/dev/null | wc -l) files"
    fi
    
    # C++ sources for JIT compilation at runtime
    cp /workspace/ex_engine/csrc/ix_full_bridge.cpp "$_EX_DST/csrc/" 2>/dev/null || true
    cp /workspace/ex_engine/csrc/moe_topk_softmax_v3.cu "$_EX_DST/csrc/" 2>/dev/null || true
    if [ -d "/workspace/ex_engine/csrc/moe_v055" ]; then
        cp -r /workspace/ex_engine/csrc/moe_v055 "$_EX_DST/csrc/" 2>/dev/null || true
    fi
    
    # Also deploy to vllm models dir for import compatibility
    _EX_VLLM="$VLLM/model_executor/models/ex_engine"
    mkdir -p "$_EX_VLLM/python" "$_EX_VLLM/csrc"
    cp /workspace/ex_engine/python/*.py "$_EX_VLLM/python/" 2>/dev/null || true
    touch "$_EX_VLLM/__init__.py"
    touch "$_EX_VLLM/python/__init__.py"
    cp /workspace/ex_engine/csrc/ix_full_bridge.cpp "$_EX_VLLM/csrc/" 2>/dev/null || true
    if [ -d "/workspace/ex_engine/build" ]; then
        cp /workspace/ex_engine/build/*.so "$_EX_VLLM/" 2>/dev/null || true
    fi
    
    echo "[patch_ops] ex_engine deployed to $_SITE and $VLLM"
fi

# ---- 9. Deploy precompiled MoE .so ----
# moe_topk_softmax_v3.so (from precompile_moe_topk.py)
for _SO in /workspace/ex_engine/moe_topk_softmax_v3*.so /tmp/torch_extensions/*/moe_topk_softmax_v3*.so; do
    if [ -f "$_SO" ]; then
        cp "$_SO" "$_SITE/" 2>/dev/null || true
        echo "[patch_ops] MoE topk .so deployed: $(basename $_SO)"
        break
    fi
done

# moe_v055 kernels .so (from precompile_moe_kernels.py)  
for _SO in /workspace/ex_engine/moe_ops_v055*.so /tmp/torch_extensions/*/moe_ops_v055*.so; do
    if [ -f "$_SO" ]; then
        cp "$_SO" "$_SITE/" 2>/dev/null || true
        echo "[patch_ops] MoE v055 .so deployed: $(basename $_SO)"
        break
    fi
done

echo "[patch_ops] FINAL: all .so and Python packages deployed"
ls -la "$_EX_DST/build/"*.so 2>/dev/null || echo "[patch_ops] WARNING: no .so in ex_engine/build/"
