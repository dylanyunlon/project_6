FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy all sources
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

# Step 1: Build EX Engine .so libraries (tolerant)
RUN chmod +x /workspace/ex_engine/build.sh && \
    (bash /workspace/ex_engine/build.sh --corex 2>&1 || true) | tee /workspace/ex_build.log ; \
    echo "[Dockerfile] ex_engine build done"

# Step 2: Precompile MoE CUDA kernels (tolerant)
RUN (python3 /workspace/ex_engine/precompile_moe_topk.py 2>&1 || true) | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] moe_topk precompile done"

# Step 3: Precompile vllm v0.5.5 MoE kernels (tolerant)
RUN (python3 /workspace/ex_engine/precompile_moe_kernels.py 2>&1 || true) | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] moe_v055 precompile done"

# Step 4: Deploy patches (serving + engine fixes) — tolerant
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    (cd /workspace/qwen3_6_scripts && bash ./patch_ops.sh 2>&1 || true) | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops done"

# Step 5: Build ix_unified_bridge.so (tolerant — ixformer symbols resolved at runtime)
RUN if [ -f /workspace/ex_engine/build_unified_bridge.sh ]; then \
        chmod +x /workspace/ex_engine/build_unified_bridge.sh && \
        (bash /workspace/ex_engine/build_unified_bridge.sh 2>&1 || true) | tee -a /workspace/ex_build.log ; \
    fi ; \
    echo "[Dockerfile] bridge build done"

# Step 6: Deploy ix_unified Python modules to vllm path (tolerant)
RUN VLLM_ROOT="$(python3 -c 'import vllm; print(vllm.__path__[0])' 2>/dev/null | grep -v '^INFO\|^WARNING\|^DEBUG' | tail -1)" ; \
    if [ -z "$VLLM_ROOT" ]; then VLLM_ROOT="/usr/local/corex/lib64/python3/dist-packages/vllm"; fi ; \
    echo "[Dockerfile] VLLM_ROOT=${VLLM_ROOT}" && \
    for f in ix_unified.py corex_so_loader.py moe_fused_dispatch.py corex_moe.py; do \
        [ -f "/workspace/ex_engine/python/$f" ] && cp "/workspace/ex_engine/python/$f" "${VLLM_ROOT}/$f" 2>/dev/null || true ; \
    done ; \
    for _so in /workspace/ex_engine/build/ix_unified_bridge*.so; do \
        [ -f "$_so" ] && cp "$_so" "${VLLM_ROOT}/" 2>/dev/null || true ; \
    done ; \
    echo "[Dockerfile] bridge deploy done"

# Step 7: Precompile GDN kernel (tolerant)
RUN (python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
    /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 || true) | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] gdn precompile done"
