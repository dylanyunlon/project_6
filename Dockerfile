FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy all sources — ex_engine for build-time .so compilation,
# qwen3_6_scripts for patches+prebuilt, vllm_overrides for core fixes
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine
COPY ./vllm_overrides /workspace/vllm_overrides

# Step 1: Build EX Engine .so libraries (tolerant of compile failures)
RUN chmod +x /workspace/ex_engine/build.sh && \
    bash /workspace/ex_engine/build.sh --corex 2>&1 | tee /workspace/ex_build.log ; \
    echo "[Dockerfile] ex_engine build exit code: $?"

# Step 2: Precompile MoE CUDA kernels (tolerant)
RUN python3 /workspace/ex_engine/precompile_moe_topk.py 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] moe_topk precompile exit code: $?"

# Step 3: Precompile vllm v0.5.5 MoE kernels (tolerant)
RUN python3 /workspace/ex_engine/precompile_moe_kernels.py 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] moe_v055 precompile exit code: $?"

# Step 4: Stage vendor_overrides into qwen3_6_scripts/ so patch_ops.sh finds them
RUN mkdir -p /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block \
             /workspace/qwen3_6_scripts/vendor_overrides/vllm/model_executor/layers && \
    cp /workspace/vllm_overrides/core/evictor_v2.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/evictor_v2.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/core/block/cpu_kv_content_cache.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/cpu_kv_content_cache.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/core/block/cpu_gpu_block_allocator.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/cpu_gpu_block_allocator.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/core/block/prefix_caching_block.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/prefix_caching_block.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/core/block/block_table.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/block_table.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/core/block_manager_v2.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block_manager_v2.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/sampling_params.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/sampling_params.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/model_executor/sampling_metadata.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/model_executor/sampling_metadata.py 2>/dev/null || true && \
    cp /workspace/vllm_overrides/model_executor/layers/sampler.py \
       /workspace/qwen3_6_scripts/vendor_overrides/vllm/model_executor/layers/sampler.py 2>/dev/null || true ; \
    echo "[Dockerfile] vendor_overrides staged"

# Step 5: Deploy patches (serving + engine fixes + prebuilt .so)
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    cd /workspace/qwen3_6_scripts && \
    bash ./patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"

# Step 6: Build ix_unified_bridge.so (ixformer::infer symbols resolved at runtime via RTLD_GLOBAL preload)
# Tolerant: bridge is optional — corex_moe.py has ixformer.functions fallback
RUN chmod +x /workspace/ex_engine/build_unified_bridge.sh && \
    (bash /workspace/ex_engine/build_unified_bridge.sh 2>&1 || echo "[Dockerfile] bridge build FAILED (non-fatal)") | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] ix_unified_bridge build exit code: $?"

# Step 7: Deploy ix_unified and ex_engine Python modules to vllm path
RUN VLLM_ROOT=$(python3 -c "import vllm; print(vllm.__path__[0])" 2>/dev/null | tail -1 || echo "/usr/local/corex/lib64/python3/dist-packages/vllm") && \
    echo "VLLM_ROOT=${VLLM_ROOT}" && \
    cp /workspace/ex_engine/python/ix_unified.py "${VLLM_ROOT}/ix_unified.py" 2>/dev/null || true && \
    cp /workspace/ex_engine/python/corex_so_loader.py "${VLLM_ROOT}/corex_so_loader.py" 2>/dev/null || true && \
    cp /workspace/ex_engine/python/moe_fused_dispatch.py "${VLLM_ROOT}/moe_fused_dispatch.py" 2>/dev/null || true && \
    if ls /workspace/ex_engine/build/ix_unified_bridge*.so 1>/dev/null 2>&1; then \
        cp /workspace/ex_engine/build/ix_unified_bridge*.so "${VLLM_ROOT}/" 2>/dev/null || true ; \
    fi ; \
    echo "[Dockerfile] ex_engine Python modules deployed"

# Step 8: Precompile GDN kernel (needs vllm in path, so after patch_ops)
RUN python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
    /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] gdn precompile exit code: $?"
