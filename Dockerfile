FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy all sources
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

# Step 1: Compile _moe_C (CUB-based topk_softmax + moe_align_block_size)
# Proven on real BI-V100: WARP_SIZE=64, -cl-fast-relaxed-math, cub/block/block_reduce.cuh
RUN python3 /workspace/ex_engine/precompile_moe_kernels.py 2>&1 | tee /workspace/ex_build.log ; \
    echo "[Dockerfile] _moe_C precompile exit code: $?"

# Step 2: Deploy patches (serving + engine fixes)
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"

# Step 3: Precompile GDN kernel (needs vllm in path, so after patch_ops)
RUN python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
    /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] gdn precompile exit code: $?"
