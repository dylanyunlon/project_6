FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy all sources
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

# Step 1: Build EX Engine .so libraries
RUN chmod +x /workspace/ex_engine/build.sh && \
    bash /workspace/ex_engine/build.sh --corex 2>&1 | tee /workspace/ex_build.log ; \
    echo "[Dockerfile] ex_engine build exit code: $?"

# Step 2: Precompile MoE CUDA kernels
RUN python3 /workspace/ex_engine/precompile_moe_topk.py 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] moe_topk precompile exit code: $?"

# Step 3: Precompile vllm v0.5.5 MoE kernels
RUN python3 /workspace/ex_engine/precompile_moe_kernels.py 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] moe_v055 precompile exit code: $?"

# Step 4: Deploy patches (serving + engine fixes)
# patch_ops.sh also builds bridge, deploys .so and Python modules
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"

# Step 5: Precompile GDN kernel (needs vllm in path, so after patch_ops)
RUN python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
    /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 | tee -a /workspace/ex_build.log ; \
    echo "[Dockerfile] gdn precompile exit code: $?"
