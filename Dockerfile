FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy sources
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

# Step 1: Compile ix_moe_bridge.so — dlopen bridge to libixformer.so
# This is THE critical .so: it exposes topk_softmax + 11 other ixformer::infer
# functions that the base image's Python binding doesn't expose.
RUN chmod +x /workspace/ex_engine/build.sh && \
    bash /workspace/ex_engine/build.sh 2>&1 | tee /workspace/build.log ; \
    echo "[Docker] build exit code: $?"

# Step 2: Deploy patches (serving layer + conditional model layer)
# patch_ops.sh v2: does NOT overwrite base qwen3_5.py (comp 168 strategy)
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Docker] patch_ops exit code: $?"

# Step 3: Precompile GDN kernel (needs vllm in path)
RUN python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
    /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 | tee -a /workspace/build.log ; \
    echo "[Docker] gdn precompile exit code: $?"
