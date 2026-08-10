FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy all our engine patches
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml

# Copy EX Engine (algorithm factor replacement system)
COPY ./ex_engine /workspace/ex_engine

# Build EX Engine .so factors for BI-V100
# These replace missing ixformer.functions ops (moe_topk_softmax, gdn_chunk_fwd)
RUN chmod +x /workspace/ex_engine/build.sh && \
    bash /workspace/ex_engine/build.sh --corex 2>&1 | tee /workspace/ex_build.log ; \
    echo "[Dockerfile] ex_engine build exit code: $?"

# Make patch script executable and run it
# patch_ops.sh also wires EX Engine into vllm
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"
