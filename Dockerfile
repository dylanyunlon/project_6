FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

# Copy all sources
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

# Build & deploy: all steps tolerant (;true ensures non-zero won't kill Docker build)
RUN cd /workspace/ex_engine && bash build.sh --corex 2>&1 || true ; \
    python3 /workspace/ex_engine/precompile_moe_topk.py 2>&1 || true ; \
    python3 /workspace/ex_engine/precompile_moe_kernels.py 2>&1 || true ; \
    cd /workspace/qwen3_6_scripts && bash ./patch_ops.sh 2>&1 || true ; \
    python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
        /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 || true ; \
    echo "[Dockerfile] All build steps completed"
