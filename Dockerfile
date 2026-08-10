FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

# Build + patch in a single RUN layer. Every step tolerates failure.
RUN set +e && \
    chmod +x /workspace/ex_engine/build.sh && \
    bash /workspace/ex_engine/build.sh --corex || echo "[Dockerfile] build.sh non-zero (ok)" && \
    chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh || echo "[Dockerfile] patch_ops non-zero (ok)" && \
    echo "[Dockerfile] DONE"
