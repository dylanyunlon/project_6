FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

ENV PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:${PATH}
ENV PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
ENV VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 BI100_EXECUTOR_STARTUP_DEBUG=1 ENABLE_CUSTOM_IPC=1
ENV BI100_PREFIX_MODEL_FINGERPRINT=Qwen3.6-35B-A3B BI100_PREFIX_DTYPE=float16 BI100_PREFIX_TP_SIZE=4

RUN mkdir -p /workspace
WORKDIR /workspace/
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"
