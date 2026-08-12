FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir -p /workspace
WORKDIR /workspace/

COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
COPY ./ex_engine /workspace/ex_engine

RUN chmod +x /workspace/ex_engine/build.sh ; \
    bash /workspace/ex_engine/build.sh --corex 2>&1 || true

RUN python3 /workspace/ex_engine/precompile_moe_topk.py 2>&1 || true

RUN python3 /workspace/ex_engine/precompile_moe_kernels.py 2>&1 || true

RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh ; \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 || true

RUN python3 /workspace/qwen3_6_scripts/precompile_gdn.py \
    /workspace/qwen3_6_scripts/flash_qla_sm70 2>&1 || true
