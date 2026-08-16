FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3
RUN mkdir -p /workspace
WORKDIR /workspace/
# Copy all our engine patches
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./computility-run.yaml /workspace/computility-run.yaml
# Copy ex_engine source for MoE bridge compilation
COPY ./ex_engine/csrc/moe_ops_impl.cu /workspace/qwen3_6_scripts/ex_engine_src/csrc/moe_ops_impl.cu
COPY ./ex_engine/csrc/ix_full_bridge_v2.cpp /workspace/qwen3_6_scripts/ex_engine_src/csrc/ix_full_bridge_v2.cpp
COPY ./ex_engine/build_moe_bridge.sh /workspace/qwen3_6_scripts/ex_engine_src/build_moe_bridge.sh
COPY ./ex_engine/python/moe_dispatch.py /workspace/qwen3_6_scripts/ex_engine_src/python/moe_dispatch.py
COPY ./ex_engine/python/patch_moe_hot_path.py /workspace/qwen3_6_scripts/ex_engine_src/python/patch_moe_hot_path.py
# Make patch script executable and run it
RUN chmod +x /workspace/qwen3_6_scripts/patch_ops.sh && \
    bash /workspace/qwen3_6_scripts/patch_ops.sh 2>&1 | tee /workspace/patch_ops.log ; \
    echo "[Dockerfile] patch_ops exit code: $?"
