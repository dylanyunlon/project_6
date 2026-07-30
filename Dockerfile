FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir /workspace
WORKDIR /workspace/
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
RUN cd ./qwen3_6_scripts && ./patch_ops.sh

# BI-V100 Triton kernel tuning: NUM_WARPS=4 for better occupancy
# Derivation: 4 warps at BLOCK=64 allows 2 concurrent blocks/SM
# (vs 1 block/SM at 8 warps, SMEM-limited to 32KB K+V tiles)
RUN python3 /workspace/qwen3_6_scripts/patch_triton_tuning.py
