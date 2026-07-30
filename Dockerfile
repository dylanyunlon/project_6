FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir /workspace
WORKDIR /workspace/

# Copy all scripts and the V2 module
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./paged_attention_v2_pytorch.py /workspace/paged_attention_v2_pytorch.py

# Run baseline patches (model registration, xformers fallback, tool parser, etc.)
RUN cd ./qwen3_6_scripts && ./patch_ops.sh

# BI-V100 performance patches:
# 1. PagedAttention V2 — fills the NotImplementedError hole
#    Enables partitioned attention for long sequences (>8192 tokens)
#    Expected: 30-50% Output TPS improvement on decode-heavy workloads
RUN python3 /workspace/qwen3_6_scripts/patch_paged_attention_v2.py

# 2. Triton kernel tuning — NUM_WARPS 8→4 for better SM occupancy
RUN python3 /workspace/qwen3_6_scripts/patch_triton_tuning.py
