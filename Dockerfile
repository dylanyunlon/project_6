FROM git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

RUN mkdir /workspace
WORKDIR /workspace/

# Copy all scripts, V2 kernels, CCCL-tuned prefill, and muh dispatch
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./paged_attention_v2_pytorch.py /workspace/paged_attention_v2_pytorch.py
COPY ./paged_attention_v2_triton.py /workspace/paged_attention_v2_triton.py
COPY ./prefix_prefill.py /workspace/prefix_prefill.py
COPY ./muh_dispatch.py /workspace/muh_dispatch.py

# Run baseline patches (model registration, xformers fallback, tool parser, etc.)
RUN cd ./qwen3_6_scripts && ./patch_ops.sh

# CRITICAL: Enable ixformer native V1/V2 paged attention kernels.
# Fixes: V1 head_mapping int→Tensor, V2 NotImplementedError → native kernel,
# Triton path mismatch.
RUN python3 /workspace/qwen3_6_scripts/patch_ixformer_native.py

# 1. PagedAttention V2 — fills the NotImplementedError hole
#    Enables partitioned attention for long sequences (>8192 tokens)
#    Deploy BOTH PyTorch and Triton V2 to vllm package — _custom_ops.py
#    tries Triton first, falls back to PyTorch if import/runtime fails.
#    Triton V2 risk: SMEM=32KB zero margin at head_dim=256 BLOCK_N=32.
#    If Triton V2 crashes, PyTorch V2 (batched bmm, no intermediate tensor
#    savings but correct) takes over automatically via try/except.
# Deploy Triton V2 kernel into vllm package
RUN cp /workspace/paged_attention_v2_triton.py \
       /usr/local/corex/lib/python3/dist-packages/vllm/paged_attention_v2_triton.py 2>/dev/null || \
    cp /workspace/paged_attention_v2_triton.py \
       /usr/local/corex/lib64/python3/dist-packages/vllm/paged_attention_v2_triton.py 2>/dev/null || true
RUN python3 /workspace/qwen3_6_scripts/patch_paged_attention_v2.py

# Deploy CCCL-tuned prefix_prefill.py (SM=16: BLOCK=64, NUM_WARPS=4)
RUN cp /workspace/prefix_prefill.py \
       /usr/local/corex/lib/python3/dist-packages/vllm/attention/ops/prefix_prefill.py 2>/dev/null || \
    cp /workspace/prefix_prefill.py \
       /usr/local/corex/lib64/python3/dist-packages/vllm/attention/ops/prefix_prefill.py 2>/dev/null || true

# Deploy muh_dispatch.py (CCCL-style type dispatch for kernel configs)
RUN cp /workspace/muh_dispatch.py \
       /usr/local/corex/lib/python3/dist-packages/vllm/muh_dispatch.py 2>/dev/null || \
    cp /workspace/muh_dispatch.py \
       /usr/local/corex/lib64/python3/dist-packages/vllm/muh_dispatch.py 2>/dev/null || true

# 2. Triton kernel tuning: BLOCK=64, NUM_WARPS=4
#    SMEM: BLOCK_N=64 × head_dim=128 × 2B × 2(K+V) = 32KB ≤ 48KB
#    Occupancy: 4 warps allows 2 blocks/SM vs 1 at 8 warps
RUN python3 /workspace/qwen3_6_scripts/patch_triton_tuning.py

# 3. Enable Triton kernels with automatic fallback to PyTorch if they hang
#    Triton Flash Attention is 10-50x faster than PyTorch for-loop fallback
RUN python3 /workspace/qwen3_6_scripts/patch_enable_triton.py

# 5. head_dim=256 support: Qwen3.6 uses head_dim=256
#    BLOCK=64 overflows SMEM (64×256×2×2=64KB > 48KB)
#    → BLOCK=32 for head_dim=256 (32×256×2×2=32KB ≤ 48KB)
RUN python3 /workspace/qwen3_6_scripts/patch_head256_triton.py

# 4. Raise decode threshold: compiled paged_attention_v1 up to 65536
#    instead of falling back to Python at 32768
RUN python3 /workspace/qwen3_6_scripts/patch_vectorized_decode.py
