# BI-V100 engine patches for Qwen3.6-35B-A3B (Qwen3_5 architecture)
#
# All modifications are FULL FILE REPLACEMENTS — no AST patch scripts.
# Each file was read in full from the base image vllm source, modified
# with the necessary fixes, and placed here as a complete copy.
#
# Base image: git.modelhub.org.cn:9443/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3
# vllm install path: /usr/local/corex/lib/python3/dist-packages/vllm/

VLLM=/usr/local/corex/lib/python3/dist-packages/vllm
VLLM64=/usr/local/corex/lib64/python3/dist-packages/vllm

# Detect which lib path exists
if [ -d "$VLLM" ]; then
    V=$VLLM
elif [ -d "$VLLM64" ]; then
    V=$VLLM64
else
    echo "[patch_ops] ERROR: vllm not found at lib or lib64 path"
    exit 1
fi

echo "[patch_ops] vllm path: $V"

# --- paged_attn.py: pure-PyTorch attention fallback --------------------------
# Bypasses Triton context_attention_fwd (hangs BI-V100 permanently).
# Uses K-tiling Flash Attention online softmax for prefix attention.
# Uses pure-PyTorch decode for seq_len > 32K.
# CCCL-ported: adaptive tile sizing from dispatch_reduce.cuh GridEvenShare.
cp ./paged_attn.py $V/attention/ops/paged_attn.py
echo "[patch_ops] paged_attn.py → attention/ops/"

# --- model_runner.py: prefix_cache_hit fix -----------------------------------
# Bug: Case 1 (prefix_cache_len <= context_len) leaves prefix_cache_hit=True,
# causing undersized block_tables in chunked prefill chunk 2+.
# Fix: set prefix_cache_hit=False for Case 1.
# FULL FILE REPLACEMENT — no patch_model_runner.py script.
cp ./model_runner.py $V/worker/model_runner.py
echo "[patch_ops] model_runner.py → worker/"

# --- xformers.py: head_dim>128 fallback + Q-tiling --------------------------
# Injects _run_sdpa_fallback (pure matmul+softmax) for head_dim=256.
# ixformer flash attention crashes (is_causal=True) or gives wrong output.
# Also disables auto chunked-prefill (Q-tiling handles long context).
# FULL FILE REPLACEMENT — no patch_xformers_sdpa_seq.py script.
cp ./xformers.py $V/attention/backends/xformers.py
echo "[patch_ops] xformers.py → attention/backends/"

# --- arg_utils.py: disable auto chunked-prefill for 32K+ --------------------
# Q-tiling in _run_sdpa_fallback handles long-context memory.
# FULL FILE REPLACEMENT.
cp ./arg_utils.py $V/engine/arg_utils.py
echo "[patch_ops] arg_utils.py → engine/"

# --- logits_processor.py: seq_groups=None guard ------------------------------
# Prevents crash when seq_groups is None during intermediate chunked-prefill.
# FULL FILE REPLACEMENT.
cp ./logits_processor.py $V/model_executor/layers/logits_processor.py
echo "[patch_ops] logits_processor.py → model_executor/layers/"

# --- sampler.py: CCCL-ported top-k fast path for sampling --------------------
# When all sequences use top_p=1.0, skip full sort+cumsum and use torch.topk.
# CCCL partition/flagged.cu insight: radix select is O(N×bits_per_pass) vs
# full sort O(N log N). For vocab=152064: topk ~11 passes vs sort ~17 passes.
# FULL FILE REPLACEMENT.
cp ./sampler.py $V/model_executor/layers/sampler.py
echo "[patch_ops] sampler.py → model_executor/layers/"

# --- transformers: Qwen3_5 tokenizer / model files --------------------------
# NOTE: patch_transformers_qwen3_5.py is the ONLY remaining patch script.
# It modifies pip-installed transformers' configuration_auto.py and __init__.py
# to register qwen3_5/qwen3_5_moe. These files come from pip (version-specific)
# so we can't pre-copy them — the patch script inserts lines after known anchors.
pip install transformers==4.55.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
cp -r ./qwen3_5 /usr/local/lib/python3.10/site-packages/transformers/models/
cp -r ./qwen3_5_moe /usr/local/lib/python3.10/site-packages/transformers/models/
python3 ./patch_transformers_qwen3_5.py
echo "[patch_ops] transformers Qwen3_5 models installed"

# --- vllm model: Qwen3.6 (Qwen3_5 arch) ------------------------------------
# FULL FILE REPLACEMENT of registry.py with Qwen3_5 entries pre-added.
# No more patch_vllm_qwen3_5.py script.
cp ./mamba_cache.py $V/model_executor/models/
cp ./qwen3_5.py $V/model_executor/models/qwen3_5.py
cp ./registry.py $V/model_executor/models/registry.py
echo "[patch_ops] qwen3_5.py + registry.py deployed"

# --- sequence.py: fix completion_tokens inflation ----------------------------
cp ./sequence.py $V/sequence.py
echo "[patch_ops] sequence.py → /"

# --- scheduler.py: record num_cached_tokens ---------------------------------
cp ./scheduler.py $V/core/scheduler.py
echo "[patch_ops] scheduler.py → core/"

# --- tool parser: Qwen3 XML tool call format --------------------------------
# FULL FILE REPLACEMENT of __init__.py with Qwen3CoderToolParser pre-added.
# No more patch_vllm_tool_parser.py script.
cp ./qwen3coder_tool_parser.py $V/entrypoints/openai/tool_parsers/
cp ./tool_parsers_init.py $V/entrypoints/openai/tool_parsers/__init__.py
echo "[patch_ops] qwen3_coder tool parser deployed"

# --- reasoning parser: Qwen3 <think>...</think> split -----------------------
cp -r ./reasoning $V/
cp ./protocol.py $V/entrypoints/openai/protocol.py
cp ./cli_args.py $V/entrypoints/openai/cli_args.py
cp ./serving_chat.py $V/entrypoints/openai/serving_chat.py
cp ./api_server.py $V/entrypoints/openai/api_server.py
cp ./chat_utils.py $V/entrypoints/chat_utils.py
echo "[patch_ops] reasoning parser + serving files installed"

echo "[patch_ops] DONE — all patches applied via full file replacement"
