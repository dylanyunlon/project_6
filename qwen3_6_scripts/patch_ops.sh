# BI-V100 patch script for Qwen3.6-27B (Qwen3_5 architecture)
#
# Triton situation on BI-V100:
#   - Standard Triton 2.3.1 is already present in the image.
#   - HAS_TRITON = False (hardcoded in vendor vllm), but Triton is still used
#     for TP-mode cache management (custom_cache_manager / libentry).
#   - The vendor's triton_utils/__init__.py, custom_cache_manager.py, libentry.py
#     are already correct for standard Triton 2.3.1 — do NOT overwrite them.
#   - DO NOT install BI-V150 corex Triton 2.1.0 (pkgs/triton): that causes
#     GPU hang on BI-V100 because the Triton CUDA PTX kernels are incompatible.

# Recommended server start command for TP=4 support 100K, need chunked prefill
# CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3.6-27B --port 1111 --served-model-name llm \
#     --max-model-len 100000 --enforce-eager --trust-remote-code -tp 4 --gpu-memory-utilization 0.95 \
#     --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
#     --max-num-batched-tokens 4096 --enable-chunked-prefill
#
# With prefix caching (GDN align-mode, requires chunked prefill):
# CUDA_VISIBLE_DEVICES="4,5,6,7" VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 python3 -m vllm.entrypoints.openai.api_server \
#     --model /workspace/models/Qwen3.6-35B-A3B --port 1111 --served-model-name llm \
#     --max-model-len 150000 --trust-remote-code -tp 4 --gpu-memory-utilization 0.90 \
#     --max-num-seqs 1 --disable-log-requests --disable-frontend-multiprocessing \
#     --max-num-batched-tokens 8192 --enable-chunked-prefill --enable-prefix-caching \
#     --max-seq-len-to-capture 32768

# --- paged_attn.py: replace forward_prefix with pure-PyTorch fallback -------
# The Triton context_attention_fwd kernel hangs BI-V100 GPUs permanently
# (standard Triton 2.3.1 PTX is not supported by the corex runtime either).
# Our paged_attn.py bypasses it entirely via _forward_prefix_pytorch, which
# utilizes K-tiling techniques, and also have _forward_decode_pytorch to bypass kernel
# when context length is high
cp ./paged_attn.py /usr/local/corex/lib/python3/dist-packages/vllm/attention/ops/paged_attn.py

# --- model_runner.py: fix prefix_cache_hit stays True in chunked-prefill chunk 2+ ---
# Bug: _compute_for_prefix_cache_hit Case 1 (prefix_cache_len <= context_len)
# leaves prefix_cache_hit=True. Then _add_seq_group uses block_table=computed_block_nums
# (only the original prefix blocks), ignoring chunk-1 KV cache blocks.
# _forward_prefix_pytorch then gets an undersized block_tables and crashes with
# "amax(): Expected reduction dim -1 to have non-zero size" on the 2nd tile.
# Fix: set prefix_cache_hit=False for Case 1 so the full block_tables is used.
python3 ./patch_model_runner.py

# --- transformers: Qwen3_5 tokenizer / model files --------------------------
pip install transformers==4.55.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
cp -r ./qwen3_5 /usr/local/lib/python3.10/site-packages/transformers/models/
cp -r ./qwen3_5_moe /usr/local/lib/python3.10/site-packages/transformers/models/
python3 ./patch_transformers_qwen3_5.py

# --- vllm model: Qwen3.6-27B (Qwen3_5 arch) --------------------------------
cp ./mamba_cache.py /usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models/
cp ./qwen3_5.py /usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models/qwen3_5.py
python3 ./patch_vllm_qwen3_5.py

# --- sequence.py: fix completion_tokens inflation under chunked prefill ------
# Bug: get_output_token_ids_to_return(delta=True) with num_new_tokens=0
# returns _cached_all_token_ids[-0:] == [0:] (the ENTIRE prompt+output list).
# Each prefill chunk step adds prompt_len to previous_num_tokens, so a 10K
# prompt processed in 3 chunks inflates completion_tokens by ~30K.
# Also adds num_cached_tokens field to RequestMetrics for prefix-cache stats.
cp ./sequence.py /usr/local/corex/lib/python3/dist-packages/vllm/sequence.py

# --- scheduler.py: record num_cached_tokens in RequestMetrics ----------------
# Sets seq_group.metrics.num_cached_tokens = prefix_cache_len on first prefill
# when --enable-prefix-caching is active, so serving_chat.py can report it in
# usage.prompt_tokens_details.cached_tokens (OpenAI-compatible API response).
cp ./scheduler.py /usr/local/corex/lib/python3/dist-packages/vllm/core/scheduler.py

# --- xformers: bypass cudnnFlashAttnForward (head_dim=256 > 128 limit) ------
# Injects _run_sdpa_fallback (pure matmul+softmax) into xformers.py.
# Required because head_dim=256 > 128 and ixformer flash attention either
# crashes (is_causal=True) or produces wrong output (attn_mask path).
# The fallback uses query_start_loc to derive actual query lengths, so it
# works correctly during profiling runs with chunked-prefill-style batches.
# also bypasses auto chunked prefill on
python3 ./patch_xformers_sdpa_seq.py

# --- tool parser: Qwen3 XML tool call format ---------------------------------
# Registers "qwen3_coder" parser for Qwen3.6 XML-style tool calls:
#   <tool_call><function=name><parameter=key>\nvalue\n</parameter></function></tool_call>
# Use at server start: --tool-call-parser qwen3_coder --enable-auto-tool-choice
cp ./qwen3coder_tool_parser.py /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/openai/tool_parsers/
python3 ./patch_vllm_tool_parser.py

# --- reasoning parser: Qwen3 <think>...</think> split ------------------------
# Adds --reasoning-parser qwen3 support.
# Routes thinking tokens to reasoning_content, rest to content in the delta.
# Works together with --tool-call-parser qwen3_coder (think → tool call flow).
cp -r ./reasoning /usr/local/corex/lib/python3/dist-packages/vllm/
cp ./protocol.py /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/openai/protocol.py
cp ./cli_args.py /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/openai/cli_args.py
cp ./serving_chat.py /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/openai/serving_chat.py
cp ./api_server.py /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/openai/api_server.py
cp ./chat_utils.py /usr/local/corex/lib/python3/dist-packages/vllm/entrypoints/chat_utils.py
