FROM harbor.4pd.io/modelhubxc/enginex-iluvatar/bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3

ENV PATH=/usr/local/corex/bin:/usr/local/corex-3.2.3/bin:/usr/local/openmpi/bin:${PATH}
ENV PYTHONPATH=/usr/local/corex/lib64/python3/dist-packages:/usr/local/corex/lib/python3/dist-packages
ENV LD_LIBRARY_PATH=/usr/local/corex/lib:/usr/local/corex/lib64:/usr/local/corex-3.2.3/lib:/usr/local/corex-3.2.3/lib64:/usr/local/openmpi/lib
ENV VLLM_ENGINE_ITERATION_TIMEOUT_S=3600 PYTHONUNBUFFERED=1 PYTHONFAULTHANDLER=1 BI100_EXECUTOR_STARTUP_DEBUG=1 ENABLE_CUSTOM_IPC=1
ENV BI100_PREFIX_MODEL_FINGERPRINT=Qwen3.6-35B-A3B BI100_PREFIX_DTYPE=float16 BI100_PREFIX_TP_SIZE=4

RUN mkdir /workspace
WORKDIR /workspace/
COPY ./qwen3_6_scripts /workspace/qwen3_6_scripts
COPY ./vllm/core/evictor_v2.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/evictor_v2.py
COPY ./vllm/core/block/cpu_kv_content_cache.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/cpu_kv_content_cache.py
COPY ./vllm/core/block/cpu_gpu_block_allocator.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/cpu_gpu_block_allocator.py
COPY ./vllm/core/block/prefix_caching_block.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/prefix_caching_block.py
COPY ./vllm/core/block/block_table.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block/block_table.py
COPY ./vllm/core/block_manager_v2.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/core/block_manager_v2.py
COPY ./vllm/sampling_params.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/sampling_params.py
COPY ./vllm/model_executor/sampling_metadata.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/model_executor/sampling_metadata.py
COPY ./vllm/model_executor/layers/sampler.py /workspace/qwen3_6_scripts/vendor_overrides/vllm/model_executor/layers/sampler.py
RUN cd ./qwen3_6_scripts && bash ./patch_ops.sh
