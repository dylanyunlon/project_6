# Upstream Reference: Deep-Spark xllm + vllm (FULL TREE)

Source repos (cloned 2026-08-09, Apache 2.0):
- `Deep-Spark/xllm` — Iluvatar official C++ LLM inference engine (1470 files)
- `Deep-Spark/vllm` — Iluvatar official vllm fork (703 files, csrc + model layer)

## What's here

### xllm/ (complete source minus git/binaries/submodules)
天数智芯官方下一代推理引擎，C++ 原生，多平台(CUDA/ILU/MLU/NPU)。
包含 kernels → layers → models → runtime → scheduler → api_service 完整栈。

Key subtrees:
- `xllm/core/kernels/ilu/` — ixformer API wrappers (ixformer.h是金矿)
- `xllm/core/kernels/cuda/moe/` — MoE CUDA kernels (topk_softmax, fused_topk)
- `xllm/core/kernels/cuda/` — activation, norm, rope, attention CUDA kernels
- `xllm/core/layers/ilu/` — Iluvatar FusedMoE完整pipeline
- `xllm/core/layers/npu_torch/` — GatedDeltaNet C++ implementation
- `xllm/models/llm/qwen3_5.h` — Qwen3.5 model definition
- `xllm/compiler/tilelang/` — GDN kernel code generation

### ds_vllm/ (csrc + model layers + fused_moe)
天数智芯官方vllm fork，Python + CUDA torch extension。
- `csrc/` — ALL CUDA source (attention, moe, quantization, cache)
- `csrc/libtorch_stable/moe/topk_softmax_kernels.cu` — vllm topk_softmax
- `vllm/_custom_ops.py` — Python → torch.ops._moe_C bridge
- `vllm/model_executor/models/qwen3_5.py` — ds_vllm的qwen3_5实现
- `vllm/model_executor/layers/fused_moe/` — vllm FusedMoE Python layer
