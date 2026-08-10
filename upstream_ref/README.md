# Upstream Reference: Deep-Spark xllm + vllm

Source repos (cloned 2026-08-09):
- `Deep-Spark/xllm` — Iluvatar's C++ inference engine (Apache 2.0)
- `Deep-Spark/vllm` — Iluvatar's vllm fork (Apache 2.0)

## Call Chain: MoE topk_softmax on BI-V100

```
Our code                           xllm reference                    Iluvatar SDK
─────────────────────────────────  ──────────────────────────────    ──────────────
qwen3_5.py                                                          
  Qwen3_5MoeSparseBlock.forward()                                  
    _custom_ops.py:topk_softmax()                                   
      ixf_F.vllm_moe_topk_softmax  ← MISSING in base image         
      │                                                              
      ├── xllm path (C++ native):                                  
      │   kernels/ilu/fused_moe.cpp                                 
      │     → ixformer::infer::topk_softmax()                      ← ixformer.h
      │       → CUDA kernel (moe_topk_softmax_kernels.cuh)          
      │         topk_gating_softmax<T,VPT,64,4,BYTES_PER_LDG>()    
      │                                                              
      ├── ds_vllm path (Python torch extension):                    
      │   csrc/moe/topk_softmax_kernels.cu                         
      │     → torch.ops._moe_C.topk_softmax()                      
      │       → topk_gating_softmax_kernel_launcher<T>()            
      │                                                              
      └── Our EX Engine path (dlopen .so):                          
          ex_engine/csrc/factor_moe_topk_softmax.cu                 
            → ex_factor_0.so via ctypes                              
            → moe_topk_softmax_kernel()                             
```

## Call Chain: GatedDeltaNet (GDN) on BI-V100

```
Our code                           xllm reference
─────────────────────────────────  ──────────────────────────────
qwen3_5.py                                                       
  GatedDeltaNet.forward()                                        
    prefill path:                                                
      _torch_chunk_gated_delta_rule  ← produces NaN (fp16 overflow)
      │                                                           
      ├── xllm path:                                             
      │   layers/npu_torch/qwen3_gated_delta_net_base.cpp        
      │     → process_mixed_qkv() + recurrent state update       
      │     → full fp32 accumulation                              
      │                                                           
      └── Our EX Engine path:                                    
          ex_engine/csrc/factor_gdn_chunk_fwd.cu                 
            → fp32 state accumulation, tile-based                
```

## File Index

### xllm/kernels/cuda/moe/ — CUDA kernels (the actual GPU code)
- `moe_topk_softmax_kernels.cuh` — **KEY**: fused softmax+topk, CUB-based, power-of-2 expert count optimized
- `moe_fused_topk.cu` — sigmoid/softmax topk dispatcher  
- `moe_topk.cuh` — topk helper functions
- `moe_topk_sigmoid_kernels.cuh` — sigmoid variant for DeepSeek-style routing

### xllm/kernels/ilu/ — Iluvatar ixformer API wrappers
- `ixformer.h` — **KEY**: official ixformer C++ API declarations (topk_softmax, paged_attention, etc.)
- `fused_moe.cpp` — how xllm calls ixformer::infer::topk_softmax()
- `activation.cpp` — silu_and_mul, gelu_and_mul wrappers
- `attention.cpp` — paged_attention wrappers
- `norm.cpp` — rms_norm, fused_add_rms_norm wrappers
- `rope.cpp` — rotary embedding wrappers

### xllm/layers/ilu/ — Complete FusedMoE layer for Iluvatar
- `fused_moe.cpp` — **KEY**: full MoE pipeline: gate → topk → expand → gemm1 → act → gemm2 → combine
- `fused_moe.h` — layer interface

### xllm/layers/npu_torch/ — GatedDeltaNet implementation
- `qwen3_gated_delta_net_base.cpp` — base GDN with fp32 state management
- `qwen3_5_gated_delta_net.cpp` — Qwen3.5 specific GDN

### ds_vllm/csrc/moe/ — vllm-native MoE CUDA kernels
- `topk_softmax_kernels.cu` — vllm's topk_softmax (TensorRT-LLM derived)
- `moeTopKFuncs.cuh` — shared topk reduction primitives
- `moe_align_sum_kernels.cu` — block alignment for scatter

### ds_vllm/vllm/ — Python layer
- `_custom_ops.py` — how vllm calls torch.ops._moe_C.topk_softmax
