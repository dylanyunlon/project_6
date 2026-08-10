# PROJECT_SUMMARY — project_6

## 项目背景
天垓100 (BI-V100) 推理引擎竞赛，在 4×BI-V100 上运行 Qwen3.5-27B 推理服务。
竞赛目标：Token吞吐加权值 ≥ 8000（Output TPS × 83% + Input TPS × 14% + Cache TPS × 3%）

## 技术栈
- Base image: bi100-3.2.3-x86-ubuntu20.04-py3.10-poc-llm-infer:v1.2.3
- vLLM 0.6.3 (base) + serving层patch
- ixformer (CoreX SDK, 含 flash_attn / paged_attention / silu_and_mul 等)
- Tensor Parallel = 4, enforce_eager=True

## 文件结构

```
project_6/
├── PRD.md                    # 竞赛需求 + CCCL→base映射
├── SYSTEM_DESIGN.md          # 架构设计: Docker/Build/Runtime/GDN dispatch
├── Dockerfile                # Docker构建
├── computility-run.yaml      # vLLM启动参数
├── qwen3_6_scripts/          # serving层 + model patches (部署到vllm)
│   ├── qwen3_5.py           (2040行) 模型代码: GDN + MoE + Attention
│   ├── serving_chat.py       OpenAI API处理核心
│   ├── protocol.py           请求/响应模型
│   ├── api_server.py         FastAPI入口
│   ├── patch_ops.sh          部署脚本 (全部patch的安装器)
│   ├── flash_qla_sm70/       GDN CUDA kernel (gdn_forward.cu 1919行)
│   └── ...                   其他patches
├── ex_engine/                # EX引擎: 算法因子置换层
│   ├── csrc/
│   │   ├── ix_full_bridge.cpp    (331行) pybind11桥接→ixformer::infer 14个C++函数
│   │   ├── ix_moe_bridge.cpp     (258行) MoE-only子集桥接
│   │   └── moe_topk_softmax_v3.cu (148行) 独立CUDA topk kernel
│   ├── python/
│   │   ├── corex_moe.py      (196行) MoE分发: ix_bridge→ixformer::infer 7步pipeline
│   │   ├── corex_gdn.py      (217行) GDN分发: chunked delta rule + decode
│   │   ├── corex_fa2.py      (228行) FA2分发: packed/paged/chunked三模式
│   │   ├── ix_bridge.py      (162行) ix_full_bridge.so加载器
│   │   └── moe_topk.py        CUDA topk Python wrapper
│   ├── build.sh              编译脚本 (corex clang/16)
│   └── include/              C++ headers
├── cccl_upstream/            (8900文件) NVIDIA CCCL strategic subset
│   ├── cub/                  tuning headers + benchmarks + tests
│   ├── thrust/               examples + tests
│   └── libcudacxx/           C++ STL headers
├── muh/                      muh工具链: BI-V100 tuning parameter生成
│   ├── include/muh/tuning/   27个BI-V100 policy_selector headers
│   └── gen_patch.py          C++ header → vllm unified diff
├── upstream_ref/             上游参考代码
│   ├── ds_vllm/              ds-vllm (vllm fork, 含topk_softmax_kernels.cu)
│   └── xllm/                 xllm (ILU backend: kernels/ilu + layers/ilu)
├── vllm/                     vllm源码副本 (参考用)
└── docs/                     分析文档
```

## 关键文件说明

### ex_engine/csrc/ix_full_bridge.cpp
- `ix_topk_softmax()` → `ixformer::infer::topk_softmax`
- `ix_moe_gen_idx()` → `ixformer::infer::moe_compute_token_index_api`
- `ix_moe_expand_input()` → `ixformer::infer::moe_expand_input`
- `ix_group_gemm()` → `ixformer::infer::moe_w16a16_group_gemm`
- `ix_silu_and_mul()` → `ixformer::infer::silu_and_mul`
- `ix_moe_combine_result()` → `ixformer::infer::moe_output_reduce_sum`
- `ix_fused_moe_forward()` — 以上6步组合, 一次C++调用完成整个MoE
- `ix_paged_attention()` → `ixformer::infer::xllm_paged_attention`
- `ix_flash_attn_prefill()` → `ixformer::infer::ixinfer_flash_attn_unpad_with_block_tables`
- `ix_rms_norm()` / `ix_fused_add_rms_norm()` / `ix_rotary_embedding()` / `ix_reshape_and_cache()`

### ex_engine/python/corex_moe.py
- `moe_forward()` — 3级分发: ix_bridge全C++ → ix_bridge逐步 → Python loop
- `topk_softmax()` — ix_bridge优先, fallback到Python softmax+topk
- `moe_prefill()` / `moe_decode()` — 日志匹配comp 168格式

### qwen3_6_scripts/qwen3_5.py
- `GatedDeltaNet.forward()` — GDN层: corex_gdn dispatch
- `Qwen3_5MoE.forward()` — MoE层: Tier 0-3分发 (ix_fused_moe → ix_bridge → corex_moe → PyTorch)

## 当前状态
- 360+ commits
- 38 GitHub issues (open) + 72 draft issues (待转真issue)
- ix_full_bridge.cpp 已写完14个ixformer::infer函数桥接
- corex_moe/corex_gdn/corex_fa2 已重写, 使用真实ixformer::infer dispatch chain
- 需要真机编译 ix_full_bridge.cpp → .so 并验证MoE走C++ pipeline

## 本次任务完成内容
重写3个dlopen模块(corex_moe.py, corex_gdn.py, corex_fa2.py):
- corex_moe.py: 接入ix_bridge→ixformer::infer 7步MoE pipeline, 移除独立CUDA topk依赖
- corex_gdn.py: gate clamp[-5,0] + state clamp±100 稳定性修复
- corex_fa2.py: 3级tiered dispatch (ix_bridge C++ → ixformer Python → V1 fallback)
- 分析comp 168 docker日志确认真机dlopen调用链条

## 历史任务摘要
- CCCL upstream导入(8900文件) + 27/27 muh tuning headers + CCCL→vllm pattern mapping
- ix_full_bridge.cpp 14函数桥接 + ix_moe_bridge.cpp MoE子集 + moe_topk_softmax_v3.cu
- GDN dtype guard + NaN clamp修复 + corex_gdn/corex_moe初始版本
- serving层部署(protocol/serving_chat/api_server等) + Sub508/509功能修复
- 38 GitHub issues创建 + PRD/SYSTEM_DESIGN文档

## 遗留问题/下次继续
1. **真机编译ix_full_bridge.cpp** — 需要在Docker中JIT编译, 验证MoE走Tier 0 (C++ 7步)
2. **72个draft issues转真issue** — 内容已写好, 需要GitHub API批量关联到repo
3. **MoE Python loop性能** — 如果ix_bridge编译失败, Tier 2的Python expert loop是性能瓶颈(64 experts × 每token)
4. **GDN prefill精度** — FlashQLA .so在BI-V100上编译通过但abs_mean=inf, 需要fp32 accumulation fix
5. **benchmark实测** — Sub168基准 TPS=11.86, 需要在新dispatch chain下重测
