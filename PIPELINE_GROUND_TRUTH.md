# muh Pipeline Ground Truth — 2026-08-07

## 管道实际状态（不是设计稿，是已部署代码的真实描述）

### scale_mem_bound: FULL PARITY ✓
11/11测试用例与CCCL `cub::detail::scale_mem_bound` 完全匹配。
返回值顺序 `{items_per_thread, threads_per_block}` — items-first，与CCCL一致。

### C++ Tuning Headers: 27/27 ✓
所有26个算法（+common）都有bi100 header，`policy_selector::operator()` 接受
`hardware_capability` 参数。SMEM overflow保护覆盖所有type_size。

### Injection现状（enginex没有.cu源码）

| 注入位置 | 状态 | 值 | commit |
|---------|------|-----|--------|
| prefix_prefill.py BLOCK | ✓ 已手动修改 | BLOCK=64, WARPS=4 | 多个commit |
| paged_attn.py _PARTITION_SIZE | ✓ 保持默认 | 512 | — |
| paged_attn.py V1/V2 dispatch | ✓ 已手动修改 | use_v1 threshold | cbd1f08 |
| _custom_ops.py SMEM | ✓ 已手动修改 | 48KB | 16f0b30 |
| triton_flash_attention.py | ✓ 已添加BI-V100 configs | BLOCK=32/64 | 多个commit |
| protocol.py 兼容性 | ✓ 已修复 | max_completion_tokens等 | 2c353da |

### gen_patch.py 角色
设计时期望: C++ header → unified diff → vllm .cu文件
实际情况: enginex只有Python + .so, 没有.cu源码
当前角色: 文档工具 + 验证（确认header值与已部署Python代码一致）

### CCCL SM100 Benchmark数据（从源码提取，已存入cccl_sm100_benchmark_values.json）

**Reduce** (paged_attention score reduction, Output TPS 83%权重):
- float32+plus: items=16, threads=512, vec=2, speedup=[1.061, 1.000, 1.065, 1.167]
- float64+plus: items=16, threads=640, vec=1, speedup=[1.018, 1.000, 1.016, 1.057]

**Scan** (softmax prefix-sum):
- 4B lookback: items=22, threads=384, delay=1904ns/dcid=6/l2w=830, speedup=[1.148, 0.997, 1.140, 1.463]
- 8B lookback: items=23, threads=416, delay=772ns/dcid=5/l2w=710, speedup=[1.089, 1.016, 1.086, 1.265]

**muh BI-V100适配**:
- reduce float32: items=24(+50%), threads=512(=), vec=2(=) → 补偿16 SMs
- scan 4B: 通过scale_mem_bound自动适配(items=22 @4B安全, @8B降级到16)
- delay参数: ns×0.5, l2w×0.6 (启发式, 待实测)

### 竞赛门槛
- 功能测试: 50+ TC, 项目看板14个FEA item覆盖
- 效果测试: benchmark偏差 ≤ ±4%
- 性能测试: Token吞吐加权值 ≥ 8000
  - Output TPS × 16.796 (83%) → reduce/scan/topk
  - Input TPS × 2.799 (14%) → scan/transform
  - Cache TPS × 0.56 (3%) → batch_memcpy
