# CCCL Reduce Architecture Notes

> Source: `dispatch_reduce.cuh`, `kernel_reduce.cuh`, `agent_reduce.cuh`, `tuning_reduce.cuh`, `util_arch.cuh`
> Read: 2026-08-04 by Claude from CCCL upstream in project_6/cccl_upstream/

## Key Architecture

### Two-pass dispatch (dispatch_reduce.cuh)

```
num_items <= single_tile.threads * single_tile.items
  → SingleTile: one CTA, one kernel launch
  → DeviceReduceSingleTileKernel(d_in, d_out, num_items, ...)

num_items > single_tile threshold
  → Pass 1: DeviceReduceKernel — N CTAs each reduce their share → d_block_reductions[N]
  → Pass 2: DeviceReduceSingleTileKernel — 1 CTA reduces d_block_reductions[N] → d_out
```

Grid size for Pass 1: `max_blocks = sm_occupancy * sm_count * subscription_factor(5)`
For BI-V100: `2 * 16 * 5 = 160 blocks` max.
Each block processes `ceil(num_items / 160)` elements.

### Tile consumption (agent_reduce.cuh)

**Critical: tile data is in registers, NOT SMEM.**

```cpp
AccumT items[ITEMS_PER_THREAD];  // <-- register array, per-thread
// ... load from global memory ...
thread_aggregate = ThreadReduce(items, reduction_op);  // per-thread reduction

// Only SMEM used:
BlockReduce(temp_storage.reduce).Reduce(thread_aggregate, reduction_op);
```

`TempStorage` = `BlockReduce::TempStorage` ≈ threads * sizeof(AccumT) bytes.
NOT threads * items * sizeof(AccumT).

### Vectorized loads

```cpp
ATTEMPT_VECTORIZATION = (vec_size > 1) && (ITEMS_PER_THREAD % vec_size == 0)
    && is_pointer<InputIteratorT>
    && (is_primitive<InputT> || is_trivially_relocatable<InputT>)
    && sizeof(InputT) <= 8;
```

For fp32 scores: vec_size=2 → loads 8 bytes (2 floats) per instruction.
For fp16 KV cache: vec_size=4 → loads 8 bytes (4 halfs) per instruction.

### scale_mem_bound vs scale_reg_bound (util_arch.cuh)

Two scaling functions with different constraints:

**scale_mem_bound** (memory-bound algorithms: reduce, transform):
- items = clamp(nominal * 4 / type_size, 1, nominal * 2)  ← allows 2x expansion
- threads = min(nominal, round_up(48KB / (type_size * items), 32))

**scale_reg_bound** (register-bound algorithms: scan with complex state):
- items = max(1, nominal * 4 / max(4, type_size))  ← no expansion past nominal
- threads = min(nominal, ceil_div(48KB / (type_size * items), 32) * 32)

Key difference: scale_reg_bound uses `max(4, type_size)` preventing items from exceeding nominal for small types, and uses `ceil_div` instead of `round_up` for thread count. Both use 48KB as the cap, but this limits REGISTER PRESSURE (spill to local memory), not actual SMEM usage.

## Impact on muh tuning

### Our SMEM model was wrong for reduce

`test_smem_safety.py` and `check_smem()` in `muh_kernel_map.py` compute
`tile_bytes = threads * items * type_size` and check against 49152.

This is the scale_mem_bound cap, NOT the actual SMEM usage. The actual SMEM
for reduce is approximately `threads * max(sizeof(AccumT), 4)` bytes — about
2-8 KB, not 32-49 KB.

CCCL's SM100 float64 tuning uses `threads=640, items=16` → scale_mem_bound
"tile" = 640*16*8 = 81920 > 49152. But this doesn't overflow SMEM — it only
means scale_mem_bound will cap threads down. The actual kernel SMEM usage
with threads=640 is only ~5120 bytes.

### Our float64/int64 tuning may be too conservative

We use threads=384 items=16 for float64, capped by scale_mem_bound. CCCL
uses threads=640 items=16 on SM100. The question is whether BI-V100's register
file (255 regs/thread) can hold 16 float64 items without spilling.

16 * 8 = 128 bytes = 32 registers per thread for tile data alone.
With overhead (thread_aggregate, loop variables, etc.), ~40 registers/thread.
255 max registers → no spill risk. threads=640 may be safe on BI-V100.

**TODO**: Benchmark threads=640 items=16 for float64 on BI-V100.

### paged_attn.py forces V1

Line 99: `use_v1 = True` overrides V1/V2 heuristic. V2 is completely disabled.
For 100K token sequences, V1 makes one CTA iterate over all KV blocks — bad
for latency. V2 would partition the work and reduce across partitions, which
is exactly CCCL's two-pass pattern.

**TODO**: Re-enable V2 for max_seq_len > 8192. Use muh's partition_size tuning.

### _PARTITION_SIZE = 512 is hardcoded

Not controlled by muh. Should be tunable: larger partition = fewer blocks =
less overhead but more work per block. Optimal value depends on SM count.
For 16 SMs: partition_size=1024 may be better (fewer partitions to reduce).
