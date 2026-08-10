#!/bin/bash
# probe_all_symbols.sh — Check which ixformer::infer symbols actually exist
echo "=== Checking all symbols we need ==="

LIBS=(
    "/usr/local/corex/lib64/python3/dist-packages/ixformer/libixformer.so"
    "/usr/local/corex/lib64/python3/dist-packages/ixformer/_ixformer_torch.cpython-310-x86_64-linux-gnu.so"
    "/usr/local/corex/lib64/python3/dist-packages/ixformer/_C.cpython-310-x86_64-linux-gnu.so"
    "/usr/local/corex/lib64/libixattn.so"
)

FUNCS=(
    "silu_and_mul"
    "topk_softmax"
    "moe_compute_token_index"
    "moe_expand_input"
    "moe_w16a16_group_gemm"
    "moe_output_reduce_sum"
    "xllm_paged_attention"
    "ixinfer_flash_attn_unpad"
    "rms_norm"
    "residual_rms_norm"
    "xllm_rotary_embedding"
    "xllm_reshape_and_cache"
    "ixformer_linear"
)

for func in "${FUNCS[@]}"; do
    echo ""
    echo "--- $func ---"
    found=0
    for lib in "${LIBS[@]}"; do
        if [ -f "$lib" ]; then
            matches=$(nm -D "$lib" 2>/dev/null | grep -i "$func" | grep " T \| W " | head -3)
            if [ -n "$matches" ]; then
                echo "  $(basename $lib):"
                echo "$matches" | while read line; do echo "    $line"; done
                found=1
            fi
        fi
    done
    if [ "$found" -eq 0 ]; then
        echo "  NOT FOUND in any .so (may need dlopen or different namespace)"
        # Also search undefined symbols to see if it's referenced somewhere
        for lib in "${LIBS[@]}"; do
            if [ -f "$lib" ]; then
                undef=$(nm -D "$lib" 2>/dev/null | grep -i "$func" | grep " U " | head -2)
                if [ -n "$undef" ]; then
                    echo "  (undefined ref in $(basename $lib)):"
                    echo "$undef" | while read line; do echo "    $line"; done
                fi
            fi
        done
    fi
done

echo ""
echo "=== Full ixformer::infer namespace in all libs ==="
for lib in "${LIBS[@]}"; do
    if [ -f "$lib" ]; then
        count=$(nm -D "$lib" 2>/dev/null | grep "ixformer.*infer" | grep " T \| W " | wc -l)
        echo ""
        echo "$(basename $lib): $count ixformer::infer symbols"
        nm -D "$lib" 2>/dev/null | grep "ixformer.*infer" | grep " T \| W " | c++filt | head -20
    fi
done
