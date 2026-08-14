#!/bin/bash
# probe_ixformer_symbols.sh — Check what ixformer::infer functions are available
# Run: bash qwen3_6_scripts/probe_ixformer_symbols.sh

echo "=== Locate ixformer .so ==="
python3 -c "
import ixformer, os, glob
d = os.path.dirname(ixformer.__file__)
print(f'ixformer dir: {d}')
for f in glob.glob(os.path.join(d, '*.so')):
    print(f'  {os.path.basename(f)}  ({os.path.getsize(f)} bytes)')
" 2>/dev/null

IXFORMER_SO=$(python3 -c "
import ixformer, os, glob
d = os.path.dirname(ixformer.__file__)
sos = glob.glob(os.path.join(d, '*ixformer*.so'))
print(sos[0] if sos else '')
" 2>/dev/null)

if [ -z "$IXFORMER_SO" ]; then
    echo "ixformer .so not found"
    exit 1
fi

echo ""
echo "=== MoE functions (grouped GEMM pipeline) ==="
nm -D "$IXFORMER_SO" 2>/dev/null | grep -i "moe_w16a16\|group_gemm\|moe_expand\|moe_compute_token\|moe_output_reduce\|topk_softmax" | head -20

echo ""
echo "=== Attention functions ==="
nm -D "$IXFORMER_SO" 2>/dev/null | grep -i "flash_attn_unpad\|paged_attention\|xllm_paged" | head -10

echo ""
echo "=== Linear functions ==="
nm -D "$IXFORMER_SO" 2>/dev/null | grep -i "ixformer_linear\|residual_rms_norm" | head -10

echo ""
echo "=== All ixformer::infer symbols ==="
nm -D "$IXFORMER_SO" 2>/dev/null | grep "ixformer.*infer\|ixinfer" | wc -l
echo "total symbols"

echo ""
echo "=== Also check _ixformer_torch .so ==="
find /usr/local/corex -name "*ixformer*torch*.so" 2>/dev/null | head -5
TORCH_SO=$(find /usr/local/corex -name "*ixformer*torch*.so" 2>/dev/null | head -1)
if [ -n "$TORCH_SO" ]; then
    echo "MoE symbols in torch ext:"
    nm -D "$TORCH_SO" 2>/dev/null | grep -i "moe_w16a16\|group_gemm\|topk_softmax\|moe_expand\|moe_compute" | head -20
fi

echo ""
echo "=== corex lib64 ixformer ==="
ls /usr/local/corex/lib64/libixformer* 2>/dev/null
if [ -f /usr/local/corex/lib64/libixformer.so ]; then
    echo "MoE symbols in libixformer.so:"
    nm -D /usr/local/corex/lib64/libixformer.so 2>/dev/null | grep -i "moe_w16a16\|group_gemm\|topk_softmax" | head -20
fi
