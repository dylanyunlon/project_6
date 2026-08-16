#!/bin/bash
# probe_all_symbols.sh — Dump ALL exported symbols from every relevant .so
# No grep filter — save full lists, then we search offline

OUTDIR="cat_files/symbol_dumps"
mkdir -p "$OUTDIR"

echo "=== 1. ALL ixformer .so files ==="
find /usr/local/corex -name "*ixformer*" -name "*.so" 2>/dev/null | sort | tee "$OUTDIR/ixformer_so_list.txt"
echo ""

echo "=== 2. Dump each ixformer .so symbols ==="
while read so; do
    base=$(basename "$so" | sed 's/[^a-zA-Z0-9._-]/_/g')
    count=$(nm -D "$so" 2>/dev/null | grep " T " | wc -l)
    echo "  $so → $base ($count T symbols)"
    nm -D "$so" 2>/dev/null | grep " T " > "$OUTDIR/sym_${base}.txt"
done < "$OUTDIR/ixformer_so_list.txt"
echo ""

echo "=== 3. libixformer.so full T symbols ==="
if [ -f /usr/local/corex/lib64/libixformer.so ]; then
    nm -D /usr/local/corex/lib64/libixformer.so 2>/dev/null | grep " T " > "$OUTDIR/sym_libixformer.txt"
    wc -l "$OUTDIR/sym_libixformer.txt"
    # Also search for ANY moe/expert/gemm/fused related
    echo "  grep moe/expert/gemm/fused/group/batch:"
    grep -i "moe\|expert\|gemm\|fused\|group\|batch\|topk\|gating\|route" "$OUTDIR/sym_libixformer.txt" | head -30
fi
echo ""

echo "=== 4. libcuinfer.so full T symbols ==="
if [ -f /usr/local/corex/lib64/libcuinfer.so ]; then
    nm -D /usr/local/corex/lib64/libcuinfer.so 2>/dev/null | grep " T " > "$OUTDIR/sym_libcuinfer.txt"
    wc -l "$OUTDIR/sym_libcuinfer.txt"
    echo "  grep moe/expert/gemm/fused/group/batch:"
    grep -i "moe\|expert\|gemm\|fused\|group\|batch\|topk\|gating\|route" "$OUTDIR/sym_libcuinfer.txt" | head -30
fi
echo ""

echo "=== 5. _ixformer_torch .so full T symbols ==="
TORCH_SO=$(find /usr/local/corex -name "_ixformer_torch*.so" 2>/dev/null | head -1)
if [ -n "$TORCH_SO" ]; then
    nm -D "$TORCH_SO" 2>/dev/null | grep " T " > "$OUTDIR/sym_ixformer_torch.txt"
    wc -l "$OUTDIR/sym_ixformer_torch.txt"
    echo "  grep moe/expert/gemm/fused/group/batch:"
    grep -i "moe\|expert\|gemm\|fused\|group\|batch\|topk\|gating\|route" "$OUTDIR/sym_ixformer_torch.txt" | head -30
fi
echo ""

echo "=== 6. _C .so (ixformer python binding) full T symbols ==="
C_SO=$(find /usr/local/corex -path "*ixformer*" -name "_C*.so" 2>/dev/null | head -1)
if [ -n "$C_SO" ]; then
    nm -D "$C_SO" 2>/dev/null | grep " T " > "$OUTDIR/sym_ixformer_C.txt"
    wc -l "$OUTDIR/sym_ixformer_C.txt"
    echo "  grep moe/expert/gemm/fused/group/batch:"
    grep -i "moe\|expert\|gemm\|fused\|group\|batch\|topk\|gating\|route" "$OUTDIR/sym_ixformer_C.txt" | head -30
fi
echo ""

echo "=== 7. ALL .so in ixformer package dir ==="
IXDIR=$(python3 -c "import ixformer, os; print(os.path.dirname(ixformer.__file__))" 2>/dev/null)
if [ -n "$IXDIR" ]; then
    echo "ixformer dir: $IXDIR"
    find "$IXDIR" -name "*.so" | while read so; do
        base=$(basename "$so")
        count=$(nm -D "$so" 2>/dev/null | grep " T " | wc -l)
        echo "  $base: $count T symbols"
        nm -D "$so" 2>/dev/null | grep " T " > "$OUTDIR/sym_ixpkg_${base}.txt"
        # Quick search
        hits=$(grep -ic "moe\|expert\|gemm\|fused\|group\|batch\|topk" "$OUTDIR/sym_ixpkg_${base}.txt")
        if [ "$hits" -gt 0 ]; then
            echo "    *** HIT: $hits MoE/GEMM related symbols:"
            grep -i "moe\|expert\|gemm\|fused\|group\|batch\|topk" "$OUTDIR/sym_ixpkg_${base}.txt"
        fi
    done
fi
echo ""

echo "=== 8. ixformer Python API — list ALL callable functions ==="
python3 << 'PY'
import ixformer
import inspect

# List all attributes
for name in sorted(dir(ixformer)):
    if name.startswith('_'):
        continue
    obj = getattr(ixformer, name)
    if callable(obj):
        try:
            sig = inspect.signature(obj)
            print(f"  ixformer.{name}{sig}")
        except (ValueError, TypeError):
            print(f"  ixformer.{name} (no signature)")
    elif hasattr(obj, '__module__'):
        print(f"  ixformer.{name} = {type(obj).__name__}")

# Check submodules
print("\n  --- submodules ---")
for name in sorted(dir(ixformer)):
    obj = getattr(ixformer, name)
    if inspect.ismodule(obj) and not name.startswith('_'):
        print(f"  ixformer.{name}:")
        for sub in sorted(dir(obj)):
            if sub.startswith('_'):
                continue
            subobj = getattr(obj, sub)
            if callable(subobj):
                try:
                    sig = inspect.signature(subobj)
                    print(f"    .{sub}{sig}")
                except:
                    print(f"    .{sub} (no sig)")
PY

echo ""
echo "=== Files saved to $OUTDIR ==="
ls -lh "$OUTDIR/"
echo ""
echo "git add cat_files/symbol_dumps/ && git commit -m 'data: full symbol dumps' && git push"
