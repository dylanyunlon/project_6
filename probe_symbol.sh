#!/bin/bash
# probe_symbol.sh — Find which .so has silu_and_mul
echo "=== Searching for silu_and_mul symbol ==="

# The mangled name from the error
SYMBOL="_ZN8ixformer5infer12silu_and_mulERN2at6TensorES3_"

echo ""
echo "--- ixformer package .so files ---"
for f in /usr/local/corex/lib64/python3/dist-packages/ixformer/*.so; do
    echo -n "  $f: "
    if nm -D "$f" 2>/dev/null | grep -q "$SYMBOL"; then
        echo "FOUND ✓"
    elif nm -D "$f" 2>/dev/null | grep -q "silu_and_mul"; then
        echo "has silu_and_mul (different mangling):"
        nm -D "$f" 2>/dev/null | grep "silu_and_mul"
    else
        echo "not found"
    fi
done

echo ""
echo "--- /usr/local/corex/lib64/*.so ---"
for f in /usr/local/corex/lib64/*.so*; do
    r=$(nm -D "$f" 2>/dev/null | grep -c "silu_and_mul")
    if [ "$r" -gt 0 ]; then
        echo "  $f: $r matches"
        nm -D "$f" 2>/dev/null | grep "silu_and_mul" | head -3
    fi
done

echo ""
echo "--- Global search (may take a moment) ---"
find /usr/local/corex -name "*.so*" 2>/dev/null | while read f; do
    r=$(nm -D "$f" 2>/dev/null | grep -c "silu_and_mul")
    if [ "$r" -gt 0 ]; then
        echo "  $f: $r matches"
        nm -D "$f" 2>/dev/null | grep "silu_and_mul" | head -3
    fi
done

echo ""
echo "--- Also check vllm/torch installed .so ---"
find /usr/local/corex/lib64/python3/dist-packages/vllm -name "*.so" 2>/dev/null | while read f; do
    r=$(nm -D "$f" 2>/dev/null | grep -c "silu_and_mul")
    if [ "$r" -gt 0 ]; then
        echo "  $f: $r matches"
        nm -D "$f" 2>/dev/null | grep "silu_and_mul" | head -3
    fi
done

echo ""
echo "--- Python check: how does ixformer.functions.silu_and_mul resolve? ---"
python3 -c "
import ixformer.functions as F
fn = F.silu_and_mul
print(f'Type: {type(fn)}')
print(f'Module: {getattr(fn, \"__module__\", \"?\")}')
# Check if it's from a torch op or C++ binding
import inspect
try:
    print(f'File: {inspect.getfile(fn)}')
except:
    print('File: built-in/C extension')
# Try to find the actual implementation
import ixformer
print(f'ixformer._C: {hasattr(ixformer, \"_C\")}')
if hasattr(ixformer, '_C'):
    c = ixformer._C
    for attr in dir(c):
        if 'silu' in attr.lower():
            print(f'  _C.{attr}')
"
