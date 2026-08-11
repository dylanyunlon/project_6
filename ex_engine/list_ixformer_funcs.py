#!/usr/bin/env python3
"""List all functions available in ixformer.functions."""
try:
    import ixformer.functions as ixf
    funcs = [x for x in dir(ixf) if not x.startswith('_')]
    print(f"ixformer.functions: {len(funcs)} functions")
    for f in sorted(funcs):
        obj = getattr(ixf, f)
        print(f"  {f}: {type(obj).__name__}")
except ImportError as e:
    print(f"ixformer.functions not available: {e}")

# Also check what torch.ops has after loading
import torch
try:
    import ixformer
    for ns in dir(torch.ops):
        if 'ix' in ns.lower() or 'corex' in ns.lower():
            print(f"  torch.ops.{ns}")
except:
    pass
