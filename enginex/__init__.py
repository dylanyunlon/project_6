"""
EngineX — Algorithm Factor Replacement Engine for BI-V100

Architecture modeled after CCCL's dispatch/tuning/kernel three-layer system:
  CCCL:     tuning_*.cuh  → dispatch_*.cuh  → kernel_*.cuh
  EngineX:  tuning/*.py   → dispatch/*.py   → ops/*.py

The engine dlopen()s native .so when available, falls back to PyTorch/ixformer.
This is NOT an adapter — it's a full algorithm factor replacement layer.
"""

__version__ = "0.1.0"

from enginex.dispatch.registry import OperatorRegistry, get_registry

__all__ = ["OperatorRegistry", "get_registry"]
