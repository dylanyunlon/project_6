from .ex_loader import EXEngine, get_engine

__all__ = ["EXEngine", "get_engine"]

# Lazy imports for new modules (don't break if deps missing)
def __getattr__(name):
    if name == "ix":
        from .ix_unified import ix
        return ix
    if name == "gdn_fp32":
        from . import gdn_fp32
        return gdn_fp32
    if name == "moe_dispatch":
        from . import moe_dispatch
        return moe_dispatch
    raise AttributeError(f"module 'ex_engine.python' has no attribute {name}")
