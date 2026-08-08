"""
CoreX API probe — runs at Docker build time (in patch_ops.sh).

Discovers the real interfaces of corex_gdn.py, corex_moe.py, corex_fa2.py
from the base Docker image. Outputs:
  1. /workspace/corex_probe_result.json — machine-readable API map
  2. stdout — human-readable summary for build log

This is NOT runtime code. It runs once during `docker build`.
"""

import importlib
import inspect
import json
import os
import sys

PROBE_RESULT = {}

def probe_module(module_path, name):
    """Try to import a module and extract its public API."""
    result = {"available": False, "classes": {}, "functions": {}, "error": None}
    
    try:
        # Try direct import first
        mod = importlib.import_module(module_path)
        result["available"] = True
        result["file"] = getattr(mod, "__file__", "unknown")
        
        for attr_name in dir(mod):
            if attr_name.startswith("_"):
                continue
            obj = getattr(mod, attr_name)
            
            if inspect.isclass(obj):
                cls_info = {
                    "bases": [b.__name__ for b in obj.__bases__],
                    "methods": {},
                }
                for method_name in dir(obj):
                    if method_name.startswith("_") and method_name != "__init__":
                        continue
                    method = getattr(obj, method_name, None)
                    if callable(method):
                        try:
                            sig = str(inspect.signature(method))
                            cls_info["methods"][method_name] = sig
                        except (ValueError, TypeError):
                            cls_info["methods"][method_name] = "(unknown)"
                result["classes"][attr_name] = cls_info
                
            elif callable(obj):
                try:
                    sig = str(inspect.signature(obj))
                    result["functions"][attr_name] = sig
                except (ValueError, TypeError):
                    result["functions"][attr_name] = "(unknown)"
                    
    except ImportError as e:
        result["error"] = f"ImportError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    
    return result


def probe_file_directly(filepath, name):
    """If import fails, try to read the file and extract class/function defs."""
    result = {"available": False, "classes": {}, "functions": {}, "error": None}
    
    if not os.path.exists(filepath):
        result["error"] = f"File not found: {filepath}"
        return result
    
    result["available"] = True
    result["file"] = filepath
    result["size"] = os.path.getsize(filepath)
    
    try:
        import ast
        with open(filepath) as f:
            tree = ast.parse(f.read())
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                methods = {}
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = []
                        for arg in item.args.args:
                            args.append(arg.arg)
                        methods[item.name] = f"({', '.join(args)})"
                result["classes"][node.name] = {
                    "bases": [ast.dump(b) for b in node.bases],
                    "methods": methods,
                    "lineno": node.lineno,
                }
            elif isinstance(node, ast.FunctionDef) and node.col_offset == 0:
                args = [arg.arg for arg in node.args.args]
                result["functions"][node.name] = {
                    "signature": f"({', '.join(args)})",
                    "lineno": node.lineno,
                }
    except Exception as e:
        result["error"] = f"AST parse error: {e}"
    
    return result


# Probe paths
VLLM_MODELS = None
for p in [
    "/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models",
    "/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models",
]:
    if os.path.isdir(p):
        VLLM_MODELS = p
        break

print("=" * 70)
print("[corex_probe] CoreX API Discovery — Build Time")
print("=" * 70)

if VLLM_MODELS:
    print(f"[corex_probe] vllm models dir: {VLLM_MODELS}")
    
    # List ALL .py files in models dir to find corex modules
    all_files = sorted(os.listdir(VLLM_MODELS))
    corex_files = [f for f in all_files if "corex" in f.lower()]
    print(f"[corex_probe] CoreX files found: {corex_files}")
    print(f"[corex_probe] All model files: {[f for f in all_files if f.endswith('.py')]}")
    
    # Probe each corex module
    for target in ["corex_gdn", "corex_moe", "corex_fa2"]:
        filepath = os.path.join(VLLM_MODELS, f"{target}.py")
        
        # Try import first
        result = probe_module(f"vllm.model_executor.models.{target}", target)
        
        # If import failed, try AST parse
        if not result["available"]:
            print(f"[corex_probe] {target}: import failed ({result['error']}), trying AST...")
            result = probe_file_directly(filepath, target)
        
        PROBE_RESULT[target] = result
        
        if result["available"]:
            print(f"[corex_probe] {target}: FOUND at {result.get('file', filepath)}")
            if result.get("size"):
                print(f"[corex_probe]   size: {result['size']} bytes")
            for cls_name, cls_info in result.get("classes", {}).items():
                print(f"[corex_probe]   class {cls_name}:")
                for method_name, sig in cls_info.get("methods", {}).items():
                    print(f"[corex_probe]     {method_name}{sig}")
            for func_name, func_info in result.get("functions", {}).items():
                if isinstance(func_info, dict):
                    print(f"[corex_probe]   def {func_name}{func_info['signature']} (line {func_info['lineno']})")
                else:
                    print(f"[corex_probe]   def {func_name}{func_info}")
        else:
            print(f"[corex_probe] {target}: NOT AVAILABLE — {result.get('error', 'unknown')}")
    
    # Also probe the native qwen3_5.py BEFORE we overwrite it
    native_qw = os.path.join(VLLM_MODELS, "qwen3_5.py")
    if os.path.exists(native_qw):
        sz = os.path.getsize(native_qw)
        print(f"[corex_probe] Native qwen3_5.py: {sz} bytes")
        # Check if it imports corex
        with open(native_qw) as f:
            content = f.read()
        for keyword in ["corex_gdn", "corex_moe", "corex_fa2", "CoreXGDN", "CoreXMoE"]:
            if keyword in content:
                print(f"[corex_probe]   → references '{keyword}'")
        PROBE_RESULT["native_qwen3_5"] = {
            "size": sz,
            "line_count": content.count("\n"),
            "has_corex_gdn": "corex_gdn" in content,
            "has_corex_moe": "corex_moe" in content,
            "has_corex_fa2": "corex_fa2" in content,
        }
    else:
        print(f"[corex_probe] Native qwen3_5.py: NOT FOUND (will deploy ours)")
        PROBE_RESULT["native_qwen3_5"] = {"size": 0, "exists": False}

else:
    print("[corex_probe] ERROR: vllm models directory not found")
    PROBE_RESULT["error"] = "vllm models dir not found"

# Also check .so files
for so_name, env_var in [
    ("libcorex_gdn.so", "VLLM_COREX_GDN_LIBRARY"),
    ("libcorex_moe.so", "VLLM_COREX_MOE_LIBRARY"),
    ("libcorex_fa2.so", "VLLM_COREX_FA2_LIBRARY"),
]:
    path = os.environ.get(env_var, f"/usr/local/corex/lib64/{so_name}")
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"[corex_probe] {so_name}: {'EXISTS' if exists else 'MISSING'} ({size} bytes) at {path}")
    PROBE_RESULT[so_name] = {"exists": exists, "size": size, "path": path}

# Write results
output_path = "/workspace/corex_probe_result.json"
with open(output_path, "w") as f:
    json.dump(PROBE_RESULT, f, indent=2, default=str)
print(f"\n[corex_probe] Results written to {output_path}")
print("=" * 70)
