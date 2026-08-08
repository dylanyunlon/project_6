"""
CoreX API probe — runs at Docker build time (NO GPU, NO runtime imports).

Uses ONLY file system inspection and AST parsing.
Never imports corex modules (they may init CUDA which kills the build).
"""

import ast
import json
import os
import sys

PROBE_RESULT = {}

def probe_file_ast(filepath, name):
    """AST-parse a Python file to extract class/function definitions."""
    result = {"available": False, "classes": {}, "functions": {}, "imports": [], "error": None}
    
    if not os.path.exists(filepath):
        result["error"] = f"File not found: {filepath}"
        return result
    
    result["available"] = True
    result["file"] = filepath
    result["size"] = os.path.getsize(filepath)
    
    try:
        with open(filepath) as f:
            source = f.read()
        result["line_count"] = source.count("\n") + 1
        tree = ast.parse(source)
        
        for node in ast.iter_child_nodes(tree):
            # Top-level imports
            if isinstance(node, ast.Import):
                for alias in node.names:
                    result["imports"].append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                for alias in node.names:
                    result["imports"].append(f"{mod}.{alias.name}")
            
            # Top-level classes
            elif isinstance(node, ast.ClassDef):
                methods = {}
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        args = [arg.arg for arg in item.args.args]
                        methods[item.name] = {
                            "args": args,
                            "lineno": item.lineno,
                        }
                bases = []
                for b in node.bases:
                    if isinstance(b, ast.Name):
                        bases.append(b.id)
                    elif isinstance(b, ast.Attribute):
                        bases.append(f"{ast.dump(b)}")
                result["classes"][node.name] = {
                    "bases": bases,
                    "methods": methods,
                    "lineno": node.lineno,
                }
            
            # Top-level functions
            elif isinstance(node, ast.FunctionDef):
                args = [arg.arg for arg in node.args.args]
                result["functions"][node.name] = {
                    "args": args,
                    "lineno": node.lineno,
                }
    except SyntaxError as e:
        result["error"] = f"SyntaxError: {e}"
    except Exception as e:
        result["error"] = f"{type(e).__name__}: {e}"
    
    return result


# Find vllm models directory
VLLM_MODELS = None
for p in [
    "/usr/local/corex/lib/python3/dist-packages/vllm/model_executor/models",
    "/usr/local/corex/lib64/python3/dist-packages/vllm/model_executor/models",
]:
    if os.path.isdir(p):
        VLLM_MODELS = p
        break

print("=" * 70)
print("[corex_probe] CoreX API Discovery — Build Time (AST only, no GPU)")
print("=" * 70)

if VLLM_MODELS:
    print(f"[corex_probe] vllm models dir: {VLLM_MODELS}")
    
    # List ALL .py files
    all_py = sorted(f for f in os.listdir(VLLM_MODELS) if f.endswith(".py"))
    corex_files = [f for f in all_py if "corex" in f.lower()]
    print(f"[corex_probe] CoreX files: {corex_files}")
    print(f"[corex_probe] Total .py files: {len(all_py)}")
    
    # Probe each corex module by AST
    for target in ["corex_gdn", "corex_moe", "corex_fa2"]:
        filepath = os.path.join(VLLM_MODELS, f"{target}.py")
        result = probe_file_ast(filepath, target)
        PROBE_RESULT[target] = result
        
        if result["available"]:
            print(f"[corex_probe] {target}: FOUND — {result['size']} bytes, {result['line_count']} lines")
            for cls_name, cls_info in result.get("classes", {}).items():
                print(f"[corex_probe]   class {cls_name} (line {cls_info['lineno']}):")
                for mname, minfo in cls_info.get("methods", {}).items():
                    print(f"[corex_probe]     def {mname}({', '.join(minfo['args'])})  # line {minfo['lineno']}")
            for fname, finfo in result.get("functions", {}).items():
                print(f"[corex_probe]   def {fname}({', '.join(finfo['args'])})  # line {finfo['lineno']}")
        else:
            print(f"[corex_probe] {target}: NOT FOUND — {result.get('error', 'unknown')}")
    
    # Inspect native qwen3_5.py BEFORE we overwrite
    native_qw = os.path.join(VLLM_MODELS, "qwen3_5.py")
    if os.path.exists(native_qw):
        sz = os.path.getsize(native_qw)
        with open(native_qw) as f:
            content = f.read()
        lc = content.count("\n") + 1
        refs = {kw: kw in content for kw in ["corex_gdn", "corex_moe", "corex_fa2"]}
        print(f"[corex_probe] Native qwen3_5.py: {sz} bytes, {lc} lines")
        for kw, found in refs.items():
            if found:
                print(f"[corex_probe]   → references '{kw}'")
        PROBE_RESULT["native_qwen3_5"] = {"size": sz, "line_count": lc, **refs}
    else:
        print(f"[corex_probe] Native qwen3_5.py: NOT FOUND")
        PROBE_RESULT["native_qwen3_5"] = {"exists": False}
else:
    print("[corex_probe] ERROR: vllm models directory not found")
    PROBE_RESULT["error"] = "vllm models dir not found"

# Check .so files
for so_name in ["libcorex_gdn.so", "libcorex_moe.so", "libcorex_fa2.so"]:
    path = f"/usr/local/corex/lib64/{so_name}"
    exists = os.path.exists(path)
    size = os.path.getsize(path) if exists else 0
    print(f"[corex_probe] {so_name}: {'EXISTS' if exists else 'MISSING'} ({size} bytes)")
    PROBE_RESULT[so_name] = {"exists": exists, "size": size, "path": path}

# Write JSON
output_path = "/workspace/corex_probe_result.json"
try:
    with open(output_path, "w") as f:
        json.dump(PROBE_RESULT, f, indent=2, default=str)
    print(f"[corex_probe] Results → {output_path}")
except Exception as e:
    print(f"[corex_probe] WARNING: could not write JSON: {e}")

print("=" * 70)
