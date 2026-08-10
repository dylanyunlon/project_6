#!/usr/bin/env python3
"""
probe_ixformer_symbols.py — 在真机上跑，探测 ixformer C++ 符号表

用法: python3 probe_ixformer_symbols.py

输出:
  1. ixformer 所有 .so 文件路径
  2. 每个 .so 里包含 topk_softmax / moe / gdn / attention 的符号
  3. 结论：ix_moe_bridge.cpp 能不能链接成功
"""

import subprocess, sys, os, glob

def find_ixformer_so():
    """找到 ixformer 的所有 .so 文件"""
    paths = []
    # 方法1: 从 Python import 路径找
    try:
        import ixformer
        pkg_dir = os.path.dirname(ixformer.__file__)
        paths.extend(glob.glob(os.path.join(pkg_dir, "**/*.so"), recursive=True))
        paths.extend(glob.glob(os.path.join(pkg_dir, "**/*.so.*"), recursive=True))
        print(f"[1] ixformer package dir: {pkg_dir}")
    except ImportError:
        print("[1] ixformer not importable")

    # 方法2: 搜索常见路径
    for base in ["/usr/local/corex/lib64", "/usr/local/corex/lib",
                 "/usr/local/lib", "/usr/lib"]:
        paths.extend(glob.glob(os.path.join(base, "**/libixformer*"), recursive=True))
        paths.extend(glob.glob(os.path.join(base, "**/*ixformer*.so"), recursive=True))
        paths.extend(glob.glob(os.path.join(base, "**/libixattn*"), recursive=True))
        paths.extend(glob.glob(os.path.join(base, "**/libixinfer*"), recursive=True))

    # 方法3: 从 torch 找已加载的 .so
    try:
        import torch
        # ixformer 的 C++ 后端可能是 _ixformer_torch.so 或 _C.so
        try:
            import ixformer._ixformer_torch as ixt
            if hasattr(ixt, '__file__') and ixt.__file__:
                paths.append(ixt.__file__)
                print(f"[2] _ixformer_torch: {ixt.__file__}")
        except:
            pass
        try:
            import ixformer._C as ic
            if hasattr(ic, '__file__') and ic.__file__:
                paths.append(ic.__file__)
                print(f"[2] _C: {ic.__file__}")
        except:
            pass
    except:
        pass

    return list(set(paths))

def nm_grep(so_path, patterns):
    """用 nm 查符号，grep 匹配"""
    results = []
    try:
        out = subprocess.run(
            ["nm", "-D", "--demangle", so_path],
            capture_output=True, text=True, timeout=10)
        for line in out.stdout.splitlines():
            for p in patterns:
                if p.lower() in line.lower():
                    results.append(line.strip())
    except Exception as e:
        # nm 可能不存在，用 objdump
        try:
            out = subprocess.run(
                ["objdump", "-T", so_path],
                capture_output=True, text=True, timeout=10)
            for line in out.stdout.splitlines():
                for p in patterns:
                    if p.lower() in line.lower():
                        results.append(line.strip())
        except Exception as e2:
            results.append(f"ERROR: nm/objdump failed: {e}, {e2}")
    return results

def check_python_binding():
    """检查 Python 层面有没有 topk_softmax"""
    print("\n=== Python Binding Check ===")
    try:
        import ixformer.functions as ixf
        attrs = dir(ixf)
        moe_attrs = [a for a in attrs if 'moe' in a.lower() or 'topk' in a.lower() 
                     or 'softmax' in a.lower() or 'expert' in a.lower()]
        print(f"  ixformer.functions MoE-related: {moe_attrs}")
        if not moe_attrs:
            print(f"  ixformer.functions ALL ({len(attrs)}): {attrs}")
    except Exception as e:
        print(f"  ixformer.functions: {e}")

    try:
        import ixformer
        # 搜索所有子模块
        for attr_name in dir(ixformer):
            obj = getattr(ixformer, attr_name)
            if hasattr(obj, 'topk_softmax'):
                print(f"  FOUND: ixformer.{attr_name}.topk_softmax")
            if hasattr(obj, 'moe_topk_softmax'):
                print(f"  FOUND: ixformer.{attr_name}.moe_topk_softmax")
    except:
        pass

def check_torch_ops():
    """检查 torch.ops 注册"""
    print("\n=== torch.ops Check ===")
    try:
        import torch
        # 检查是否有 ixformer 注册的 ops
        for ns in ['ixformer', '_ixformer', 'ixf', '_C']:
            try:
                ns_obj = getattr(torch.ops, ns, None)
                if ns_obj:
                    ops = [x for x in dir(ns_obj) if 'topk' in x.lower() or 'moe' in x.lower()]
                    if ops:
                        print(f"  torch.ops.{ns} MoE ops: {ops}")
                    else:
                        print(f"  torch.ops.{ns} exists but no MoE ops: {dir(ns_obj)[:10]}...")
            except:
                pass
    except:
        pass

def try_jit_compile():
    """尝试 JIT 编译 ix_moe_bridge.cpp 看链接是否成功"""
    print("\n=== JIT Compile Test ===")
    test_cpp = "/tmp/ix_probe_test.cpp"
    with open(test_cpp, "w") as f:
        f.write("""
#include <torch/extension.h>

// Forward-declare — this is what ix_moe_bridge.cpp needs
namespace ixformer { namespace infer {
void topk_softmax(torch::Tensor&, torch::Tensor&, torch::Tensor&, 
                  torch::Tensor&, bool);
}}

void test_link() {
    auto a = torch::empty({1,1});
    auto b = torch::empty({1,1});
    auto c = torch::empty({1,1});
    auto d = torch::empty({1,1});
    ixformer::infer::topk_softmax(a, b, c, d, false);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("test_link", &test_link);
}
""")
    try:
        from torch.utils.cpp_extension import load
        ext = load(name="ix_probe_test", sources=[test_cpp],
                   extra_cflags=["-O0"], verbose=True)
        print("  JIT COMPILE + LINK: SUCCESS ✓")
        print("  ixformer::infer::topk_softmax symbol resolved!")
        return True
    except Exception as e:
        err = str(e)
        if "undefined reference" in err or "undefined symbol" in err:
            print(f"  JIT LINK FAILED: symbol not found in .so")
            print(f"  Error: {err[:500]}")
        else:
            print(f"  JIT COMPILE FAILED: {err[:500]}")
        return False

if __name__ == "__main__":
    print("=" * 70)
    print("ixformer Symbol Probe")
    print("=" * 70)

    # Step 1: Find .so files
    print("\n=== .so Files ===")
    so_files = find_ixformer_so()
    if not so_files:
        print("  No ixformer .so files found!")
    for f in sorted(set(so_files)):
        size = os.path.getsize(f) if os.path.exists(f) else 0
        print(f"  {f} ({size/1024/1024:.1f} MB)")

    # Step 2: Search for symbols
    patterns = ["topk_softmax", "moe_topk", "topk_gating", 
                "moe_compute_token", "moe_expand", "moe_output_reduce",
                "moe_w16a16", "group_gemm"]
    print("\n=== Symbol Search (MoE-related) ===")
    found_any = False
    for f in sorted(set(so_files)):
        results = nm_grep(f, patterns)
        if results:
            found_any = True
            print(f"\n  {os.path.basename(f)}:")
            for r in results[:20]:
                print(f"    {r}")
    if not found_any:
        print("  No MoE symbols found in any .so")
        # Also search for ANY ixformer::infer symbols
        print("\n=== Symbol Search (ixformer::infer namespace) ===")
        for f in sorted(set(so_files)):
            results = nm_grep(f, ["ixformer", "infer"])
            if results:
                print(f"\n  {os.path.basename(f)} ({len(results)} matches):")
                for r in results[:30]:
                    print(f"    {r}")

    # Step 3: Python binding
    check_python_binding()

    # Step 4: torch.ops
    check_torch_ops()

    # Step 5: JIT compile test (the definitive answer)
    jit_ok = try_jit_compile()

    # Summary
    print("\n" + "=" * 70)
    if jit_ok:
        print("RESULT: ix_moe_bridge.cpp CAN link to ixformer::infer::topk_softmax")
        print("ACTION: proceed with C++ bridge approach")
    else:
        print("RESULT: ix_moe_bridge.cpp CANNOT link to ixformer C++ API")
        print("ACTION: need alternative — options:")
        print("  A) Build topk_softmax kernel from upstream_ref/xllm CUDA source")
        print("  B) Build from upstream_ref/ds_vllm/csrc/moe/topk_softmax_kernels.cu")
        print("  C) Keep PyTorch path but add explicit error logging")
    print("=" * 70)
