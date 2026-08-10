"""
Pre-compile SM70 GDN CUDA kernel → .so at Docker build time.
Avoids 2-minute JIT delay at runtime.

Usage: python3 precompile_gdn.py /path/to/flash_qla_sm70/
"""
import os
import sys

def main():
    if len(sys.argv) < 2:
        print("[precompile] Usage: python3 precompile_gdn.py <flash_qla_sm70_dir>")
        sys.exit(1)

    flash_dir = sys.argv[1]
    cu_src = os.path.join(flash_dir, "csrc", "gdn_forward.cu")
    if not os.path.exists(cu_src):
        print(f"[precompile] ERROR: {cu_src} not found")
        sys.exit(1)

    # Set arch for BI-V100 (SM70 compatible)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.0;7.5"

    build_dir = os.path.join(flash_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    print(f"[precompile] Compiling {cu_src} → .so in {build_dir}")
    print(f"[precompile] TORCH_CUDA_ARCH_LIST = {os.environ['TORCH_CUDA_ARCH_LIST']}")

    try:
        from torch.utils.cpp_extension import load
        ext = load(
            name="flash_qla_sm70_gdn_strided",
            sources=[cu_src],
            extra_cuda_cflags=["-O3"],
            extra_cflags=["-O3"],
            build_directory=build_dir,
            verbose=True,
        )
        print(f"[precompile] SUCCESS — compiled .so in {build_dir}")
        # List the built files
        for f in os.listdir(build_dir):
            if f.endswith(".so"):
                full = os.path.join(build_dir, f)
                print(f"[precompile]   {f} ({os.path.getsize(full)} bytes)")
    except Exception as e:
        print(f"[precompile] FAILED: {e}")
        print("[precompile] Kernel will JIT compile at runtime instead (~2min)")
        sys.exit(1)


if __name__ == "__main__":
    main()
