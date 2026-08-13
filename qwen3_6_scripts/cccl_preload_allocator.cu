// cccl_preload_allocator.cu — LD_PRELOAD interception of cudaMalloc/cudaFree
//
// Replaces the default cudaMalloc/cudaFree with CCCL CUB CachingDeviceAllocator.
// This eliminates the CUDA driver's allocation overhead (cudaMalloc is slow on
// BI-V100: ~2-50ms per call) by reusing freed blocks from a bin-based cache.
//
// Build on BI-V100:
//   /usr/local/corex-3.2.3/bin/clang++ -std=c++17 -O3 -shared -fPIC \
//     --cuda-path=/usr/local/corex-3.2.3 --cuda-gpu-arch=ivcore10 \
//     --no-cuda-version-check -D_GLIBCXX_USE_CXX11_ABI=0 \
//     -I/usr/local/corex-3.2.3/include \
//     cccl_preload_allocator.cu \
//     -L/usr/local/corex-3.2.3/lib64 -lcudart -ldl \
//     -o cccl_preload_allocator.so
//
// Usage:
//   LD_PRELOAD=/workspace/qwen3_6_scripts/cccl_preload_allocator.so python3 -m vllm.entrypoints.openai.api_server ...
//
// Tuning (env vars):
//   CCCL_ALLOC_BIN_GROWTH=8      Geometric growth factor (default 8)
//   CCCL_ALLOC_MIN_BIN=3         Min bin exponent (default 3 → 512B)
//   CCCL_ALLOC_MAX_BIN=13        Max bin exponent (default 13 → 512MB for growth=8; was 7→2MB)
//   CCCL_ALLOC_MAX_CACHED_MB=4096  Max cached bytes per device in MB (default 4096=4GB)
//   CCCL_ALLOC_DEBUG=0           Print alloc/free events (default 0)
//
// Design notes:
// - Only intercepts cudaMalloc and cudaFree (the synchronous variants).
// - cudaMallocAsync/cudaFreeAsync are NOT intercepted (PyTorch on BI-V100
//   doesn't use them; the corex runtime may not support them).
// - Thread-safe via CUB's internal mutex.
// - Stream association: all allocations use the default stream (nullptr).
//   PyTorch's CUDACachingAllocator handles stream ordering itself, so we
//   don't need to track streams here.
// - Large allocations (> max_bin_bytes) pass through to real cudaMalloc.
// - The allocator is process-global (static singleton).

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <mutex>
#include <map>
#include <set>
#include <atomic>

// We inline the essential logic from CUB CachingDeviceAllocator rather than
// #include it, because the corex toolchain may not have full CCCL headers
// installed, and we need to link against corex's cudart, not NVIDIA's.

// Forward declare the real CUDA functions we'll dlsym
typedef int cudaError_t;
static constexpr cudaError_t cudaSuccess = 0;

typedef void* cudaStream_t;
typedef void* cudaEvent_t;

// Real function pointers (resolved via dlsym on first call)
using cudaMalloc_fn = cudaError_t(*)(void**, size_t);
using cudaFree_fn = cudaError_t(*)(void*);
using cudaGetDevice_fn = cudaError_t(*)(int*);
using cudaEventCreate_fn = cudaError_t(*)(cudaEvent_t*);
using cudaEventRecord_fn = cudaError_t(*)(cudaEvent_t, cudaStream_t);
using cudaEventQuery_fn = cudaError_t(*)(cudaEvent_t);
using cudaEventDestroy_fn = cudaError_t(*)(cudaEvent_t);
using cudaEventSynchronize_fn = cudaError_t(*)(cudaEvent_t);

static cudaMalloc_fn  real_cudaMalloc  = nullptr;
static cudaFree_fn    real_cudaFree    = nullptr;
static cudaGetDevice_fn real_cudaGetDevice = nullptr;
static cudaEventCreate_fn real_cudaEventCreate = nullptr;
static cudaEventRecord_fn real_cudaEventRecord = nullptr;
static cudaEventQuery_fn  real_cudaEventQuery  = nullptr;
static cudaEventDestroy_fn real_cudaEventDestroy = nullptr;
static cudaEventSynchronize_fn real_cudaEventSynchronize = nullptr;

static std::once_flag resolve_flag;

static void resolve_real_functions() {
    real_cudaMalloc = (cudaMalloc_fn)dlsym(RTLD_NEXT, "cudaMalloc");
    real_cudaFree = (cudaFree_fn)dlsym(RTLD_NEXT, "cudaFree");
    real_cudaGetDevice = (cudaGetDevice_fn)dlsym(RTLD_NEXT, "cudaGetDevice");
    real_cudaEventCreate = (cudaEventCreate_fn)dlsym(RTLD_NEXT, "cudaEventCreate");
    real_cudaEventRecord = (cudaEventRecord_fn)dlsym(RTLD_NEXT, "cudaEventRecord");
    real_cudaEventQuery = (cudaEventQuery_fn)dlsym(RTLD_NEXT, "cudaEventQuery");
    real_cudaEventDestroy = (cudaEventDestroy_fn)dlsym(RTLD_NEXT, "cudaEventDestroy");
    real_cudaEventSynchronize = (cudaEventSynchronize_fn)dlsym(RTLD_NEXT, "cudaEventSynchronize");

    if (!real_cudaMalloc || !real_cudaFree) {
        fprintf(stderr, "[CCCL_PRELOAD] FATAL: cannot resolve cudaMalloc/cudaFree via dlsym\n");
        abort();
    }
}

// ============================================================================
// Simplified CachingDeviceAllocator (from CCCL cub/util_allocator.cuh)
// Stripped to essentials: no debug logging macros, no CCCL config dependencies
// ============================================================================

struct BlockDescriptor {
    void*         d_ptr;
    size_t        bytes;
    unsigned int  bin;
    int           device;
    cudaStream_t  associated_stream;
    cudaEvent_t   ready_event;

    BlockDescriptor(void* p, int dev)
        : d_ptr(p), bytes(0), bin(~0u), device(dev),
          associated_stream(nullptr), ready_event(nullptr) {}

    BlockDescriptor(int dev)
        : d_ptr(nullptr), bytes(0), bin(~0u), device(dev),
          associated_stream(nullptr), ready_event(nullptr) {}

    static bool PtrCompare(const BlockDescriptor& a, const BlockDescriptor& b) {
        return (a.device == b.device) ? (a.d_ptr < b.d_ptr) : (a.device < b.device);
    }
    static bool SizeCompare(const BlockDescriptor& a, const BlockDescriptor& b) {
        return (a.device == b.device) ? (a.bytes < b.bytes) : (a.device < b.device);
    }
};

using Compare = bool(*)(const BlockDescriptor&, const BlockDescriptor&);
using CachedBlocks = std::multiset<BlockDescriptor, Compare>;
using BusyBlocks = std::multiset<BlockDescriptor, Compare>;

struct DeviceBytes { size_t free = 0; size_t live = 0; };

struct CachingAllocator {
    std::mutex     mtx;
    unsigned int   bin_growth;
    unsigned int   min_bin;
    unsigned int   max_bin;
    size_t         min_bin_bytes;
    size_t         max_bin_bytes;
    size_t         max_cached_bytes;
    bool           debug;
    CachedBlocks   cached_blocks;
    BusyBlocks     live_blocks;
    std::map<int, DeviceBytes> cached_bytes;

    // Stats
    std::atomic<uint64_t> stat_hits{0};
    std::atomic<uint64_t> stat_misses{0};
    std::atomic<uint64_t> stat_frees{0};
    std::atomic<uint64_t> stat_bypasses{0};

    static unsigned int IntPow(unsigned int base, unsigned int exp) {
        unsigned int r = 1;
        while (exp > 0) {
            if (exp & 1) r *= base;
            base *= base;
            exp >>= 1;
        }
        return r;
    }

    void NearestPowerOf(unsigned int& power, size_t& rounded,
                        unsigned int base, size_t value) {
        power = 0; rounded = 1;
        if (value * base < value) {
            power = sizeof(size_t) * 8;
            rounded = size_t(-1);
            return;
        }
        while (rounded < value) { rounded *= base; power++; }
    }

    CachingAllocator()
        : cached_blocks(BlockDescriptor::SizeCompare),
          live_blocks(BlockDescriptor::PtrCompare) {
        // Read config from env
        auto env_or = [](const char* name, int def) -> int {
            const char* v = getenv(name);
            return v ? atoi(v) : def;
        };

        bin_growth      = env_or("CCCL_ALLOC_BIN_GROWTH", 8);
        min_bin         = env_or("CCCL_ALLOC_MIN_BIN", 3);
        max_bin         = env_or("CCCL_ALLOC_MAX_BIN", 13);
        int max_mb      = env_or("CCCL_ALLOC_MAX_CACHED_MB", 4096);
        debug           = env_or("CCCL_ALLOC_DEBUG", 0) != 0;
        min_bin_bytes   = IntPow(bin_growth, min_bin);
        max_bin_bytes   = IntPow(bin_growth, max_bin);
        max_cached_bytes = (size_t)max_mb * 1024ULL * 1024ULL;

        fprintf(stderr, "[CCCL_PRELOAD] CachingDeviceAllocator: growth=%u "
                "bins=[%u..%u] bin_bytes=[%zu..%zu] max_cached=%zuMB\n",
                bin_growth, min_bin, max_bin,
                min_bin_bytes, max_bin_bytes, max_cached_bytes / (1024*1024));
    }

    ~CachingAllocator() {
        fprintf(stderr, "[CCCL_PRELOAD] Stats: hits=%lu misses=%lu frees=%lu bypasses=%lu\n",
                stat_hits.load(), stat_misses.load(),
                stat_frees.load(), stat_bypasses.load());
        // Free all cached blocks
        for (auto& b : cached_blocks) {
            if (b.ready_event) real_cudaEventDestroy(b.ready_event);
            real_cudaFree(b.d_ptr);
        }
    }

    cudaError_t Allocate(void** d_ptr, size_t bytes) {
        std::call_once(resolve_flag, resolve_real_functions);
        *d_ptr = nullptr;

        // Get current device
        int device = 0;
        if (real_cudaGetDevice) real_cudaGetDevice(&device);

        // Bin classification
        unsigned int bin;
        size_t rounded_bytes;
        bool oversized = false;

        if (bytes > max_bin_bytes) {
            // Too large for caching — pass through
            bin = max_bin + 1;
            rounded_bytes = bytes;
            oversized = true;
        } else {
            NearestPowerOf(bin, rounded_bytes, bin_growth, bytes);
            if (bin < min_bin) {
                bin = min_bin;
                rounded_bytes = min_bin_bytes;
            }
        }

        BlockDescriptor search_key(device);
        search_key.bytes = rounded_bytes;
        search_key.bin = bin;

        // Lock
        std::lock_guard<std::mutex> lock(mtx);

        if (!oversized) {
            // Search cached blocks for a match
            auto range = cached_blocks.equal_range(search_key);
            for (auto it = range.first; it != range.second; ++it) {
                if (it->device == device && it->bin == bin) {
                    // Check if the stream work has completed
                    bool ready = true;
                    if (it->ready_event) {
                        cudaError_t ev_status = real_cudaEventQuery(it->ready_event);
                        if (ev_status != cudaSuccess) {
                            // Event not ready — try to synchronize briefly
                            // For BI-V100 with enforce_eager, events should be ready
                            real_cudaEventSynchronize(it->ready_event);
                        }
                        real_cudaEventDestroy(it->ready_event);
                    }

                    // Reuse this block
                    search_key.d_ptr = it->d_ptr;
                    search_key.bytes = it->bytes;
                    live_blocks.insert(search_key);
                    cached_bytes[device].free -= it->bytes;
                    cached_bytes[device].live += it->bytes;
                    cached_blocks.erase(it);

                    *d_ptr = search_key.d_ptr;
                    stat_hits++;

                    if (debug) {
                        fprintf(stderr, "[CCCL_PRELOAD] HIT  dev=%d bin=%u "
                                "req=%zu alloc=%zu ptr=%p\n",
                                device, bin, bytes, search_key.bytes, *d_ptr);
                    }
                    return cudaSuccess;
                }
            }
        }

        // Cache miss — allocate new block
        cudaError_t err = real_cudaMalloc(&search_key.d_ptr, rounded_bytes);

        // If OOM, try evicting cached blocks and retry
        if (err != cudaSuccess) {
            // Free all cached blocks on this device
            auto it = cached_blocks.begin();
            while (it != cached_blocks.end()) {
                if (it->device == device) {
                    if (it->ready_event) {
                        real_cudaEventSynchronize(it->ready_event);
                        real_cudaEventDestroy(it->ready_event);
                    }
                    real_cudaFree(it->d_ptr);
                    cached_bytes[device].free -= it->bytes;
                    it = cached_blocks.erase(it);
                } else {
                    ++it;
                }
            }
            // Retry
            err = real_cudaMalloc(&search_key.d_ptr, rounded_bytes);
        }

        if (err != cudaSuccess) {
            return err;
        }

        search_key.bytes = rounded_bytes;
        live_blocks.insert(search_key);
        cached_bytes[device].live += rounded_bytes;

        *d_ptr = search_key.d_ptr;

        if (oversized) {
            stat_bypasses++;
        } else {
            stat_misses++;
        }

        if (debug) {
            fprintf(stderr, "[CCCL_PRELOAD] %s dev=%d bin=%u "
                    "req=%zu alloc=%zu ptr=%p\n",
                    oversized ? "PASS" : "MISS",
                    device, bin, bytes, rounded_bytes, *d_ptr);
        }
        return cudaSuccess;
    }

    cudaError_t Free(void* d_ptr) {
        std::call_once(resolve_flag, resolve_real_functions);

        if (d_ptr == nullptr) return cudaSuccess;

        int device = 0;
        if (real_cudaGetDevice) real_cudaGetDevice(&device);

        BlockDescriptor search_key(d_ptr, device);

        std::lock_guard<std::mutex> lock(mtx);

        auto it = live_blocks.find(search_key);
        if (it == live_blocks.end()) {
            // Not tracked by us — pass through to real cudaFree
            return real_cudaFree(d_ptr);
        }

        search_key.bytes = it->bytes;
        search_key.bin = it->bin;
        cached_bytes[device].live -= it->bytes;
        live_blocks.erase(it);

        stat_frees++;

        // Check if this block is too large or would exceed cache limit
        bool should_cache = (search_key.bin <= max_bin) &&
                            (cached_bytes[device].free + search_key.bytes <= max_cached_bytes);

        if (should_cache) {
            // Record an event so we know when it's safe to reuse
            if (real_cudaEventCreate) {
                cudaEvent_t event = nullptr;
                cudaError_t ev_err = real_cudaEventCreate(&event);
                if (ev_err == cudaSuccess && real_cudaEventRecord) {
                    real_cudaEventRecord(event, nullptr);  // default stream
                    search_key.ready_event = event;
                }
            }

            cached_blocks.insert(search_key);
            cached_bytes[device].free += search_key.bytes;

            if (debug) {
                fprintf(stderr, "[CCCL_PRELOAD] CACHE dev=%d bin=%u "
                        "bytes=%zu cached_free=%zu\n",
                        device, search_key.bin, search_key.bytes,
                        cached_bytes[device].free);
            }
            return cudaSuccess;
        } else {
            // Don't cache — actually free
            if (debug) {
                fprintf(stderr, "[CCCL_PRELOAD] FREE  dev=%d bin=%u bytes=%zu\n",
                        device, search_key.bin, search_key.bytes);
            }
            return real_cudaFree(d_ptr);
        }
    }
};

// Global singleton
static CachingAllocator& get_allocator() {
    static CachingAllocator alloc;
    return alloc;
}

// ============================================================================
// LD_PRELOAD interception points
// ============================================================================

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    return get_allocator().Allocate(devPtr, size);
}

cudaError_t cudaFree(void* devPtr) {
    return get_allocator().Free(devPtr);
}

}  // extern "C"
