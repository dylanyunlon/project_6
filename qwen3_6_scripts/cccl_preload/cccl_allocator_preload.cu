/*
 * cccl_allocator_preload.cu
 *
 * LD_PRELOAD .so that replaces PyTorch's CUDA memory allocator with
 * CUB's CachingDeviceAllocator (extracted from CCCL upstream).
 *
 * Purpose: CoreX's CUDACachingAllocator.cpp:545 asserts
 * "expandable segment not supported". Instead of patching libtorch,
 * we intercept cudaMalloc/cudaFree at the dynamic linker level and
 * route them through CUB's battle-tested caching allocator.
 *
 * Source: cccl_upstream/cub/cub/util_allocator.cuh
 * License: BSD-3 (NVIDIA/CUB)
 *
 * Build (on BI-V100 with CoreX clang++):
 *   bash build_cccl_preload.sh
 *
 * Usage:
 *   LD_PRELOAD=/workspace/qwen3_6_scripts/cccl_preload/libcccl_allocator.so \
 *   CCCL_ALLOC_DEBUG=0 \
 *   PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
 *   python3 -m vllm.entrypoints.openai.api_server ...
 */

#include <cuda_runtime_api.h>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <set>

/* ========================================================================
 * CUB CachingDeviceAllocator — extracted from CCCL
 * cccl_upstream/cub/cub/util_allocator.cuh
 *
 * All CUB/CCCL macro dependencies replaced with plain C++.
 * ======================================================================== */

static bool g_cccl_debug = false;

#define CcclDebug(e) (e)
#define CcclLog(...)               \
    do {                           \
        if (g_cccl_debug) {        \
            fprintf(stderr, "[cccl_alloc] "); \
            fprintf(stderr, __VA_ARGS__); \
        }                          \
    } while (0)

struct CachingDeviceAllocator
{
    static constexpr unsigned int INVALID_BIN = (unsigned int) -1;
    static constexpr size_t INVALID_SIZE = (size_t) -1;
    static constexpr int INVALID_DEVICE_ORDINAL = -1;

    struct BlockDescriptor
    {
        void* d_ptr;
        size_t bytes;
        unsigned int bin;
        int device;
        cudaStream_t associated_stream;
        cudaEvent_t ready_event;

        BlockDescriptor(void* d_ptr_, int device_)
            : d_ptr(d_ptr_), bytes(0), bin(INVALID_BIN), device(device_),
              associated_stream(nullptr), ready_event(nullptr) {}

        BlockDescriptor(int device_)
            : d_ptr(nullptr), bytes(0), bin(INVALID_BIN), device(device_),
              associated_stream(nullptr), ready_event(nullptr) {}

        static bool PtrCompare(const BlockDescriptor& a, const BlockDescriptor& b) {
            return (a.device == b.device) ? (a.d_ptr < b.d_ptr) : (a.device < b.device);
        }
        static bool SizeCompare(const BlockDescriptor& a, const BlockDescriptor& b) {
            return (a.device == b.device) ? (a.bytes < b.bytes) : (a.device < b.device);
        }
    };

    using Compare = bool (*)(const BlockDescriptor&, const BlockDescriptor&);

    struct TotalBytes { size_t free; size_t live; TotalBytes() : free(0), live(0) {} };

    using CachedBlocks = std::multiset<BlockDescriptor, Compare>;
    using BusyBlocks = std::multiset<BlockDescriptor, Compare>;
    using GpuCachedBytes = std::map<int, TotalBytes>;

    static unsigned int IntPow(unsigned int base, unsigned int exp) {
        unsigned int retval = 1;
        while (exp > 0) {
            if (exp & 1) retval *= base;
            base *= base;
            exp >>= 1;
        }
        return retval;
    }

    void NearestPowerOf(unsigned int& power, size_t& rounded_bytes,
                        unsigned int base, size_t value) {
        power = 0;
        rounded_bytes = 1;
        if (value * base < value) {
            power = sizeof(size_t) * 8;
            rounded_bytes = size_t(0) - 1;
            return;
        }
        while (rounded_bytes < value) {
            rounded_bytes *= base;
            power++;
        }
    }

    std::mutex mutex;
    unsigned int bin_growth;
    unsigned int min_bin;
    unsigned int max_bin;
    size_t min_bin_bytes;
    size_t max_bin_bytes;
    size_t max_cached_bytes;
    bool skip_cleanup;
    GpuCachedBytes cached_bytes;
    CachedBlocks cached_blocks;
    BusyBlocks live_blocks;

    /*
     * Constructor tuned for BI-V100 (32GB per card, 4 cards):
     *   bin_growth=8, min_bin=3 (512B), max_bin=13 (~550MB)
     *   max_cached_bytes = 4GB per device (reasonable for 32GB card)
     *
     * This replaces PyTorch's expandable_segments with a proven
     * geometric-bin caching strategy from CUB/CCCL.
     */
    CachingDeviceAllocator()
        : bin_growth(8)
        , min_bin(3)           /* 8^3 = 512B minimum allocation */
        , max_bin(13)          /* 8^13 = ~550MB maximum cached bin */
        , min_bin_bytes(IntPow(8, 3))
        , max_bin_bytes(IntPow(8, 13))
        , max_cached_bytes((size_t)4 * 1024 * 1024 * 1024)  /* 4GB per device */
        , skip_cleanup(true)   /* CoreX may tear down CUDA before our dtor */
        , cached_blocks(BlockDescriptor::SizeCompare)
        , live_blocks(BlockDescriptor::PtrCompare)
    {
        CcclLog("CachingDeviceAllocator init: bin_growth=%u min_bin=%u "
                "max_bin=%u max_cached=%.1fGB\n",
                bin_growth, min_bin, max_bin,
                (double)max_cached_bytes / (1024.0*1024.0*1024.0));
    }

    /* ---- Real cudaMalloc/cudaFree via dlsym(RTLD_NEXT) ---- */
    using RealMalloc_t = cudaError_t (*)(void**, size_t);
    using RealFree_t = cudaError_t (*)(void*);

    static RealMalloc_t get_real_malloc() {
        static RealMalloc_t fn = (RealMalloc_t)dlsym(RTLD_NEXT, "cudaMalloc");
        return fn;
    }
    static RealFree_t get_real_free() {
        static RealFree_t fn = (RealFree_t)dlsym(RTLD_NEXT, "cudaFree");
        return fn;
    }

    cudaError_t DeviceAllocate(int device, void** d_ptr, size_t bytes,
                               cudaStream_t active_stream = nullptr)
    {
        *d_ptr = nullptr;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error = cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
            error = cudaGetDevice(&entrypoint_device);
            if (error != cudaSuccess) return error;
            device = entrypoint_device;
        }

        bool found = false;
        BlockDescriptor search_key(device);
        search_key.associated_stream = active_stream;
        NearestPowerOf(search_key.bin, search_key.bytes, bin_growth, bytes);

        if (search_key.bin > max_bin) {
            search_key.bin = INVALID_BIN;
            search_key.bytes = bytes;
        } else {
            mutex.lock();
            if (search_key.bin < min_bin) {
                search_key.bin = min_bin;
                search_key.bytes = min_bin_bytes;
            }

            CachedBlocks::iterator block_itr = cached_blocks.lower_bound(search_key);
            while ((block_itr != cached_blocks.end()) &&
                   (block_itr->device == device) &&
                   (block_itr->bin == search_key.bin))
            {
                bool is_reusable = false;
                if (active_stream == block_itr->associated_stream) {
                    is_reusable = true;
                } else {
                    cudaError_t event_status = cudaEventQuery(block_itr->ready_event);
                    if (event_status != cudaErrorNotReady) {
                        is_reusable = true;
                    }
                }

                if (is_reusable) {
                    found = true;
                    search_key = *block_itr;
                    search_key.associated_stream = active_stream;
                    live_blocks.insert(search_key);
                    cached_bytes[device].free -= search_key.bytes;
                    cached_bytes[device].live += search_key.bytes;

                    CcclLog("reuse %p (%zu bytes) dev=%d\n",
                            search_key.d_ptr, search_key.bytes, device);
                    cached_blocks.erase(block_itr);
                    break;
                }
                block_itr++;
            }
            mutex.unlock();
        }

        if (!found) {
            if (device != entrypoint_device) {
                if (entrypoint_device == INVALID_DEVICE_ORDINAL)
                    cudaGetDevice(&entrypoint_device);
                cudaSetDevice(device);
            }

            /* Use real cudaMalloc, not ourselves */
            error = get_real_malloc()(&search_key.d_ptr, search_key.bytes);

            if (error == cudaErrorMemoryAllocation) {
                CcclLog("OOM for %zu bytes on dev=%d, freeing cache...\n",
                        search_key.bytes, device);
                cudaGetLastError();  /* reset */

                mutex.lock();
                BlockDescriptor free_key(device);
                CachedBlocks::iterator block_itr = cached_blocks.lower_bound(free_key);
                while ((block_itr != cached_blocks.end()) &&
                       (block_itr->device == device))
                {
                    error = get_real_free()(block_itr->d_ptr);
                    if (error != cudaSuccess) break;
                    cudaEventDestroy(block_itr->ready_event);
                    cached_bytes[device].free -= block_itr->bytes;
                    block_itr = cached_blocks.erase(block_itr);
                }
                mutex.unlock();

                if (error != cudaSuccess) return error;
                error = get_real_malloc()(&search_key.d_ptr, search_key.bytes);
                if (error != cudaSuccess) return error;
            } else if (error != cudaSuccess) {
                return error;
            }

            cudaEventCreateWithFlags(&search_key.ready_event, cudaEventDisableTiming);

            mutex.lock();
            live_blocks.insert(search_key);
            cached_bytes[device].live += search_key.bytes;
            mutex.unlock();

            CcclLog("alloc %p (%zu bytes, bin=%u) dev=%d\n",
                    search_key.d_ptr, search_key.bytes, search_key.bin, device);

            if ((entrypoint_device != INVALID_DEVICE_ORDINAL) &&
                (entrypoint_device != device))
                cudaSetDevice(entrypoint_device);
        }

        *d_ptr = search_key.d_ptr;
        return cudaSuccess;
    }

    cudaError_t DeviceAllocate(void** d_ptr, size_t bytes,
                               cudaStream_t active_stream = nullptr) {
        return DeviceAllocate(INVALID_DEVICE_ORDINAL, d_ptr, bytes, active_stream);
    }

    cudaError_t DeviceFree(int device, void* d_ptr)
    {
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        cudaError_t error = cudaSuccess;

        if (d_ptr == nullptr) return cudaSuccess;

        if (device == INVALID_DEVICE_ORDINAL) {
            error = cudaGetDevice(&entrypoint_device);
            if (error != cudaSuccess) return error;
            device = entrypoint_device;
        }

        mutex.lock();
        bool recached = false;
        BlockDescriptor search_key(d_ptr, device);
        BusyBlocks::iterator block_itr = live_blocks.find(search_key);

        if (block_itr != live_blocks.end()) {
            search_key = *block_itr;
            live_blocks.erase(block_itr);
            cached_bytes[device].live -= search_key.bytes;

            if ((search_key.bin != INVALID_BIN) &&
                (cached_bytes[device].free + search_key.bytes <= max_cached_bytes))
            {
                recached = true;
                cached_blocks.insert(search_key);
                cached_bytes[device].free += search_key.bytes;
                CcclLog("cache %p (%zu bytes) dev=%d\n",
                        d_ptr, search_key.bytes, device);
            }
        }
        mutex.unlock();

        if (device != entrypoint_device) {
            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
                cudaGetDevice(&entrypoint_device);
            cudaSetDevice(device);
        }

        if (recached) {
            cudaEventRecord(search_key.ready_event, search_key.associated_stream);
        } else {
            /* Not tracked or cache full — real free */
            CcclLog("free %p dev=%d (not cached)\n", d_ptr, device);
            error = get_real_free()(d_ptr);
            if (block_itr != live_blocks.end())
                cudaEventDestroy(search_key.ready_event);
        }

        if ((entrypoint_device != INVALID_DEVICE_ORDINAL) &&
            (entrypoint_device != device))
            cudaSetDevice(entrypoint_device);

        return error;
    }

    cudaError_t DeviceFree(void* d_ptr) {
        return DeviceFree(INVALID_DEVICE_ORDINAL, d_ptr);
    }

    cudaError_t FreeAllCached()
    {
        cudaError_t error = cudaSuccess;
        int entrypoint_device = INVALID_DEVICE_ORDINAL;
        int current_device = INVALID_DEVICE_ORDINAL;

        mutex.lock();
        while (!cached_blocks.empty()) {
            CachedBlocks::iterator begin = cached_blocks.begin();
            if (entrypoint_device == INVALID_DEVICE_ORDINAL)
                cudaGetDevice(&entrypoint_device);
            if (begin->device != current_device) {
                cudaSetDevice(begin->device);
                current_device = begin->device;
            }
            get_real_free()(begin->d_ptr);
            cudaEventDestroy(begin->ready_event);
            cached_bytes[current_device].free -= begin->bytes;
            cached_blocks.erase(begin);
        }
        mutex.unlock();

        if (entrypoint_device != INVALID_DEVICE_ORDINAL)
            cudaSetDevice(entrypoint_device);
        return error;
    }
};

/* ========================================================================
 * Global singleton + LD_PRELOAD intercepts
 * ======================================================================== */

static CachingDeviceAllocator& get_allocator() {
    static CachingDeviceAllocator instance;
    return instance;
}

static bool g_preload_active = false;

/* Called once at .so load time */
__attribute__((constructor))
static void cccl_preload_init() {
    const char* debug_env = getenv("CCCL_ALLOC_DEBUG");
    g_cccl_debug = (debug_env && atoi(debug_env) > 0);

    const char* disable_env = getenv("CCCL_ALLOC_DISABLE");
    if (disable_env && atoi(disable_env) > 0) {
        fprintf(stderr, "[cccl_alloc] DISABLED by CCCL_ALLOC_DISABLE=1\n");
        g_preload_active = false;
        return;
    }

    /* Strip expandable_segments from PYTORCH_CUDA_ALLOC_CONF
     * so CoreX's allocator doesn't hit the assert.
     * We handle the caching ourselves. */
    const char* alloc_conf = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (alloc_conf) {
        /* Build a new conf string without expandable_segments */
        std::string conf(alloc_conf);
        std::string clean;
        size_t pos = 0;
        while (pos < conf.size()) {
            size_t comma = conf.find(',', pos);
            if (comma == std::string::npos) comma = conf.size();
            std::string token = conf.substr(pos, comma - pos);
            /* Skip expandable_segments:* */
            if (token.find("expandable_segments") == std::string::npos) {
                if (!clean.empty()) clean += ",";
                clean += token;
            }
            pos = comma + 1;
        }
        if (clean.empty()) {
            unsetenv("PYTORCH_CUDA_ALLOC_CONF");
        } else {
            setenv("PYTORCH_CUDA_ALLOC_CONF", clean.c_str(), 1);
        }
        fprintf(stderr, "[cccl_alloc] stripped expandable_segments from "
                "PYTORCH_CUDA_ALLOC_CONF: \"%s\" -> \"%s\"\n",
                alloc_conf, clean.empty() ? "(unset)" : clean.c_str());
    }

    /* Force-initialize the allocator singleton */
    (void)get_allocator();
    g_preload_active = true;
    fprintf(stderr, "[cccl_alloc] LD_PRELOAD active — CUB CachingDeviceAllocator "
            "replacing cudaMalloc/cudaFree\n");
}

/* ---- cudaMalloc intercept ---- */
extern "C" cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    if (!g_preload_active) {
        /* Fallback to real cudaMalloc during init or if disabled */
        static auto real_fn = (CachingDeviceAllocator::RealMalloc_t)
            dlsym(RTLD_NEXT, "cudaMalloc");
        return real_fn(devPtr, size);
    }
    return get_allocator().DeviceAllocate(devPtr, size);
}

/* ---- cudaFree intercept ---- */
extern "C" cudaError_t cudaFree(void* devPtr)
{
    if (!g_preload_active || devPtr == nullptr) {
        static auto real_fn = (CachingDeviceAllocator::RealFree_t)
            dlsym(RTLD_NEXT, "cudaFree");
        return real_fn(devPtr);
    }
    return get_allocator().DeviceFree(devPtr);
}
