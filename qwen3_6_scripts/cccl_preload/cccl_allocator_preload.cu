#include <cub/util_allocator.cuh>
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static constexpr unsigned int ALLOC_BIN_GROWTH = 2;
static constexpr unsigned int ALLOC_MIN_BIN    = 8;
static constexpr unsigned int ALLOC_MAX_BIN    = 32;
static constexpr size_t       ALLOC_MAX_CACHED = (size_t)8*1024*1024*1024;

using CubAllocator = CUB_NS_QUALIFIER::CachingDeviceAllocator;

static CubAllocator& get_allocator() {
    static CubAllocator instance(ALLOC_BIN_GROWTH, ALLOC_MIN_BIN, ALLOC_MAX_BIN, ALLOC_MAX_CACHED, true);
    return instance;
}

static bool g_preload_active = false;
static bool g_debug = false;
static thread_local bool inside_cub = false;

using RealMalloc_t = cudaError_t(*)(void**,size_t);
using RealFree_t   = cudaError_t(*)(void*);
static RealMalloc_t get_real_malloc(){ static RealMalloc_t fn=(RealMalloc_t)dlsym(RTLD_NEXT,"cudaMalloc"); return fn; }
static RealFree_t   get_real_free()  { static RealFree_t fn=(RealFree_t)dlsym(RTLD_NEXT,"cudaFree"); return fn; }

__attribute__((constructor))
static void cccl_preload_init() {
    g_debug = (getenv("CCCL_ALLOC_DEBUG") && atoi(getenv("CCCL_ALLOC_DEBUG"))>0);
    if (getenv("CCCL_ALLOC_DISABLE") && atoi(getenv("CCCL_ALLOC_DISABLE"))>0) return;
    const char* ac = getenv("PYTORCH_CUDA_ALLOC_CONF");
    if (ac) {
        std::string conf(ac), clean; size_t p=0;
        while(p<conf.size()){ size_t c=conf.find(',',p); if(c==std::string::npos)c=conf.size();
            std::string t=conf.substr(p,c-p); if(t.find("expandable_segments")==std::string::npos){if(!clean.empty())clean+=",";clean+=t;} p=c+1;}
        if(clean.empty()) unsetenv("PYTORCH_CUDA_ALLOC_CONF"); else setenv("PYTORCH_CUDA_ALLOC_CONF",clean.c_str(),1);
        fprintf(stderr,"[cccl_alloc] PYTORCH_CUDA_ALLOC_CONF: \"%s\" -> \"%s\"\n",ac,clean.empty()?"(unset)":clean.c_str());
    }
    auto& a=get_allocator(); if(g_debug)a.debug=true; g_preload_active=true;
    fprintf(stderr,"[cccl_alloc] LD_PRELOAD active — CUB CachingDeviceAllocator (growth=%u, bins=[%u..%u], max_cached=%.1fGB)\n",
        ALLOC_BIN_GROWTH,ALLOC_MIN_BIN,ALLOC_MAX_BIN,(double)ALLOC_MAX_CACHED/(1024.0*1024.0*1024.0));
}

extern "C" cudaError_t cudaMalloc(void** p, size_t s) {
    if(!g_preload_active||inside_cub) return get_real_malloc()(p,s);
    inside_cub=true; auto e=get_allocator().DeviceAllocate(p,s); inside_cub=false; return e;
}
extern "C" cudaError_t cudaFree(void* p) {
    if(!g_preload_active||inside_cub||!p) return get_real_free()(p);
    inside_cub=true; auto e=get_allocator().DeviceFree(p); inside_cub=false; return e;
}
