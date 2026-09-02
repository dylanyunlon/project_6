/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// Commit: 494f293b5629 · feat · PR #2260  (adapted for Iluvatar BI-V100)
// gflag definition for enabling/disabling layerwise split KV cache.
//
// Usage:
//   --enable_layerwise_split=true   (enable the feature)
//   --enable_layerwise_split=false  (default — uniform sharding, no change)

#include <gflags/gflags.h>

DEFINE_bool(enable_layerwise_split, false,
            "Enable layerwise-split KV cache sharding.  When true, each "
            "layer's KV cache is independently sharded across a configurable "
            "subset of TP ranks, allowing dense attention layers to spread "
            "across all ranks while MoE layers (few KV heads, GQA) "
            "concentrate on fewer ranks.  Requires a heterogeneous-layer "
            "model (e.g. DeepSeek-V3).  Default: false (uniform sharding).");
