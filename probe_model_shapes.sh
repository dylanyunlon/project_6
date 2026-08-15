#!/bin/bash
set -e

echo "=== 模型权重实际shape ==="
python3 -c "
import torch, os, json
# 读config.json
cfg_path = '/model/config.json'
if os.path.exists(cfg_path):
    with open(cfg_path) as f:
        cfg = json.load(f)
    print('Model config:')
    for k in ['hidden_size', 'intermediate_size', 'num_attention_heads', 
              'num_key_value_heads', 'num_hidden_layers', 'num_experts',
              'num_experts_per_tok', 'moe_intermediate_size', 'vocab_size',
              'max_position_embeddings']:
        print(f'  {k}: {cfg.get(k, \"N/A\")}')
else:
    print(f'{cfg_path} not found')
    # 搜索
    import glob
    for p in glob.glob('/model/**/config.json', recursive=True):
        print(f'  found: {p}')
"

echo ""
echo "=== safetensor权重shape（第一个shard）==="
python3 -c "
from safetensors import safe_open
import glob, os
shards = sorted(glob.glob('/model/model*.safetensors'))
if not shards:
    shards = sorted(glob.glob('/model/*.safetensors'))
if shards:
    print(f'Found {len(shards)} shards, reading first: {shards[0]}')
    with safe_open(shards[0], framework='pt') as f:
        for key in sorted(f.keys()):
            if 'experts' in key and ('w1' in key or 'w2' in key or 'w13' in key):
                print(f'  {key}: {f.get_tensor(key).shape}')
                break  # 只看一个就够了
        # 也看gate
        for key in sorted(f.keys()):
            if 'gate' in key and 'weight' in key:
                print(f'  {key}: {f.get_tensor(key).shape}')
                break
else:
    print('No safetensor shards found')
" 2>&1 || echo "safetensors not available"
