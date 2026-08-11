#!/usr/bin/env bash
# Run this on the real machine to simulate Docker build steps and find failures.
# Usage: bash diagnose_build.sh

set +e  # Don't exit on errors

echo "=== STEP 1: ex_engine build.sh ==="
cd /home/dylan/project_6
chmod +x ex_engine/build.sh
bash ex_engine/build.sh --corex 2>&1 | tail -10
echo "EXIT: $?"

echo ""
echo "=== STEP 2: precompile_moe_topk ==="
python3 ex_engine/precompile_moe_topk.py 2>&1 | tail -10
echo "EXIT: $?"

echo ""
echo "=== STEP 3: precompile_moe_kernels ==="
python3 ex_engine/precompile_moe_kernels.py 2>&1 | tail -10
echo "EXIT: $?"

echo ""
echo "=== STEP 4: patch_ops.sh ==="
cd qwen3_6_scripts
chmod +x patch_ops.sh
bash patch_ops.sh 2>&1 | tail -20
echo "EXIT: $?"

echo ""
echo "=== STEP 5: precompile_gdn ==="
cd /home/dylan/project_6
python3 qwen3_6_scripts/precompile_gdn.py qwen3_6_scripts/flash_qla_sm70 2>&1 | tail -10
echo "EXIT: $?"

echo ""
echo "=== STEP 6: Test qwen3_5.py import ==="
python3 -c "
import sys
sys.path.insert(0, '/usr/local/corex/lib64/python3/dist-packages')
sys.path.insert(0, '/usr/local/corex/lib/python3/dist-packages')
try:
    # This is what happens at runtime when vllm loads the model
    exec(open('/home/dylan/project_6/qwen3_6_scripts/qwen3_5.py').read())
    print('IMPORT OK')
except Exception as e:
    print(f'IMPORT FAIL: {type(e).__name__}: {e}')
" 2>&1 | tail -10
echo "EXIT: $?"

echo ""
echo "=== DONE ==="
