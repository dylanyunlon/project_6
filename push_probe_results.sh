#!/bin/bash
# 在真机上执行：把probe结果和.so文件commit到repo
set -e

cd /home/dylan/0814/project_6

# 1. 先跑第二个probe（如果还没跑的话）
if [ ! -f probe_bridge_output.txt ]; then
    echo "[1/4] Running probe_ix_unified_bridge.sh..."
    bash probe_ix_unified_bridge.sh 2>&1 | tee probe_bridge_output.txt
else
    echo "[1/4] probe_bridge_output.txt already exists"
fi

# 2. commit probe结果（不commit .so文件，太大了）
echo "[2/4] Committing probe results..."
git add probe_bridge_output.txt
git add -f probe_output.txt 2>/dev/null || true
git commit -m "data: probe results — ixformer API + ix_unified_bridge + corex_*.so函数列表" || echo "nothing to commit"

# 3. push到modelhub
echo "[3/4] Pushing to modelhub..."
git push origin main

# 4. 提示中转机操作
echo ""
echo "[4/4] 现在去中转机执行:"
echo "  cd /home/dylan/Downloads/github_0804/project_6"
echo "  git pull modelhub main"
echo "  git push origin main"
