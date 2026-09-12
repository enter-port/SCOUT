#!/bin/bash
set -e
cd /root/workspace/baojiachun
export UV_CACHE_DIR=/root/workspace/baojiachun/.uv-cache
export VIRTUAL_ENV=/root/workspace/baojiachun/.venv
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
echo "[coredeps start $(date)]"
uv pip install --python .venv/bin/python --index-url $MIRROR \
  "numpy<2" "zarr<3" "mujoco<3" \
  diffusers==0.27.2 hydra-core einops scipy scikit-learn \
  opencv-python matplotlib tqdm huggingface-hub wandb
echo "[robosuite --no-deps $(date)]"
uv pip install --python .venv/bin/python --index-url $MIRROR --no-deps robosuite==1.4.1
echo "[robomimic --no-deps $(date)]"
uv pip install --python .venv/bin/python --index-url $MIRROR --no-deps -e dependencies/robomimic
echo "[coredeps done $(date)]"
echo COREDEPS_OK
