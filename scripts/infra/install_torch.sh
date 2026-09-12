#!/bin/bash
set -e
cd /root/workspace/baojiachun
export UV_CACHE_DIR=/root/workspace/baojiachun/.uv-cache
export VIRTUAL_ENV=/root/workspace/baojiachun/.venv
echo "[start $(date)]"
echo "tuna http: $(curl -s -o /dev/null -w '%{http_code}' https://pypi.tuna.tsinghua.edu.cn/simple/torch/ || echo FAIL)"
uv pip install --python .venv/bin/python torch==2.4.1 torchvision==0.19.1 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple
echo "[done $(date)]"
.venv/bin/python -c "import torch; print('torch', torch.__version__); print('cuda_avail', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"
echo TORCH_VERIFY_OK
