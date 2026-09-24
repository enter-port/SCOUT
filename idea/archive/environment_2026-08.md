# 2026-08 服务器环境快照

> 历史记录：从旧 README 保留的安装命令与版本，不是当前状态或通用安装承诺。
> 当前入口与验证范围见 [项目 README](../../README.md) 和 [模型笔记](../idea_notes.md)。

已在共享 GPU 服务器 `106.14.2.243:1022`（Ubuntu 22.04，8× NVIDIA H20，CUDA toolkit 12.6，驱动 550.54）上用 **uv** 验证通过。所有内容（venv、缓存、源码依赖）都装在 `/root/workspace/baojiachun/` 下，便于整体清理，不碰服务器上别的文件。

> **为什么不用 SOE 写死的 Python 3.8 + torch 1.13？** 这台机器是 H20 卡（Hopper, sm_90），torch 1.13 的预编译包没有这块卡的算力核，**装上也用不了 GPU**。必须升到 torch 2.x。实测 **Python 3.10 + torch 2.4.1+cu121** 在 H20 上稳定可用；SCOUT 复用的 LPB 代码（`diffusion_policy/`、`robomimic`）是纯 Python，对版本不挑。

> **网络**：该服务器在国内，官方 `download.pytorch.org` 和 `github.com` 都连不上/不稳，只有国内镜像（清华 TUNA、fb 的 CloudFront）可达。下面全程走镜像。

### 0. 前置（服务器已有）
`uv 0.11.14`、系统 Python 3.10、`/usr/local/cuda`（nvcc 12.6 + gcc 11.4）。

### 1. 建独立 venv（缓存/Python 都收在 baojiachun 里）
```bash
cd /root/workspace/baojiachun
export UV_CACHE_DIR=/root/workspace/baojiachun/.uv-cache
export UV_PYTHON_INSTALL_DIR=/root/workspace/baojiachun/.uv-python
uv venv --python 3.10 .venv
export VIRTUAL_ENV=/root/workspace/baojiachun/.venv   # 之后所有 uv pip 都进这个 venv
MIRROR=https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 装 PyTorch（默认 cu121 版，driver 550 可跑 H20）
```bash
uv pip install --python .venv/bin/python --index-url $MIRROR torch==2.4.1 torchvision==0.19.1
# 验证：.venv/bin/python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# 期望：True NVIDIA H20
```

### 3. robomimic 源码（固定 commit，跟 SOE 一致）
```bash
mkdir -p dependencies && cd dependencies
git clone https://github.com/ARISE-Initiative/robomimic.git
cd robomimic && git checkout 9273f9cce85809b4f49cb02c6b4d4eeb2fe95abb && cd ../..
```
> 服务器连不上 GitHub 时，可在能联网的机器上 clone 好再用 `rsync`/`scp` 传上去（或临时挂代理）；`dependencies/robomimic` 是源码目录，不进 git。

### 4. 核心依赖（版本钉死，理由见下方「坑」）
```bash
uv pip install --python .venv/bin/python --index-url $MIRROR \
  "numpy<2" "zarr<3" "mujoco<3" \
  diffusers==0.27.2 "huggingface-hub==0.24.6" transformers==4.44.2 \
  hydra-core einops scipy scikit-learn opencv-python matplotlib tqdm wandb \
  dill imageio av easydict
```

### 5. robosuite / robomimic 用 `--no-deps`（见「坑」），再补漏的轻量依赖
```bash
uv pip install --python .venv/bin/python --index-url $MIRROR --no-deps robosuite==1.4.1
uv pip install --python .venv/bin/python --index-url $MIRROR --no-deps -e dependencies/robomimic
uv pip install --python .venv/bin/python --index-url $MIRROR \
  termcolor absl-py transforms3d fastjsonschema jsonschema numba
```

### 6. pytorch3d（用 fb 预编译 wheel，免源码编译）
GitHub 源码下不动、源码编译又慢，直接装 fb 发布的预编译 wheel（py3.10 + cu121 + torch2.4，与本项目完全匹配）：
```bash
uv pip install --python .venv/bin/python \
  --find-links https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt240/download.html \
  pytorch3d
```

### 7. 验证（全部 SCOUT/LPB 模块能 import + H20 能算）
```bash
cd /root/workspace/baojiachun/scout
.venv/bin/python -c "
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))
from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import DiffusionUnetHybridImagePolicy
from dyn_model.models.resnet_encoder import ResNetEncoder
from scout.model.scout_vib import ScoutVIB
from scout.guidance.planner import ScoutPlanner
from scout.guidance.policy import ScoutPolicy
import scout.eval.rollout, scout.eval.self_improvement
print('ALL_IMPORTS_OK')
"
```

### 已验证版本（2026-08-10）

| 组件 | 版本 |
|---|---|
| Python / uv | 3.10.12 / 0.11.14 |
| torch / torchvision | 2.4.1+cu121 / 0.19.1 |
| numpy / scipy / scikit-learn | 1.26.4 / 1.15.3 / 1.7.2 |
| diffusers / transformers / huggingface-hub | 0.27.2 / 4.44.2 / 0.24.6 |
| hydra-core / omegaconf / einops | 1.3.5 / 2.3.1 / 0.8.2 |
| zarr / opencv-python / matplotlib | 2.18.3 / 4.11.0.86 / 3.10.9 |
| wandb / tqdm / dill / imageio / av | 0.28.1 / 4.70.0 / 0.4.1 / 2.37.4 / 17.1.0 |
| robomimic（源码 @9273f9c）/ robosuite / mujoco / numba | 0.3.0 / 1.4.1 / 2.3.7 / 0.66.0 |
| pytorch3d / easydict | 0.7.8 / 1.13 |

### 几个坑（踩过，记录在此防再踩）
- **H20 vs torch1.13**：见上，必须 torch 2.x，否则 `cuda.is_available()` 假、或跑起来 `no kernel image`。
- **`numpy<2`**：torch 2.4 本身兼容 numpy2，但 robomimic / pytorch3d / diffusers 在 numpy2 下不稳，钉 1.26.4。
- **`zarr<3`**：LPB 的 `replay_buffer` / `normalizer` 用 zarr 2.x API，3.x 改了。
- **`mujoco<3`**：robosuite 1.4.1 配 mujoco 2.3.x；3.x 有 breaking change。
- **`huggingface-hub==0.24.6`**：diffusers 0.27.2 用了已删除的 `cached_download`，新版 hub 直接 ImportError。
- **`transformers==4.44.2`**：LPB base DP 的 `diffusion_policy/common/language_models.py` 要 transformers；选与 hub 0.24.6 兼容的版本。
- **`robosuite --no-deps`**：robosuite 1.4.1 会拉 `pynput → evdev`，evdev 源码编译要 `Python.h`（得 `apt install python3.10-dev`，属系统改动，共享服务器上不宜做）。SCOUT 不用 pynput（那是真机遥操），故 `--no-deps` 跳过，手动补 `termcolor / numba / transforms3d / absl-py / fastjsonschema`。
- **pytorch3d**：GitHub 源码下不动；`rotation_transformer.py` 只用到 `pytorch3d.transforms`，fb 预编译 wheel 足够，免去 30 分钟源码编译。
- **镜像**：官方 `download.pytorch.org` 与 `github.com` 在此服务器都连不上；PyPI 系走 TUNA，pytorch3d 走 fbaipublicfiles（CloudFront 可达）。

### 日常使用
```bash
cd /root/workspace/baojiachun/scout
source /root/workspace/baojiachun/.venv/bin/activate   # 或直接 .venv/bin/python
# python train.py --config-name=...            # E0：训 LPB base DP
# python -m scout.train_vib                    # E1：训 VIB dynamics
# python -m scout.eval.self_improvement        # E4：multi-round loop
```
> 环境（import + CUDA）已验证；**真正跑训练/评估还缺数据与 ckpt**：需要 robomimic lift image 数据（`image_v141.hdf5`）和一个训好的 base-DP checkpoint，以及 5 个集成点的真实验证（见 `idea/` 计划）。这部分待后续。

---
