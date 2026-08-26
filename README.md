# SCOUT — Robot Policy Self-Improvement via Classifier-Guided Optimization of Unsupervised Trajectories

> 一个机器人模仿学习的探索方法：在**冻结的**扩散策略（Diffusion Policy）之上，训练一个轻量的 VIB 动力学模型学习 skill 潜空间，并在测试时用 **classifier guidance** 引导冻结策略探索多样的 skill，最终实现 multi-round self-improvement。

SCOUT 本意为「侦察 / 探索」，恰好对应本方法的核心——**探索（exploration）**。

---

## 名字含义

**SCOUT** 是首字母缩写，每个字母都对应方法里的一个真实组件，并非硬凑：

| 字母 | 词 | 在方法里的角色 |
|---|---|---|
| **S** | **S**elf-improving | multi-round self-improvement 闭环：探索 → 收集成功 rollout → 回灌训练 |
| **C** | **C**lassifier-guided | score-based classifier guidance，测试时引导冻结 DP（本方法区别于基线的标志） |
| **O** | **O**ptimization | guidance 在去噪每步加 $\nabla[-Cost]$ 梯度，本质是对动作的优化 |
| **U** | **U**nsupervised | skill 潜变量 $z$ 无标签自学，纯从状态转移 $(S_t, A_t, S_{t+1})$ 学得 |
| **T** | **T**rajectories | 探索产出轨迹，作为 self-improvement 的燃料 |

整句读：**「通过无监督轨迹的分类器引导优化来实现自我改进」**——即方法机制本身。

---

## 核心思想

把「会做什么动作」（policy）和「想达成什么 skill」（goal）解耦：

- **base Diffusion Policy**：预训练、**冻结**。负责生成合法动作，提供去噪 score；训练时不在场。
- **VIB 动力学模型**：一个编码器 + 一个 next-state 解码器，学习一个与 $\mathcal N(0,I)$ 对齐的 skill 潜空间。
- **测试时 classifier guidance（entropy cost）**：在 DP 的去噪循环里注入 Cost 梯度，把候选动作的编码后验推离 **DP 自身无引导意图**的后验——即「做策略自己不会做的事」（cost 定义见下文 §2，完整推导见 [`idea/entropy_cost.md`](idea/entropy_cost.md)）。

冻结的 base DP 自己只会重复同类动作（无探索）；SCOUT 用 entropy cost 给重试动作一个明确、可控的「偏离习惯行为」方向，从而产生**有意义的多样性探索**。

### 与 SOE 的对比

SCOUT 建立在学长 [SOE](https://arxiv.org/abs/2509.19292)（*Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration*）之上，二者的关键区别：

|  | SOE | SCOUT |
|---|---|---|
| 信息瓶颈 | action reconstruction（动作低维、好预测） | next-state prediction（抓住「要把状态带成什么样」） |
| 探索方式 | 潜空间加噪 $z=\mu+\sigma\varepsilon\cdot\alpha$ | classifier guidance（entropy cost，去噪每步加 Cost 梯度） |
| base DP | 训练时与插件共享 | 冻结，训练时不在场 |
| skill 来源 | 观测压缩潜空间 | $(s,a)\to$ skill 语义潜空间 |

---

## Pipeline

### 1. 训练阶段：VIB 动力学模型

**只训练 VIB 模块，base DP 冻结且不参与。**

```
(Sₜ, Aₜ) → Encoder p̄_θ(z | Sₜ, Aₜ) → (μ, σ) → reparam z
(Sₜ, z)  → Dynamics Decoder q_φ(Sₜ₊₁ | Sₜ, z) → Ŝₜ₊₁
```

**信息论目标**：
$$\max\ I(Z;\, S_{t+1} \mid S_t) - \beta\, I(Z;\, A_t \mid S_t)$$

- 第一项逼 $z$ 抓住「要把状态带成什么样」（结果 / 目标）；
- 第二项逼 $z$ 尽量少依赖「具体怎么动」（手段）。

**VIB 变分上界**（落地 loss）：
$$\mathcal{L} = \underbrace{-\mathbb{E}_{z \sim \bar{p}_\theta(z|S_t,A_t)} \log q_\phi(S_{t+1}|S_t,z)}_{\text{① next-state 重建}} + \underbrace{\beta\, KL[\bar{p}_\theta(z|S_t,A_t)\,\|\,r(z)]}_{\text{② KL 正则}}$$

- ① next-state 重建：扩散式去噪 MSE，或确定性回归 $\|S_{t+1}-\hat S_{t+1}\|^2$；
- ② KL 正则：解析的对角高斯 KL 到 $\mathcal N(0,I)$。

前向是一条直链，单次 `loss.backward()` 即可，无需梯度隔离（base DP 本就冻结、不在场）。

### 2. 测试阶段：classifier-guided 探索（entropy cost）

测试时**只用 VIB 编码器 $q_\phi$ 和冻结的 base DP**，动力学解码器下线；**不再从先验采样目标 $z$**——引导的参照系换成 DP 自己的无引导意图动作 $a^0$（每个动作块的第一个 guided 去噪步捕获一次）。

**score 分解**（注入 DP 去噪循环）：
$$\nabla_a \log p(a \mid s) = \nabla_a \log \bar{p}_{DP}(a \mid s) + \nabla_a[-Cost(a \mid s)]$$

- 第一项：base DP 自身的去噪方向；
- 第二项：entropy cost 梯度——把动作往「编码后验偏离 DP 意图后验」的方向推（KL 梯度上升，封顶前）。

$$Cost(a \mid s) = -\min\Big(\mathrm{KL}\big(q_\phi(z \mid \bar s, a)\,\big\|\,q_\phi(z \mid \bar s, a^0)\big),\ \kappa\Big)$$

- $q_\phi(z\mid\bar s,\cdot)=\mathcal N(\mu_\phi,\mathrm{diag}\,\sigma_\phi^2)$：VIB 编码器（16 维 skill 潜空间）；$\bar s = E_s(o)$（冻结视觉前端摘要，块内缓存）；$a=\mathrm{bridge}(\hat x_0)$：回到编码器训练动作空间的 80 维动作块；
- $a^0$：本块 DP 无引导意图动作——KL 度量「候选动作的行为编码离策略习惯行为多远」；
- 对角高斯 KL 有闭式解，均值差按 $1/\sigma^{0\,2}_i$ 马氏加权（意图后验越确定的维度，偏离罚得越重）；
- $\kappa = 2.5$ nats：KL 封顶——与 DP 先验构成两个信任域；引导后采样分布 $\propto p_{DP}\cdot e^{\min(\mathrm{KL},\kappa)}$，似然至多放大 $e^\kappa\approx 12$ 倍。

**设计来源**（DIAYN 一脉）：探索目标 $\max\ I(Z;S')=H(S')-H(S'\mid Z)$ 中 $H(S')$ 不可直接计算（需要未来状态密度）；由确定性解码器的推前引理，**后验不动的动作不可能改变未来分布**，于是以「后验相对意图的移动量」（DIAYN 变分界的同一 $z$ 差分 = 后验间 KL）为可计算代理。逐步推导与论文依据见 [`idea/entropy_cost.md`](idea/entropy_cost.md)。

**去噪循环**（$t = T \to 0$，全程引导）：每步先取 DP 去噪方向与干净估计 $\hat x_0$，算 $Cost$ 对带噪轨迹 $x_t$ 的梯度、乘 $\eta\sqrt{1-\bar\alpha_t}$（$\eta=3.0$，批内 sum 归约——每行梯度不随并发 env 数稀释）叠加，沿调整方向走一步（并加随机噪声）。走完得到的动作**既在 DP 支集上、又偏离策略习惯行为**——这就是受控的探索。

> **历史（v0 cost）**：曾用 $Cost=-\log q_\theta(z\mid s,a)$（高斯 NLL，$z$ 从 $\mathcal N(0,I)$ 采样、每条 rollout 定住）——「给定 skill 目标，逼动作编码回去等于 $z$」。因需要外部 $z$ 目标且 guidance↔训练数据存在正反馈（梯度膨胀，`experiments/e2_scout_guidance_gradient_analysis.md`），2026-08-24 被 entropy cost 取代为正式方法；实现保留在 `scout/guidance/cost.py`（`--guide dyn` / `expert`）。

### 3. Self-improvement 闭环

探索产出的轨迹 → 筛选成功 rollout → 与原始 demo 合并 → 重新训练 → 更强的 base DP → 更高质量的探索 …… 循环多轮，成功率逐轮提升（对标 SOE 的 multi-round self-improvement）。

```
  冻结 base DP  ──classifier guidance──>  探索 rollout
       ▲                                       │
       │                                       ▼
    重训 DP  <──  合并成功 rollout  <──  筛选成功
```

---

## 关键风险

- **$\beta$（KL 权重）是 make-or-break 旋钮**：
  - $\beta$ 太大 → $z$ 与动作脱钩 → 换 $z$ 动作不变，探索失败；
  - $\beta$ 太小 → 退化为普通动力学模型。
  - guidance 要 $\mu(s,a)$ 对 $a$ 敏感，而训练目标的 $-\beta\,I(Z;A_t)$ 项恰好在压制这种依赖——二者构成内在张力。
- **next-state prediction 是头号工程风险**：图像观测下预测下一帧是 world-model 级难题（SOE 靠 action reconstruction 绕开了它）。计划先在低维状态（robomimic `low_dim`）上验证可行性，作为 go/no-go 里程碑。

---

## 项目结构

```
SCOUT/                              # 当前分支:impl/scout-stage1
├── README.md                       # 本文件
├── train.py                        # LPB 训练入口（hydra + OmegaConf）
├── configs/                        # base DP 等 yaml 配置（如 base_dp_lift_image.yaml）
├── diffusion_policy/               # 【LPB 复用】base DP：DiffusionUnetHybridImagePolicy + workspace + env_runner
├── dyn_model/                      # 【LPB 复用】E_s 前端：ResNetEncoder + proprio embed + robomimic image dataset
├── scout/                          # 【SCOUT 自研】
│   ├── model/                      #   scout_vib（VIB dynamics）、encoder（StateEncoder）、vib（VIB enc / D_s）
│   ├── guidance/                   #   policy（ScoutPolicy 注入）、planner、entropy_costs（★现行 entropy cost/方案二三）、cost（旧 NLL）、expert_bank
│   └── eval/                       #   rollout、metrics、self_improvement（multi-round 闭环）
└── idea/                           # 研究构思 + 落地计划 + 组会笔记
    ├── idea.md / idea_notes.md     #   导师原始 idea + 流程梳理
    ├── scout_design.md             #   ★ 权威设计文档（LPB 对齐架构，冲突以此为准）
    ├── entropy_cost.md             #   ★ entropy cost 推导（现行 guidance cost）
    ├── stage1_plan.md / evaluation_plan.md
    └── long_term_plan.md / group_meeting_notes.md
```

> `SOE/`（学长基线代码）、`papers/`（参考论文 PDF）、`CLAUDE.md` / `.claude/`（本地工具配置）已按项目约定从本仓库中排除。

---

## 环境配置（服务器 · uv）

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

## 状态

✅ **Stage-1 已实现（branch `impl/scout-stage1`）。** SCOUT 三件套 —— VIB 动力学模型（`scout/model/`）、classifier guidance 去噪路径（`scout/guidance/`）、multi-round self-improvement 闭环 + eval（`scout/eval/`）—— 均已落地，采用 **LPB 对齐架构**：base DP 复用 LPB 的 `DiffusionUnetHybridImagePolicy`（冻结，其 ResNet 给 `E_s` 复用）；dynamics 是 SCOUT 自研的 VIB（`VIB_enc → z → D_s` + latent MSE + βKL），与 LPB 的确定性 embedding 结构不同，未 fork。每步 mock/合成验证通过。服务器环境（uv，py3.10 / torch 2.4.1+cu121）已配好并验证：SCOUT/LPB 全模块 import + H20 CUDA（见上方「环境配置」）。

⏳ **真实验 deferred。** E0 base DP 训练 / E1 β 扫描 + 生死诊断 / E4 multi-round loop 尚未实跑 —— 缺 robomimic lift **image** 数据、训好的 base-DP checkpoint，以及 5 个集成点（robomimic env adapter、LPB ckpt load、core demo 提取、round warm-start、ScoutPolicy 实例化）的真实验证。落地路线见 `idea/long_term_plan.md`，**权威设计见 `idea/scout_design.md`**（正文若与之冲突，以 `scout_design.md` 为准）。

> 📝 **2026-08 更新**：上述「deferred」为 stage-1 完成时点的快照。此后 base DP / VIB / multi-round 已在服务器实跑（e2–e5 系列实验，见 `experiments/experiment_log.md`）；**正式 entropy-cost 实验**（can，3 seed × DP/SCOUT 双臂，SOE rescue 口径 ×10 重试，`--guide atypical`）自 2026-08-24 起运行（服务器 `data/2026_8_21_entropy/`，驱动 `soe_scripts/round_entropy.sh`）。

---

## 参考

- **SOE** — *Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration*. [arXiv:2509.19292](https://arxiv.org/abs/2509.19292) · [项目页](https://ericjin2002.github.io/SOE/)
- **Classifier Guidance** — Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, 2021.
- **DIAYN** — Eysenbach et al., *Diversity is All You Need*, 2019.
- **DeepVIB** — Alemi et al., *Deep Variational Information Bottleneck*, 2017.
- **Diffusion Policy** — Chi et al., 2023.
