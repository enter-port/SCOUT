# SCOUT — Robot Policy Self-Improvement via Classifier-Guided Exploration

**S**elf-improving via **C**lassifier-guided **O**ptimization of **U**nsupervised **T**rajectories

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
- **测试时 classifier guidance**：从 $\mathcal N(0,I)$ 采样一个 skill 目标 $z$，在 DP 的去噪循环里注入 Cost 梯度，把动作推向「能命中该 skill」的方向。

冻结的 base DP 自己只会输出默认动作（无探索）；SCOUT 通过给定 skill 目标，逼它生成奔向不同 skill 的动作，从而产生**有意义的多样性探索**。

### 与 SOE 的对比

SCOUT 建立在学长 [SOE](https://arxiv.org/abs/2509.19292)（*Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration*）之上，二者的关键区别：

|  | SOE | SCOUT |
|---|---|---|
| 信息瓶颈 | action reconstruction（动作低维、好预测） | next-state prediction（抓住「要把状态带成什么样」） |
| 探索方式 | 潜空间加噪 $z=\mu+\sigma\varepsilon\cdot\alpha$ | classifier guidance（去噪每步加 Cost 梯度） |
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

### 2. 测试阶段：classifier-guided 探索

测试时**只用编码器的 $\mu$ 和冻结的 base DP**，动力学解码器下线。

```
z ~ N(0, I)        # 从先验采样一个 skill 目标，整段生成定住
```

**score 分解**（注入 DP 去噪循环）：
$$\nabla_a \log p(a \mid s) = \nabla_a \log \bar{p}_{DP}(a \mid s) + \nabla_a[-Cost(a, z \mid s)]$$

- 第一项：base DP 自身的去噪方向；
- 第二项：Cost 梯度，把动作往「编码回去正好等于 $z$」的方向推。

$$Cost(a, z \mid s) = \big\|\, z - \mu(s, a)\,\big\|_2$$

**去噪循环**（$t = T \to 0$）：每步先取 DP 去噪方向，再算 Cost 对 $a$ 的梯度、乘缩放叠加，沿调整方向走一步（并加随机噪声）。走完得到的动作**既在 DP 支集上、又朝着目标 skill $z$ 走**——这就是受控的探索。

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
SCOUT/
├── README.md                  # 本文件
├── .gitignore
└── idea/
    ├── idea.md                # 导师原始 idea（训练 + 测试阶段的数学）
    ├── idea_notes.md          # 流程梳理：网络 / loss / 前向链路 / 测试 guidance
    ├── idea.JPEG              # 导师原始 idea 手稿图
    ├── long_term_plan.md      # 五阶段落地计划
    └── group_meeting_notes.md # 组会汇报提纲 + 关键开放问题
```

> `SOE/`（学长基线代码）、`papers/`（参考论文 PDF）、`CLAUDE.md` / `.claude/`（本地工具配置）已按项目约定从本仓库中排除。

---

## 状态

🚧 **早期研究阶段（idea + 规划）**。当前仓库仅含研究构思与落地计划；核心代码（VIB 动力学模型 policy 类、classifier guidance 去噪路径、multi-round 编排）尚待实现，将以冻结的 SOE base DP 为起点。落地路线见 `idea/long_term_plan.md`。

---

## 参考

- **SOE** — *Sample-Efficient Robot Policy Self-Improvement via On-Manifold Exploration*. [arXiv:2509.19292](https://arxiv.org/abs/2509.19292) · [项目页](https://ericjin2002.github.io/SOE/)
- **Classifier Guidance** — Dhariwal & Nichol, *Diffusion Models Beat GANs on Image Synthesis*, 2021.
- **DIAYN** — Eysenbach et al., *Diversity is All You Need*, 2019.
- **DeepVIB** — Alemi et al., *Deep Variational Information Bottleneck*, 2017.
- **Diffusion Policy** — Chi et al., 2023.
