# Entropy Cost（方案三 / atypical）——公式推导与论文依据

> 正式实验（2026-08-24 起，`CAN-8-24-entropy-s{233,2333,23333}`）中 SCOUT 臂 explore 阶段使用的引导代价。
> 内部代号 `AtypicalCostPlanner`（`--guide atypical`）。本文所有公式与代码逐项对应，代码锚点见文末。

---

## 1. 设定

冻结的 base 扩散策略 π_θ（Diffusion Policy，DDPM 采样）在失败场景上做 ×10 重试采样。每次重试的每个动作块（action chunk）生成时，在去噪循环中注入 entropy cost 的梯度，使候选动作偏离策略自身的无引导意图，同时冻结的 DP 先验约束整体分布。

符号总表（正文各公式处另有逐符号说明）：

| 符号 | 意义 | 维度/取值 |
|---|---|---|
| o_t | t 时刻观测（2 个相机图像 + 9 维本体感受） | — |
| s̄_t | 冻结视觉前端 E_s(o_t) 的输出 | 1088 = 512×2 + 64 |
| a_t | 一个动作块的原始动作向量（8 步 × 10 维 6d 旋转表示，展平） | 80 |
| z | 技能潜变量 | 16 |
| q_φ(z\|s̄,a) | VIB 编码器（对角高斯） | — |
| x̂₀ | 去噪过程中的干净动作估计（策略归一化空间） | 8×10 |
| x_t | 去噪中间变量（与 x̂₀ 同空间） | 8×10 |
| ᾱ_t | DDPM 累积噪声系数（t 步） | 标量 |
| η | guidance scale | 3.0 |
| κ | KL 上限（nats） | 2.5 |
| B | 同一次 replan 调用并发的环境数 | ≤ n_envs=12 |

---

## 2. 编码器 q_φ

**(1) 视觉编码**

$$\bar{s}_t = E_s(o_t)$$

- $E_s$：冻结的 base-DP ResNet 图像编码器（每个视角 512 维）拼接本体感受嵌入（64 维），参数不更新；
- $o_t$：当前时刻两相机图像与本体感受向量；
- $\bar{s}_t$：观测的潜表示，一个动作块内缓存复用（去噪各步不重算，`policy.py` 注释 "pre-encode s̄_t once"）。

**(2) VIB 编码器**

$$q_\phi(z \mid \bar{s}, a) = \mathcal{N}\!\left(z;\; \mu_\phi(\bar{s},a),\; \mathrm{diag}\,\sigma_\phi^2(\bar{s},a)\right)$$

- $z$：技能潜变量，16 维；
- $\mu_\phi$：编码器输出的高斯均值，$\mathrm{MLP}_\phi([\bar{s};a])$ 输出的前 16 维；
- $\sigma_\phi^2$：对角方差，同一 MLP 输出的后 16 维经 $\exp(\mathrm{logvar})$；
- $\phi$：编码器参数（训练期更新，推理期冻结）；
- $a$：原始动作向量，80 维（8 步 × 10 维）。

**(3) 输入归一化（实现细节，影响数值稳定性）**

$$h = \mathrm{LayerNorm}([\bar{s}; a]), \quad (\mu_\phi, \mathrm{logvar}_\phi) = \mathrm{MLP}_\phi(h)$$

- $h$：1168 维拼接向量的逐样本标准化；
- $\mathrm{LayerNorm}$：输入层归一化，2026-08-17 修复引入——$E_s$ 输出是 ReLU 后特征（非负、均值大），未归一化时首层 ReLU 全死区导致编码器为常数函数、引导梯度恒为 0；
- $\mathrm{logvar}_\phi$：对数方差，$\sigma_\phi^2 = \exp(\mathrm{logvar}_\phi)$。

**(4) 动作空间转换（bridge）**

$$a = \mathrm{bridge}(\hat{x}_0) = \mathrm{unnormalize}_{\text{DP}}(\hat{x}_0)$$

- $\hat{x}_0$：DP 预测的动作块，在 DP 的归一化动作空间中；
- $\mathrm{unnormalize}_{\text{DP}}$：DP 训练时拟合的动作归一化仿射变换的逆；
- $a$：编码器训练时见过的原始动作向量；
- bridge 必须可微：代价对 $x_t$ 的梯度需穿过它反传。

---

## 3. 无引导意图基线

每个动作块的去噪循环开始时（第一步、$x_t$ 尚未被修改），捕获策略自身意图：

**(5) 基线捕获**

$$a^0 = \mathrm{bridge}(\hat{x}_0^{(0)}), \qquad (\mu^0, \mathrm{logvar}^0) = \mathrm{enc}_\phi(\bar{s}, a^0)$$

- $\hat{x}_0^{(0)}$：去噪第一步的干净动作估计（此时采样尚未受引导影响，代表 DP 的无引导意图）；
- $a^0$：该意图对应的原始动作；
- $(\mu^0, \mathrm{logvar}^0)$：基线高斯的均值与对数方差，按批内每行存档，整个动作块内固定（实现为 `select_z` 钩子，`entropy_costs.py:178-186`）。

---

## 4. KL 散度的闭式解

**(6) 一般高斯 KL**

$$\mathrm{KL}\!\left(\mathcal{N}(\mu,\Sigma)\;\|\;\mathcal{N}(\mu_0,\Sigma_0)\right) = \tfrac{1}{2}\left[ \mathrm{tr}(\Sigma_0^{-1}\Sigma) + (\mu-\mu_0)^\top \Sigma_0^{-1} (\mu-\mu_0) - k + \ln\frac{\det\Sigma_0}{\det\Sigma} \right]$$

- $\mu, \Sigma$：第一个高斯（候选动作对应的 $q_\phi$）的均值与协方差；
- $\mu_0, \Sigma_0$：第二个高斯（基线 $q_\phi(\cdot|a^0)$）的均值与协方差；
- $k$：维数（此处 16）；
- $\mathrm{tr}$：矩阵迹；
- $\det$：行列式。

**(7) 对角特化**（$\Sigma=\mathrm{diag}(\sigma^2)$，$\Sigma_0=\mathrm{diag}(\sigma_0^2)$）

$$\mathrm{KL} = \frac{1}{2}\sum_{i=1}^{k}\left[ \frac{(\mu_i-\mu^0_i)^2}{\sigma^{0\,2}_i} + \frac{\sigma^2_i}{\sigma^{0\,2}_i} - 1 - \left(\mathrm{logvar}_i - \mathrm{logvar}^0_i\right) \right]$$

- $\mu_i, \sigma^2_i, \mathrm{logvar}_i$：候选动作下 $q_\phi(z|\bar{s},a)$ 第 $i$ 维的均值、方差、对数方差；
- $\mu^0_i, \sigma^{0\,2}_i, \mathrm{logvar}^0_i$：基线高斯第 $i$ 维的对应量；
- $-\left(\mathrm{logvar}_i - \mathrm{logvar}^0_i\right) = \ln(\sigma^{0\,2}_i/\sigma^2_i)$，即对数行列式比的逐维和；
- $-1$ 项：迹差的常数部分；
- 代码：`entropy_costs.py:200-201`，与本式逐项一致。

---

## 5. Cost 定义

**(8) capped atypical cost**

$$C(\hat{x}_0) = -\min\!\left(\mathrm{KL}\left(q_\phi(z|\bar{s},a)\;\|\;q_\phi(z|\bar{s},a^0)\right),\; \kappa\right), \qquad a = \mathrm{bridge}(\hat{x}_0)$$

- $C$：单个批行的代价，最小化它等价于最大化（封顶的）KL；
- $\kappa=2.5$：KL 上限，单位 nats。当 $\mathrm{KL}\ge\kappa$ 时代价关于输入为常数、梯度为 0，推力饱和——DP 先验之外唯一的信任域机制；
- 负号：注入路径统一做代价下降，见下节。

---

## 6. 去噪循环中的注入

DDPM 反向去噪每步（$t < $ `guidance_start_timestep` = 100，即全部步）执行：

**(9) 干净动作估计**

$$\hat{x}_0(x_t) = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\hat\varepsilon_\theta(x_t,t)}{\sqrt{\bar\alpha_t}}$$

- $x_t$：当前去噪中间变量（`trajectory`）；
- $\hat\varepsilon_\theta$：DP 的噪声预测网络（冻结）；
- $\bar\alpha_t = \prod_{s\le t}\alpha_s$：累积噪声系数（`scheduler.alphas_cumprod[t]`）；
- 实现：`scheduler.step(...).pred_original_sample`（`policy.py:241-243`）。

**(10) 代价对 $x_t$ 的梯度（链式法则）**

$$g(x_t) = \frac{\partial}{\partial x_t} \sum_{b=1}^{B} C\!\left(\hat{x}_0^{(b)}(x_t)\right)$$

- $b$：同一次 replan 调用的并发环境行号；
- $B$：并发行数（≤ n_envs）；
- **sum 归约**：各行代价独立（块对角），求和的梯度给每行完整梯度，注入力不依赖 $B$。此前的 mean 归约使每行梯度被除以 $B$（有效引导力 = η/B），2026-08-21 修复（`policy.py:251-257`，`idea/guidance_batch_scaling_bug.md`）；
- 实现用 `torch.autograd.grad` 对 $x_t$ 求导，梯度自动穿过 (9)→(4)→(8)。

**(11) 注入更新**

$$x_t \leftarrow x_t - \eta\,\sqrt{1-\bar\alpha_t}\; g(x_t)$$

- $\eta = 3.0$：guidance scale（$\eta=1$ 对应规范引导采样；$\eta>1$ 是放大步长，实践中常用，见第 8 节 Ho & Salimans 2022）；
- $\sqrt{1-\bar\alpha_t}$：随去噪进行（$\bar\alpha_t\to 1$）衰减到 0 的缩放因子，与 Dhariwal & Nichol (2021) 的 ε-空间引导项形式一致；
- 负号：对 $C$ 做梯度下降 = 对 KL 做梯度上升（在 $\mathrm{KL}<\kappa$ 区间内）；
- 代码：`cond_grad = -autograd.grad(loss, trajectory)`；`grad_scale = η·(1−ᾱ_t).sqrt()`；`trajectory = trajectory.detach() + grad_scale * cond_grad`（`policy.py:261-265`）。

**(12) 隐式引导后的采样分布**

$$p_{\text{guided}}(x) \;\propto\; p_{\text{DP}}(x)\cdot \exp\!\left(\min\!\left(\mathrm{KL}(x),\,\kappa\right)\right)$$

- $p_{\text{DP}}$：冻结 DP 的动作分布（先验/信任域主体）；
- $\mathrm{KL}(x) = \mathrm{KL}(q_\phi(z|\bar{s},\mathrm{bridge}(x))\,\|\,q_\phi(z|\bar{s},a^0))$；
- 该乘积形式是引导采样的标准结果：向反向过程每步加 $\sqrt{1-\bar\alpha_t}\,\nabla_{\hat{x}_0}\log f(\hat{x}_0)$（此处 $f=\exp(\min(\mathrm{KL},\kappa))$）渐近产生 $p_{\text{DP}}\cdot f$ 的样本（Dhariwal & Nichol 2021；以 $-\log f$ 视作分类器代价即其 classifier guidance 设定）；
- κ 的作用在分布层面：似然权重最多放大 $e^{\kappa}\approx 12.2$ 倍，之后不再增长。

---

## 7. 超参数与定标（探索期结论，供复现）

| 参数 | 值 | 定标依据（entropy_e2e 剂量-反应，39 失败场景 × try5/10） |
|---|---|---|
| η | 3.0 | 救回 0.2→2→8→11→**15**→15（2.0 与 4.0 平台，3.0 取平台起点） |
| κ | 2.5 | κ=2.5 与 10 无显著差；3.0+κ=5 组合变差（jerk 0.68），参数非可加 |
| guidance_start_timestep | 100 | s=3.0 下全窗优于部分窗（15 vs 8）；s=2.0 时相反——交互效应 |
| try 次数 | 10 | pass@10 随 try 增长饱和点 |
| 力度锚定 | dyn NLL 实测梯度中位数 18 | 预定标时代价力度弱 100×+，以量级对齐校准 |

---

## 8. 论文依据映射

| 本文公式/机制 | 出处 | 关系 |
|---|---|---|
| (9) ε-/x̂₀-参数化与反向过程 | Ho, Jain, Abbeel 2020, *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239 | DDPM 采样器本体 |
| (11) $\sqrt{1-\bar\alpha_t}$ 缩放的加性引导项 | Dhariwal, Nichol 2021, *Diffusion Models Beat GANs*, arXiv:2105.05233 | classifier guidance：向去噪步注入 $\log f$ 的梯度以采样 $p\cdot f$；本文 $f=\exp(\min(\mathrm{KL},\kappa))$ |
| (11) 的实现范本 | Sun, Song 2025, *Latent Policy Barrier*, arXiv:2508.05941，`guided_conditional_sample`（其代码 `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:212-271`） | 冻结 DP + 去噪循环代价梯度注入的直接模板；其代价为到 expert 流形的距离，本文换为对自身意图的 KL |
| (2)(3) q_φ 的对角高斯参数化与 VIB 目标 | Alemi, Fischer, Dillon, Murphy 2017, *Deep Variational Information Bottleneck*, arXiv:1612.00410 | 编码器形式与训练期 KL-to-prior 正则 |
| z 作为行为摘要、q(z\|s,a) 条件的使用 | Eysenbach et al. 2019, *DIAYN*, arXiv:1802.06070 | DIAYN 在训练期最大化 $I(z;s)$；本文反向使用同一形式的条件分布——推理期以 KL 距离推离自身意图，不引入目标 z |
| 动作块 → 潜变量的系统上下文 | SOE, arXiv:2509.19292 | SOE 在潜空间加噪探索（action reconstruction VIB）；本文在动作空间经编码器 Jacobian 引导（next-latent VIB），对比点 |
| κ 对散度项设上界 | Schulman et al. 2017, *PPO*, arXiv:1707.06347（clipped surrogate） | 结构先例：对散度型目标设上界以防止单步过度更新 |
| η>1 的放大引导 | Ho, Salimans 2022, *Classifier-Free Diffusion Guidance*, arXiv:2207.12598 | guidance 权重 >1 的外推采样实践先例 |
| DP 本体 | Chi et al. 2023, *Diffusion Policy*, arXiv:2303.04137 | 被引导的冻结基座策略 |

---

## 9. 代码锚点（entropy-dev 分支）

| 内容 | 位置 |
|---|---|
| AtypicalCostPlanner（(5)(8) 全部） | `scout/guidance/entropy_costs.py:163-208` |
| KL 闭式解（(7)） | `scout/guidance/entropy_costs.py:199-201` |
| 基线捕获 select_z（(5)） | `scout/guidance/entropy_costs.py:178-186`（触发点 `scout/guidance/policy.py:244-250`，每块第一步） |
| 动作空间转换 bridge（(4)） | `scout/normalizer.py:57-69`（`NormalizerBridge`）；调用 `scout/guidance/entropy_costs.py:46-52`（`_enc_forward`） |
| 去噪注入循环（(9)-(11)，含 sum 归约与遥测） | `scout/guidance/policy.py:236-285` |
| VIB 编码器（(2)(3)） | `scout/model/vib.py:36-65`（`VIBEncoder`，含 LayerNorm 修复注释） |
| E_s / ScoutVIB | `scout/model/scout_vib.py:37-78` |
| 正式实验驱动（参数 η/κ/窗口的落点） | `soe_scripts/round_entropy.sh` + `configs/eval_can_entropy.yaml`（`exploration.guidance_scale: 3.0`、`guidance_start_timestep: 100`、`ATT_CAP=2.5`） |

**与旧 scout cost（NLL）的区别**：旧代价 $\mathrm{cost} = \lVert z - \mu_\phi(\bar{s},a)\rVert^2_{\sigma^{-2}}$ 需要先验采样一个目标 z 并整段锁定；entropy cost 不引入目标 z，目标纯粹是"偏离自身意图的（封顶）KL"，每块以当前无引导意图为参照，批内各行独立。
