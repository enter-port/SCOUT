# SCOUT 模型笔记（与代码实现对齐版）

## 1. 输入 / 输出

- 训练：输入转移三元组 $(S_t,\ A_{t:t+fs},\ S_{t+fs})$（fs=8，robomimic image 演示数据）。
  一切先过状态编码器进入 latent 空间：产出 $\hat{\bar s}_{t+1}$，与真实
  $\bar s_{t+1}=E_s(S_{t+fs})$ 比较（**latent 级**比较，不重建像素/状态）。
- 测试：输入观测 $s$ + $z\sim\mathcal N(0,I)$（每条 rollout 采样一个 z，
  **整段定住、跨 chunk 不变**）；输出动作 $a$（被引导的 base DP 生成）。
  测试时用到 $E_s$ + VIB 编码器 + base DP；**动力学解码器不用**。

## 2. 涉及的网络（共 4 个）

> 维度口径：B=batch、T=时间步（VIB 训练中 T=1）。can/lift/square 为 2 视角
> （agentview + eye_in_hand），transport 为 4 视角；图像原生 84×84。

### ① 状态编码器 E_s：S_t → s̄_t（图像与 proprio 永远同时进）

视觉分支（每视角独立、**冻结**，2 视角结构相同）：
```
(B,T,3,84,84) →[训练时先 RandomCrop(76)]→ (B,T,3,76,76)
  →[ResNet-18：conv7×7/s2 → maxpool → 4 组 [2,2,2,2] 残差块；去 avgpool/fc]→
(B,T,512,3,3) →[AdaptiveAvgPool2d((1,1))]→ (B,T,512)        # 每视角
```
proprio 分支（**可训练**）：
```
(B,T,9) →[permute]→ (B,9,T) →[Conv1d(9→64, k=1)]→ (B,64,T) →[permute]→ (B,T,64)
```
融合：
```
2 视角 concat (B,T,1024) ⊕ proprio (B,T,64) →→ s̄_t = (B,T,1088)   # T=1 时 squeeze 成 (B,1088)
```
- ResNet-18 = base DP ckpt 里的 robomimic `ResNet18Conv`（torchvision resnet18 去
  avgpool/fc，pretrained=False）；每视角 ~11.18M 参数，`requires_grad=False` + 永远 `eval()`。
- proprio Conv1d = 单帧线性嵌入（tubelet=1），640 参数。
- 注意：E_s 的读出 = 512 维 avgpool，≠ base DP obs encoder 的 64 维 SpatialSoftmax ——
  同 backbone、不同读出；权重从 base DP ckpt 抠出共享。

### ② VIB 编码器 VIB_enc：(s̄_t, a_chunk) → (μ, logvar) → z（**可训练**）

```
s̄_t (B,1088) ⊕ a_chunk (B,80)          # a_chunk = 8 步 × 10 维展平（frameskip=8）
  →[concat]→ (B,1168)
  →[Linear(1168→128) + ReLU]→ (B,128)
  →[Linear(128→128)  + ReLU]→ (B,128)
  →[Linear(128→32)]→ (B,32) →[chunk 两半]→ μ (B,16), logvar (B,16)
  z = μ + exp(½·logvar) ⊙ ε,  ε~N(0,I) →→ z (B,16)
```
- EncoderMLP，layer_num=1，无 norm / dropout / residual，orthogonal 初始化。
  参数量 **170,272**。

### ③ 动力学解码器 D_s：(z, s̄_t) → ŝ̄_{t+1}（**可训练**）

```
z (B,16) ⊕ s̄_t (B,1088)
  →[concat]→ (B,1104)
  →[Linear(1104→128) + ReLU]→ (B,128)
  →[Linear(128→128)  + ReLU]→ (B,128)
  →[Linear(128→1088)]→ ŝ̄_{t+1} (B,1088)
```
- 与 target s̄_{t+1} = E_s(S_{t+fs}).detach() 求 MSE；预测的是**下一个 latent**
  （E_s 空间），不是 next state / 像素。参数量 **298,304**。
  可训练总量（vib_enc + D_s + proprio）≈ 469k。

### ④ base DP（DiffusionUnetHybridImagePolicy，预训练、**冻结**）

（a）观测编码（robomimic bc_rnn 默认，BN→GroupNorm；obs 走 global cond）：
```
每视角: (B·To,3,84,84) →[CropRandomizer 76（训练随机 / eval 固定中心）]→ (B·To,3,76,76)
        →[ResNet18Conv（pretrained=False）]→ (B·To,512,3,3)
        →[Conv1×1: 512→32 → 逐 keypoint 空间 softmax → (x,y) 坐标]→ (B·To,64)
2 视角 concat → (B·To,128) ⊕ low_dim 9 维（直通，不编码） → obs_feature (B·To,137)
堆 To=2 帧 → global_cond (B,274)       # transport: 4×64+18=274 → global 548
```
（b）扩散 UNet 去噪（ε 预测）：
```
时间步 t →[SinusoidalPosEmb(128)→Linear 512→Mish→Linear 128]→ (B,128)
条件 cond = step-emb (B,128) ⊕ global_cond (B,274) = (B,402)
        →→ FiLM（每通道 scale+bias）注入每个 block

带噪轨迹 (B,16,10):
  down:  →[block 10→512]→ →[block 512→512]→ →[Down k3s2]→ (B,512,8)
         →[block 512→1024]→ →[block 1024→1024]→ →[Down k3s2]→ (B,1024,4)
         →[block 1024→2048]→ →[block 2048→2048]→ (B,2048,4)
  mid:   →[block 2048→2048]→ →[block 2048→2048]→ (B,2048,4)
  up:    ⊕skip →[block 4096→1024]→ →[block 1024→1024]→ →[Up ×2]→ (B,1024,8)
         ⊕skip →[block 2048→512]→ →[block 512→512]→ →[Up ×2]→ (B,512,16)
         ⊕skip →[block 1024→512]→ →[block 512→512]→ (B,512,16)
  final: →[Conv1d(512→512,k5)+GroupNorm(8)+Mish]→ →[Conv1d(512→10,1)]→ ε̂ (B,16,10)

  block = 2×(Conv1d(k5) → GroupNorm(8) → Mish)；Down = Conv1d k3 s2；Up = ConvTranspose1d k4 s2
```
（c）训练 / 采样：
```
训练: 干净动作 chunk 加噪到随机 t → UNet 预测 ε → MSE(ε̂, ε)
采样: 100 步 DDPM（guidance 只注入最后 50 步）→ x̂₀ (B,16,10)
      →[unnormalize：10 维 6d → 7 维 axis-angle]→ 取 [1:1+8] 执行 8 步
```
- 参数量：UNet can/lift/square **255.6M**、transport **263.5M**；视觉 2 视角 ~22.4M（4 视角 ~44.8M）。
- 调度器：DDPM 100 步，β 1e-4→0.02 squaredcos_cap_v2，ε 预测，clip_sample；
  horizon 16 / n_obs_steps 2 / n_action_steps 8。

## 3. 训练目标与 loss

- 信息论目标（经 E_s 后落到 latent 动力学上）：
  $$\max\ I(Z;\,\bar s_{t+1}\mid\bar s_t)\ -\ \beta\,I(Z;\,A_t\mid\bar s_t)$$
  - 第一项：z 能预测下一个 latent（给定 $\bar s_t$）→ 抓住"要把状态带成什么样"
  - 第二项：z 尽量少依赖 action → z 是"目标/结果"，非"具体怎么动"
- VIB 变分上界：
  $$\mathcal L=\underbrace{-\mathbb E_{z\sim\bar p_\theta(z\mid\bar s_t,a_t)}\log q_\phi(\bar s_{t+1}\mid\bar s_t,z)}_{\text{① 下一 latent 重建}}+\underbrace{\beta\,KL[\bar p_\theta(z\mid\bar s_t,a_t)\,\|\,r(z)]}_{\text{② KL 正则}}$$
- 第①项：两种候选（扩散去噪 / 确定性回归）中，**实现选了确定性回归**：
  $$(\bar s_t,a_t)\xrightarrow{\text{VIB enc}}(\mu,\sigma)\xrightarrow{\text{reparam}}z,\quad
  (\bar s_t,z)\xrightarrow{D_s}\hat{\bar s}_{t+1},\quad
  \text{loss}=\big\|\bar s_{t+1}-\hat{\bar s}_{t+1}\big\|^2$$
  其中 target $\bar s_{t+1}=E_s(S_{t+fs})$ 被 `.detach()`。不做像素/状态重建 =
  规避 world-model 级难度（设计上的 #1 风险规避）。
- 第②项（KL，解析）：$\beta\,KL=-\tfrac{\beta}{2}\sum_i\big(1+\log\sigma_i^2-\mu_i^2-\sigma_i^2\big)$
  （= 代码里的 $\tfrac{\beta}{2}\sum_i(\mu_i^2+\sigma_i^2-1-\log\sigma_i^2)$，同一式）。
  β 把 z 压向 $\mathcal N(0,I)$ 先验 —— 训练后 kl→0、μ→0 是**预期结果而非 bug**，
  这样测试时 z~N(0,I) 盲采才有意义。
- **内在张力**：KL 压制 z 对 a 的依赖（∂μ/∂a→0），而测试 guidance 需要 cost 对 a
  有梯度；靠 σ 通道化解（见 §5 双通道）。

## 4. 训练前向链路与梯度回传

- 前向一条链：
  $$S_t\xrightarrow{E_s}\bar s_t;\quad(\bar s_t,a_t)\xrightarrow{\text{VIB enc}}(\mu,\sigma)\xrightarrow{\text{reparam}}z\xrightarrow{D_s}\hat{\bar s}_{t+1};\quad S_{t+fs}\xrightarrow{E_s}\bar s_{t+1}(\text{target},\ .detach())$$
- 一次 `loss.backward()`，直链、无额外隔离：
  - 第①项：穿 D_s（更新 $\phi$）→ 穿 z/reparam → 回 VIB 编码器（更新 $\theta$）
  - 第②项：只更新 $\theta$
  - **可训练参数 = {proprio 嵌入, VIB 编码器, D_s}**；base-DP ResNet
    `requires_grad=False` 且固定 eval（BN 不动），梯度为 None。

## 5. 测试阶段：classifier-guided 探索

- 用 E_s + VIB 编码器 + 冻结 base DP；动力学解码器测试时不用。
- z：每条 rollout 采样一次 $z\sim\mathcal N(0,I)$，整段定住（SOE 是每 chunk 重采，
  我们是 per-trajectory）；$\bar s_t$ 在每个 chunk 内定住（整段去噪循环缓存一次 E_s 前向）。
- **Cost（核心）**：$\text{Cost}(a,z\mid s)=\big\|z-z_\theta(\bar s_t,a)\big\|^2$，
  其中 **$z_\theta=\text{reparam}(\mu,\sigma)=\mu+\sigma\varepsilon$** 是 VIB 编码器对
  $(\bar s_t,a)$ 的**采样输出** $p_\theta(\bar s_t,a)$
  这里的 a 是 DP 干净估计 $\hat a_0$ 的前 8 步展平（80 维，与训练时 encoder 输入对齐）。
- score 分解（概念框架）：
  $$\nabla_a\log p(a\mid s)=\nabla_a\log\bar p_{DP}(a\mid s)+\nabla_a[-\text{Cost}(a,z\mid s)]$$
- 去噪循环（t=100→0，**仅 t<50 的最后 50 步**注入 guidance）：
  1. DP 看（带噪 $a_t$、步 t、obs）→ 去噪方向 $\varepsilon_\theta$
  2. 由 $\varepsilon_\theta$ 推一步干净估计 $\hat a_0$（`pred_original_sample`）
  3. 算 Cost($\hat a_0$)，对带噪轨迹取梯度 $\nabla_{a_t}\text{Cost}$，
     乘缩放加到**带噪轨迹**上：
     $$a_t\leftarrow a_t-\eta\sqrt{1-\bar\alpha_t}\;\nabla_{a_t}\text{Cost}$$
     （注意：加在带噪样本上，**不是**加在去噪方向上）
  4. 用原 $\varepsilon_\theta$ 沿调整后的 $a_t$ 走一步 DDPM 反步 → $a_{t-1}$
  5. 走完得：既在 DP 支集上、又朝着"重新编码回去正好等于 z"走的动作块

## 6. 采样 z 的意义

- base DP 自跑只给默认动作（无探索）；采 z = 在"合法 skill 空间"挑目标，逼 DP 生成奔向它的动作 → 探索多样性
- 从 $\mathcal N(0,I)$ 采：训练 KL 已把真实 skill 压到标准正态附近 → 采到的是"合法 skill"
- 每个 rollout 一个 z（跨 chunk 定住）：让同一条轨迹有连贯的"目标"，而非每步换目标

## 7. 代码位置与训练超参

- E_s：`scout/model/encoder.py`；VIB：`scout/model/vib.py` + `scout/model/scout_vib.py`
- 训练：`scout/train_vib.py`（config `configs/vib_{task}_image.yaml`）
- Cost：`scout/guidance/cost.py`；planner：`scout/guidance/planner.py`
- 去噪循环：`scout/guidance/policy.py`（`guided_conditional_sample`）
- base DP：`diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py`（config `configs/base_dp_{task}_image.yaml`）

| | base DP（E0） | VIB dynamics（Step 1） |
|---|---|---|
| batch / lr | 64 / 1e-4 AdamW(0.95,0.999) eps 1e-8 wd 1e-6 | 256 / 1e-3 AdamW(0.9,0.999) wd 1e-6 |
| 调度 | cosine warmup 500，EMA(0.75) | — |
| 轮数 | 600 epoch，rollout/ckpt 每 20 | 300 epoch × 200 step，val 10% demos |
| 其他 | seed 42 | β=1e-3，seed 233，frameskip 8 |

