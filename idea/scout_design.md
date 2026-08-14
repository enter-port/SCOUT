# SCOUT 设计文档(Design Spec)

> **状态**:设计已与用户逐节确认;**任何代码落实仍需用户审核**。
> **相关**:[`idea.md`](idea.md)(导师原始 idea)、[`stage1_plan.md`](stage1_plan.md)(实验计划)、[`evaluation_plan.md`](evaluation_plan.md)(评估口径)、memory `lpb-reference.md`(LPB 参考)。
> **优先级**:与 `stage1_plan.md` 冲突处以本文档为准。
> **架构基线(2026-08-08 修订)**:`E_s` = **LPB 式双输入**(image + proprio **永远同时**进:冻结 base-DP ResNet + 训练的 proprio embed)。**没有 low_dim/image 两种模式、没有 stage1/stage2 之分**——一条管线,永远 image+proprio。数据用 robomimic **image** 数据集。

---

## 0. SCOUT 是什么

导师 classifier-guided exploration idea 的落地,对标学长 SOE。三件套:

1. **冻结 base Diffusion Policy(image DP)**——其 **ResNet 编码器冻结复用**给 `E_s`(LPB 式);base DP 自身不更新。
2. **潜空间 VIB 动力学模型**——`E_s`(双输入:image + proprio)→ `s̄_t`;VIB 学 skill 潜空间 `z`;目标 $\max\ I(Z;S_{t+1}\mid S_t) - \beta\,I(Z;A_t\mid S_t)$。
3. **测试期 classifier guidance**——采样 `z`,在 base DP 去噪循环把动作推向"编码回去 $\approx z$",产生有意义的多样性探索,驱动 self-improvement。

**与 idea 的已知偏离(LP B 式所致)**:idea 写"base DP 训练时不在场";但 LPB 式 `E_s` **复用 base DP 的冻结 ResNet** → VIB 训练时 base DP 的**编码器在线(冻结、不更新)**。这是为图像输入(必须有编码器),LPB 已验证可行。base DP 的其余部分(动作解码器等)仍不在场。

**代码来源 + 复用边界(关键,防混淆)**:
- **LPB 可复用 —— 全是「非动力学」件**:base DP(`DiffusionUnetHybridImagePolicy`)、数据(`RobomimicImageDynamicsModelDataset`)、**E_s 前端编码器**(`ResNetEncoder` + `ProprioceptiveEmbedding`,只做 `obs → s̄_t`)、guidance 注入(`guided_conditional_sample`)。
- **SCOUT 自研 —— = dynamics 本身,LPB 没有,绝不能 fork**:`VIB_enc → z(变分 skill)→ D_s` + latent/KL loss。**LPB 的 `z` 是确定性 embedding(无 μ,logvar/KL);SCOUT 的 `z` 是采样的变分 skill —— 结构根本不同**。dynamics 必须 SCOUT 自己写(已在 `scout/model/`)。
- SCOUT cost(`‖z−z_θ‖`,z_θ=reparam 采样=p_θ(s̄_t,a);**注意非均值 μ**——见 §4)+ self-improvement loop 同为自研。
- 实现:方案 B —— 在当前 `scout/` 上把 SOE 件逐个换成 LPB 的**非动力学件**;dynamics 保持 SCOUT 自研。

---

## 1. 总览(管线)

潜空间动力学(world-model 味),VIB 是中间核心:

```
训练(VIB enc / D_s / proprio embed 一起训;base DP 的 ResNet 冻结在线、不更新):
  {image_t, proprio_t} ─[E_s]→ s̄_t ─┐
                                      ├─[VIB enc]→ (μ,logvar) →reparam z ─┐
                            a_t ──────┘                                     │
                                                                            ├─[D_s]→ ŝ̄_{t+1}   (到此为止,无 decode)
                                                           s̄_t ────────────┘
  E_s = 冻结 base-DP ResNet(image, per-view)+ 训练 proprio embed → concat → s̄_t   (永远两个同时进)
  loss = latent MSE( ŝ̄_{t+1}, E_s(S_{t+1}).detach() ) + β·KL    (latent 级监督 = LPB;无 state decoder)

测试(D_s 下线;只用 z_θ + 冻结 base DP):
  z ~ N(0,I)  整段 chunk 定住
  → 在 base DP 去噪循环注入 ∇_{x_t}[ −‖z − z_θ(s̄_t, a)‖ ]   (z_θ=reparam 采样=p_θ(s̄_t,a);LPB 范式,详见 §4)
```

**永远 image + proprio 同时输入**(LPB 式),无 low_dim/image 模式之分、无 stage 分。**无 state decoder、不解码**;D_s 预测 next-latent(下一帧 ResNet 特征 + proprio,特征空间非像素)→ #1 风险(像素预测)规避。

---

## 2. 网络与维度

`EncoderMLP`(SOE `src/policy/vqvae_modules/vqvae.py:12`):`Linear→ReLU→[Linear→ReLU]×layer_num→Linear(fc)`,`hidden_dim=128, layer_num=1`,无 norm/dropout。

| 网络 | 结构 | 维度 |
|---|---|---|
| **E_s**(LPB 式双输入,**无 AE**) | image:**冻结 base-DP ResNet**(per-view,`AdaptiveAvgPool2d`→512/view);proprio:训练 embed(`ProprioceptiveEmbedding`:Conv1d / 或 MLP);**concat** → `s̄_t` | `s̄_t = 512·n_views + proprio_emb_dim`(lift image 2 视图 → ~1024+) |
| VIB encoder | `concat(s̄_t, a_t) → EncoderMLP → (μ,logvar)` | in = `s_bar_dim + action_dim`;out = `2·style_dim = 32` |
| **D_s**(dynamics decoder) | `concat(z, s̄_t) → EncoderMLP → ŝ̄_{t+1}`(**到此为止,无 decode**) | in = `style_dim + s_bar_dim`;out = `s_bar_dim` |
| skill latent `z` | reparam:`z = μ + σ·ε` | `style_dim = 16` |
| **base DP** | **LPB `DiffusionUnetHybridImagePolicy`**(Chi et al. fork:hybrid image local_cond + low-dim global_cond;ResNet 在 `obs_encoder.obs_nets[view].backbone`)。先训好;**ResNet 冻结复用给 E_s** | 动作 chunk `(B, horizon, action_dim)`;冻结 |

> 维度从 hdf5 读:`action_dim`、`proprio` keys(`robot0_eef_pos/eef_quat/gripper_qpos`)、`n_views`。action 必须 base DP 与 VIB encoder 一致。`env_state`/`states` 不再需要(latent 级监督)。

**文件映射**(搬入 / 参照):
- base DP:**LPB `diffusion_policy/` 整套**(`DiffusionUnetHybridImagePolicy`、workspace、`train.py`、`MultiImageObsEncoder`、`ConditionalUnet1D` …)——替换当前 `scout/policy/`(SOE DP)。
- **E_s**:LPB `dyn_model/models/resnet_encoder.py`(`ResNetEncoder`:从 base DP ckpt 抠冻结 ResNet 主干)+ `dyn_model/models/proprio.py`(`ProprioceptiveEmbedding` Conv1d)。
- guidance:LPB `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:212-271`(`guided_conditional_sample`)。

---

## 3. 训练(联合 + 数据解耦)

**联合训练**(VIB enc / D_s / proprio embed 一起训;base DP 的 ResNet **冻结在线、不更新**):
- 前向(一个 transition batch):$s̄_t = E_s(\{image_t, proprio_t\})$ → $(μ,\logvar)=\text{VIB\_enc}(s̄_t, A_t)$ → $z=\text{reparam}$ → $\hat{s̄}_{t+1}=D_s(z, s̄_t)$。(无 state decoder,到此为止)
- loss(一次 backward 更新 VIB enc / D_s / proprio embed;**latent 级监督 = LPB**):
$$\mathcal L = \underbrace{\|\hat{s̄}_{t+1} - E_s(S_{t+1})\!.detach()\,\|^2}_{\text{latent MSE(target = 真实下一观测再编码)}} + \underbrace{\beta\,\mathrm{KL}[\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,I)]}_{\text{KL}}$$
- **无 AE、无 state decoder、无重建**;target = $E_s(S_{t+1})$(冻结 ResNet 给的稳定视觉锚 + proprio embed)。防坍:冻结 ResNet 是稳定锚;proprio embed / D_s 靠 latent 预测本身携信息。
- base DP 的 **ResNet 冻结在线**(LPB 式,偏离 idea"不在场",见 §0);单链、单次 backward、**无梯度隔离**(冻结参数 `requires_grad=False` 自然不更新)。
- **β = make-or-break 旋钮**:E1 扫 $\beta \in \{10^{-4},10^{-3},10^{-2},10^{-1}\}$ + 生死诊断(§5)定。

**数据(LPB 式 Dataset,无 ReplayBuffer)**:
- 数据 = LPB `RobomimicImageDynamicsModelDataset`(zarr 缓存 + `DataLoader` + `LinearNormalizer`);`__getitem__` 出 `(obs, act, state)` 窗口,`obs` 含 `visual`(多视图图像)+ `proprio`。
- target = $E_s(S_{t+1})$(下一观测再编码),**不需 env_state**。
- 数据集 = robomimic **image**(`image_v141.hdf5`:`obs/<images>` + `obs/<proprio>` + `actions`)。
- **不再用** `TransitionSource`/`ReplayBuffer`/`RunningStats`(LPB 没有这套)。
- **self-improvement 回灌**走"写增强 hdf5(原 demo + 成功 rollout)→ 重载训练"(SOE `run_full_multi_round` 式),见 §5;**不做** in-memory online buffer。

---

## 4. 测试期 guidance(标准 classifier-guided denoising,照搬 LPB)

注入 LPB `guided_conditional_sample` 的确切机制——**改采样 trajectory,不改 ε**:

```python
for t in scheduler.timesteps:
    trajectory[condition_mask] = condition_data[condition_mask]   # inpaint conditioning
    trajectory = trajectory.detach().requires_grad_()             # 对 x_t 开梯度
    model_output = model(trajectory, t, local_cond, global_cond)  # ε_θ(x_t, t)

    if classifier_guidance and t < guidance_start_timestep:      # (a) 只在最后 K 步
        x0_hat = scheduler.step(model_output, t, trajectory).pred_original_sample
        loss   = cost(x0_hat, current_obs)                        # cost 算在 x̂_0 上
        cond   = -torch.autograd.grad(loss, trajectory)[0]         # −∇_{x_t} cost
        scale  = guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()  # η·√(1−ᾱ_t)
        trajectory = trajectory.detach() + scale * cond           # 直接改 x_t
    trajectory = scheduler.step(model_output, t, trajectory, ...).prev_sample  # DDPM 反向步
```

- 梯度对 `trajectory`($x_t$)求,cost 算在 $\hat{x}_0$(一步估计的干净动作);`autograd.grad` 从 $\hat{x}_0$ 流回 $x_t$。
- 改 $x_t$、不改 ε;然后正常 `scheduler.step`。
- 缩放 $\eta\sqrt{1-\bar\alpha_t}$(Dhariwal & Nichol 标准式)。

**SCOUT 的 cost 函数**(替换 LPB 的 NN 距离):
$$\text{cost}(\hat{x}_0,\, s) \;=\; \text{mean}_t\,\big\|\,z - z_\theta(s̄_t,\, a_t)\,\big\|_2,\quad a = \hat{x}_0,\ \ s̄_t = E_s(\{image, proprio\})\ \text{(定住)},\ \ z\ \text{整段定住}$$
$z_\theta = \text{reparam}(\text{VIB\_enc}(s̄_t,a_t))$ = p_θ(s̄_t,a_t) 即 **reparam 采样的 skill**(逐 chunk 步)。**非均值 μ**——见 §4 注(μ-only 会让 KL 压制 ∂μ/∂a 使 guidance 失效)。

**门控**:
- (a) `t < guidance_start_timestep`(最后 K 步引导)→ **保留**(标准 CG 做法)。
- (b) LPB 的 OOD 门 `current_cost > threshold` → **去掉**(SCOUT 是探索,每个 chunk 都主动引导;且无 expert-latent NN 距离)。

**归一化桥**(实现细节):cost 里要把 base DP(SOE 归一化)的动作 **unnormalize → 再 normalize 进 VIB 空间**(参照 LPB `dyn_model/planner.py:211-213`)。

---

## 5. 评估(= SCOUT self-improvement 闭环,metric 参照 SOE)

5 步 multi-round loop(探索用 §4 guidance,回灌用"写增强 hdf5 再重载"):

```
Round 0:  DP₀ ← 训自部分 robomimic 数据(core demos,如 core_20)            [step 1]
          ↓ 训 VIB dynamics + z(§3,base DP 的 ResNet 冻结在线)            [step 2]
          ↓ 冻结 DP₀,采样 z 引导生成 exploration rollouts(§4,robomimic sim) [step 3]
          ↓ 筛成功 rollout → 写增强 hdf5(原 demo + 成功 rollout)→ 重载训 DP₁  [step 4]
Round 1:  DP₁ vs DP₀ 性能对比;多轮滚,success rate / round 应单调升          [step 5]
```

**metric(参照 SOE,定义见 `evaluation_plan.md §一.4`)**:

| 指标 | 含义 | 论证 |
|---|---|---|
| success rate / round | 每轮 N 个初始态成功率 | 自我改进有效(头条:新旧 DP 差异) |
| Pass@5 | N 初始态、5 次探索内可解比例 | 探索覆盖广(SCOUT 最该赢) |
| exploration yield | 每轮成功 exploration rollout 数 | 探索高产 |
| jerk | 动作三阶差分范数 | on-manifold / 平滑 |

**默认参数(沿用 SOE,可调)**:core demos = core_20;初始态 N = 100;探索 try_times = 5;轮数 6;成功判定——sim 用 `env.is_success()["task"]`(成功即止),真机人标 j / k / d。

**前置 action 级闸门(建议,跑 loop 前过)**:生死诊断 $\|\partial\mu/\partial a\|$(敏感比 = $\|\partial\mu/\partial a\|\cdot\sigma_a / \sigma_\mu$,阈值 ~0.3)、guidance 三判据(多样性 / 一致性 / Cost 方向)、on-manifold(jerk / Mahalanobis)。

**实现依赖**:step 3 / 5 需 robomomxic sim rollout + 判成功 → 用 LPB / SOE 的 robomimic rollout 脚手架。在 robomimic **lift image** 上跑。回灌参照 SOE `run_full_multi_round.py`(写增强 hdf5 + `scout_aug` mask),**不用** in-memory buffer。

---

## 6. self-improvement 接口(无 low_dim/image stage 之分)

- **管线永远 image + proprio 同时输入**(LPB 式),不存在"low_dim stage 1 / image stage 2"的划分——这是 2026-08-08 修订的关键点。图像从第一步就在里面。
- **self-improvement 接口**:已含在 §3 `ReplayBuffer` + §5 loop,无新组件。online 训练 = teleop / rollout 产出 transition → `buffer.add()` → 训练从增长 buffer 采样 + running 归一化增量更新。

---

## 7. 关键风险

1. **β 是 make-or-break**:太大 → $\mu$ 与动作脱钩 → guidance no-op → 探索死。靠 §5 生死诊断 + β 扫描把控。
2. **像素预测的 #1 风险已规避**:无 state decoder、不解码;D_s 预测 next-**latent**(下一帧 ResNet 特征 + proprio,特征空间),不是像素。残余:next-image-**特征**预测(LP B 式,已证可行)。
3. **坍缩**:**已无 AE、无 state decoder**。冻结 ResNet 是稳定锚;proprio embed / D_s 靠 latent 预测携信息。残余:`z` 被动力学忽略 → 由 β 把控(见 #1)。
4. **归一化桥错位**:DP 动作空间与 VIB 动作空间不一致会让 cost 失真(§4 桥)。
5. **base DP 编码器在线**(LPB 式,偏离 idea"不在场"):只冻结复用、不更新;与 LPB 一致、已验证可行。

---

## 8. 待定(实现期再决)

- `proprio embed` 结构(Conv1d vs MLP)、`n_views`、冻结 ResNet 接入细节(从 base DP ckpt 抠主干)。
- 超参(`style_dim=16`、`hidden=128`、`D_s` 维度、`guidance_scale`、`guidance_start_timestep`)。
- online buffer 的图像存储(索引/路径 vs 内联)。
- 真机后端数据格式(teleop → transition)。
