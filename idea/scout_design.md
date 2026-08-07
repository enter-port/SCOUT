# SCOUT 设计文档(Design Spec)

> **状态**:设计已与用户逐节确认(§1–§6,2026-08-07)。待用户对本文档过审后转入实现规划(writing-plans);**任何代码落实仍需用户审核**。
> **相关**:[`idea.md`](idea.md)(导师原始 idea)、[`stage1_plan.md`](stage1_plan.md)(阶段 1 实验计划)、[`evaluation_plan.md`](evaluation_plan.md)(评估口径)、memory `lpb-reference.md`(LPB 代码库参考)。
> **优先级**:与 `stage1_plan.md` 冲突处,以本文档为准——① guidance 注入"改 trajectory、不改 ε";② eval 含 self-improvement loop;③ 训练为联合训练。

---

## 0. SCOUT 是什么

导师 classifier-guided exploration idea 的落地,对标学长 SOE 的成熟度。三件套:

1. **冻结 base Diffusion Policy**——训练时不在场;测试期提供去噪 score(动作分布)。
2. **潜空间 VIB 动力学模型**——学一个与 $\mathcal N(0,I)$ 对齐的 skill 潜空间 $z$;信息论目标 $\max\ I(Z;S_{t+1}\mid S_t) - \beta\,I(Z;A_t\mid S_t)$。
3. **测试期 classifier guidance**——采样 $z$,在 base DP 去噪循环里把动作推向"编码回去 $\approx z$"的方向,产生**有意义的多样性探索**,驱动 multi-round self-improvement。

**代码来源**:
- base DP ← **完全照搬 SOE 的 `DP`**。
- 测试期 guidance 注入 ← **完全照搬 LPB 的 `guided_conditional_sample`**(标准 classifier-guided denoising)。
- VIB 动力学模型、数据 loader、self-improvement 编排 ← **全新写**(只借理念)。
- 代码库:**全新最小实现**(独立,不依赖 SOE/lpb 目录);SOE 的 DP 与 robomimic rollout 脚手架搬入。

---

## 1. 总览(管线)

潜空间动力学(world-model 味),VIB 是中间核心:

```
训练(base DP 不在场,单链、单次 backward):
  S_t ─[E_s]→ s̄_t ─┐
                    ├─[VIB enc]→ (μ,logvar) →reparam z ─┐
          a_t ─────┘                                     │
                                                          ├─[VIB dec]→ ŝ̄_{t+1} ─[D_s]→ Ŝ_{t+1}
                                         s̄_t ────────────┘
  loss = AE重建 + next-latent MSE + β·KL    (联合训练,详见 §3)

测试(q_φ / D_s 下线;只用 μ + 冻结 base DP):
  z ~ N(0,I)  整段 chunk 定住
  → 在 base DP 去噪循环注入 ∇_{x_t}[ −‖z − μ(s̄_t, a)‖ ]   (LPB 范式,详见 §4)
```

双观测路径:**low_dim(stage 1)** $E_s/D_s$=MLP;**image(stage 2)** $E_s/D_s$=CNN,其余结构不变(§6)。

---

## 2. 网络与维度

`EncoderMLP` block(SOE `src/policy/vqvae_modules/vqvae.py:12`):
`Linear(in→hid)→ReLU → [Linear(hid→hid)→ReLU]×layer_num → Linear(fc: hid→out)`,`hidden_dim=128, layer_num=1`,无 norm/dropout。

| 网络 | 结构 | 维度(stage-1 low_dim lift) |
|---|---|---|
| `E_s / D_s` | MLP 自编码器(`EncoderMLP`) | state(≈19,从 hdf5 读)↔ s̄_t(**32**) |
| VIB encoder | `concat(s̄_t, a_t) → EncoderMLP → (μ,logvar)` | in = 32 + action_dim;out = 2·style_dim = **32** |
| VIB decoder | `concat(z, s̄_t) → EncoderMLP → ŝ̄_{t+1}` | in = 16 + 32;out = 32 |
| skill latent `z` | reparam:`z = μ + σ·ε`,`σ = exp(0.5·logvar)` | style_dim = **16** |
| **base DP** | **SOE `DP`**:`MultiImageObsEncoder`(low_dim: sorted keys identity 拼接;image: per-key ResNet-18)+ 可选 `bottleneck`(`EncoderMLP`)+ `DiffusionUNetPolicy`(ε-DDPM) | obs_feature_dim = Σ低维 key 维(≈19);动作 chunk `(B, 20, action_dim)`;测试期冻结 |

> 动作维度须 **base DP 与 VIB encoder 一致**(同一 hdf5)。low_dim lift:raw delta = 7,abs_6drot = 10;实现时按选用的数据文件定。

**文件映射**(实现时搬入 / 参照):
- base DP:`SOE/src/policy/dp.py`、`diffusion.py`(`DiffusionUNetPolicy` + `conditional_sample`,注入点 :187 / :197)、`img_encoder/multi_image_obs_encoder.py`、`vqvae_modules/vqvae.py`(`EncoderMLP`)、`img_encoder/crop_randomizer.py`、`common/pytorch_util.py`、`dataset/robomimic_v2.py`。
- VIB 模块:参照 SOE `dp_ext.py:72-81`(down/up_module 用 `EncoderMLP`)的同款 block,只改 I/O。
- guidance 注入:参照 LPB `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:212-271`(`guided_conditional_sample`)。

---

## 3. 训练(联合 + 数据解耦)

**联合训练**(单阶段,$E_s/D_s/$VIB enc/dec 一起训):
- 前向(一个 transition batch):$s̄_t = E_s(S_t)$ → $(μ,\logvar) = \text{VIB\_enc}(s̄_t, A_t)$ → $z = \text{reparam}$ → $\hat{s̄}_{t+1} = \text{VIB\_dec}(z, s̄_t)$。
- loss(一次 backward 更新全部):
$$\mathcal L = \underbrace{\|D_s(E_s(S_t)) - S_t\|^2 + \|D_s(E_s(S_{t+1})) - S_{t+1}\|^2}_{\text{AE 重建(锚,$S_t$ 与 $S_{t+1}$ 都重建)}} + \underbrace{\|\hat{s̄}_{t+1} - E_s(S_{t+1})\|^2}_{\text{next-latent 动力学(不 detach)}} + \underbrace{\beta\,\mathrm{KL}[\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,I)]}_{\text{KL}}$$
- **AE 重建是防坍缩锚**:动力学不 detach 让 $E_s$ 朝"好预测"漂;AE 重建在 $S_t$ **和** $S_{t+1}$ 上都施加(否则非 detach 的 $E_s(S_{t+1})$ 只被动力学拉向预测、即坍缩方向)→ 钉死"可还原 $S$"→ 不塌成平凡解。(可选:AE-only warmup 几个 epoch 再开动力学,当稳定旋钮。)
- base DP **不在场**;单链、单次 backward、**无梯度隔离**。
- **β = make-or-break 旋钮**:E1 扫 $\beta \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}\}$ + 生死诊断(§5)定。

**数据 loader 解耦**(`data/` 模块):
- 核心单元 = transition $(S_t, A_t, S_{t+1})$;`S_t` 为拼好的状态向量(AE 重建直接用 $S_t/S_{t+1}$,不另开 state 流)。
- 抽象 `TransitionSource` 接口(= `ReplayBuffer`):`sample(batch) → {S_t, A_t, S_{t+1}}`、`add(transitions)`、`__len__`、`stats()`。
- **可插拔后端**:robomimic low_dim(现)/ 真机(后)/ 其他仿真(后);每个后端只管"怎么把自家数据拼成 $(S_t, A_t, S_{t+1})$",训练代码只认接口。
- **online training**:`add()` 边录边加(真机 teleop / rollout 产出 transition 直接进 buffer);训练从**不断增长**的 buffer 采样;归一化用 **running 统计(Welford 增量)**;add / sample 并发用共享内存或周期 rebuild。
- 该 buffer 同时是 self-improvement 回灌入口(§5)。

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
$$\text{cost}(\hat{x}_0,\, s) \;=\; \text{mean}_t\,\big\|\,z - \mu(s̄_t,\, a_t)\,\big\|_2,\quad a = \hat{x}_0,\ \ s̄_t = E_s(S_t)\ \text{(定住)},\ \ z\ \text{整段定住}$$
$\mu$ = VIB encoder 的均值(逐 chunk 步)。

**门控**:
- (a) `t < guidance_start_timestep`(最后 K 步引导)→ **保留**(标准 CG 做法)。
- (b) LPB 的 OOD 门 `current_cost > threshold` → **去掉**(SCOUT 是探索,每个 chunk 都主动引导;且无 expert-latent NN 距离)。

**归一化桥**(实现细节):cost 里要把 base DP(SOE 归一化)的动作 **unnormalize → 再 normalize 进 VIB 空间**(参照 LPB `dyn_model/planner.py:211-213`)。

---

## 5. 评估(= SCOUT self-improvement 闭环,metric 参照 SOE)

5 步 multi-round loop(探索用 §4 guidance,回灌用 §3 buffer):

```
Round 0:  DP₀ ← 训自部分 robomimic 数据(core demos,如 core_20)            [step 1]
          ↓ 训 VIB dynamics + z(§3,base DP 冻结)                          [step 2]
          ↓ 冻结 DP₀,采样 z 引导生成 exploration rollouts(§4,robomimic sim) [step 3]
          ↓ 筛成功 rollout → buffer.add → 合 core → 训 DP₁                 [step 4]
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

**实现依赖**:step 3 / 5 需 robomimic sim rollout + 判成功 → 把 SOE 的 robomimic rollout / env_runner 脚手架搬进 fresh 库。stage-1 在 robomimic **lift** low_dim 上跑。

---

## 6. 图像路径(stage 2)+ self-improvement 接口

**图像路径**:4 阶段管线**观测无关**,stage-1 → stage-2 只换 $E_s/D_s$(MLP → CNN;$E_s$ 参照 LPB `ResNetEncoder` 的 per-view ResNet)。VIB enc/dec(E1/D1,MLP)、guidance、base DP 不变——核心机制在 low_dim 证通后,图像是平滑扩展。

**stage-2 #1 风险(留 stage-2 定方案)**:图像路径若要 $D_s$ 解码回**像素**(latent → image),撞 next-state prediction 这个 world-model 级难题。两条回避路线届时二选一:
- (i) 学 LPB——只预测 next-**latent**、不解码像素;图像阶段 $E_s$ 复用 base DP 冻结 ResNet,或单独预训练一个 image AE;
- (ii) 在轻量离散 latent(VQ-AE)上做动力学,绕开像素重建。

→ stage-1 low_dim 不受影响,先证机制。

**self-improvement 接口**:已含在 §3 `ReplayBuffer` + §5 loop,无新组件。online 训练 = teleop / rollout 产出 transition → `buffer.add()` → 训练从增长 buffer 采样 + running 归一化增量更新。

---

## 7. 关键风险

1. **β 是 make-or-break**:太大 → $\mu$ 与动作脱钩 → guidance no-op → 探索死。靠 §5 生死诊断 + β 扫描把控。
2. **next-state / latent prediction**(stage-2 图像)= #1 工程风险;stage-1 low_dim 先绕开。
3. **联合训练坍缩**:靠 AE 重建锚 + 可选 warmup。
4. **归一化桥错位**:DP 动作空间与 VIB 动作空间不一致会让 cost 失真(§4 桥)。

---

## 8. 待定(stage-2 / 实现期再决)

- 图像路径像素解码方案(§6 的 (i) / (ii))。
- 具体超参(s̄_t = 32、style_dim = 16、hidden = 128、guidance_scale、guidance_start_timestep)。
- online buffer 并发实现(共享内存 vs 周期 rebuild)。
- 真机后端数据格式(teleop → transition)。
