# DDPM / DDIM 采样器全解:理论推导、bridge、确定性、两套代码逐行对照

> 2026-08-30。回答四个问题:① DDPM/DDIM 的所有细节;② "bridge" 的数学原理;③ 确定性 vs 不确定性的区别;④ 实现方法(我们 scout/exploit 与 SOE 两条路径的逐行对照)。
> 写法按惯例:推导链逐步标注【恒等】(数学等价变形)/【变分】(变分界)/【代理】(换目标、丢权重)/【构造】(人为定义)/【引用】(论文定理,不重推)。工程细节一行指针。
> 事实核对基准:服务器 scout venv `diffusers 0.27.2`(我们与 SOE campaign 实际运行版本),源码 `scheduling_ddpm.py` / `scheduling_ddim.py` 已逐行读过;本地代码行号以 exploit-dev 分支为准。

---

## 0. 一页速览:两边现状对照

| | **我们**(scout 全线,含 exploit) | **SOE baseline**(soe-scout-align campaign) |
|---|---|---|
| scheduler 类 | `diffusers.DDPMScheduler` | `diffusers.DDIMScheduler` |
| 训练目标 | ε-MSE(随机 t)+ DDPM 前向加噪 | **完全相同**(ε-MSE,`SOE/src/policy/diffusion.py:309-344`) |
| 推理步数 | `num_inference_steps=100` = 训练 T=100,**全链** t=99…0 | `num_inference_steps=20`(默认,未覆盖),网格 **[95,90,…,0]** 每 5 步一跳 |
| 每步加噪? | **加**:每步 ~ N(0, β̃_t·I),`variance_type: fixed_small`,t>0 共 99 次采样 | **不加**:`eta=0.0`(diffusers 默认,从未被覆盖)→ 每步 0 噪声 |
| 给定同一个 x_T | **仍然随机**(每步新噪声) | **完全确定**(采样器是 x_T 的确定函数) |
| 全部随机源 | x_T + 99×步间方差噪声 + VIB z(独立流) | 只有 x_T(+开启探索时的 VIB z / 条件特征扰动) |
| guidance 嵌入 | 每步:cost 在 x̂₀ 上,均值注入 `η_g·√(1−ᾱ_t)`,随后照常加方差噪声 | SOE 自己不引导(其探索扰动加在**条件特征**上,`diffusion.py:170-183`) |

一句话:**两边训练出的 ckpt 是同族模型(同一 ε-MSE 目标),差别纯在"怎么从 N(0,I) 走回报动作空间"**——这正是 SR 绝对值不可直接比较的根源(08-30 已记录于协议差异笔记)。

---

## 1. 记号与白话开场

扩散模型学的是"把噪声变回数据"的逆过程。训练时把干净数据一步步加噪直到纯噪声;采样时从纯噪声出发,用网络一步步"去噪"走回数据。

| 记号 | 含义(白话) | 在我们代码里的实体 |
|---|---|---|
| `x_0` | 干净数据 = 一整段动作序列 (T_a, 10)(DP 归一化空间) | `trajectory` 的目标形态 |
| `x_t` | 加了 t 步噪的版本,t ∈ {0,…,99} | 循环变量 `trajectory` |
| `ε` | 加噪时用的标准高斯噪声 | `compute_loss` 里 `torch.randn` |
| `ε_θ(x_t, t)` | 网络对"这一步里混了多少噪声"的预测 | `model(trajectory, t, …)` |
| `x̂_0` | 由 x_t 和 ε_θ 反解出的"干净动作序列一步估计" | `scheduler.step(...).pred_original_sample` |
| `α_t = 1−β_t` | 每步保留旧信号的比例 | 由 schedule 算出 |
| `ᾱ_t = ∏_{s≤t} α_s` | 累计保留比例;ᾱ→1 几乎没噪,ᾱ→0 几乎纯噪 | `scheduler.alphas_cumprod` |
| `β̃_t` | 反向一步的后验方差(见 §2.3) | `fixed_small` 用的就是它 |
| `z` | 采样时新抽的标准高斯(不是训练那个 ε) | `randn_tensor(...)` |
| `η_DDIM` | DDIM 的随机度旋钮(0=确定,1≈DDPM) | `DDIMScheduler.step(eta=…)` |
| `η_g` | guidance 力度(LPB 的 guidance_scale) | `self.guidance_scale`。**与 η_DDIM 无关,纯重名** |

schedule 实况(cosine,`squaredcos_cap_v2`,T=100,实测):`ᾱ_99 ≈ 2.4e-7`(x_T ≈ N(0,I))、`ᾱ_0 ≈ 0.9994`、DDIM 的 `final_alpha_cumprod = 1.0`(`set_alpha_to_one=True`)。

---

## 2. DDPM(Ho et al. 2020, arXiv:2006.11239)

### 2.1 前向加噪链【构造】

    q(x_t | x_{t-1}) = N( √α_t · x_{t-1},  β_t·I ),   t = 1…T

每步:旧信号缩 √α_t,再混 β_t 份高斯噪声。β_t 是人为定的 schedule(我们和 SOE 都是 cosine)。

### 2.2 任意时刻一步到位【恒等】

T 个高斯逐个卷积仍是高斯,直接合并:

    q(x_t | x_0) = N( √ᾱ_t · x_0,  (1−ᾱ_t)·I )

重参数化后就是训练时实际的加噪公式:

    x_t = √ᾱ_t · x_0 + √(1−ᾱ_t) · ε,   ε ~ N(0, I)

代码:两个 policy 的 `compute_loss` 里 `noise_scheduler.add_noise(trajectory, noise, timesteps)` 就是这一行。

### 2.3 反向 bridge 核(DDPM 的"桥")【贝叶斯恒等】

真正的推导核心。反着走一步时,**如果偷看 x_0**,后验条件分布是闭式高斯(Bayes:q(x_{t-1}|x_t,x_0) = q(x_t|x_{t-1})·q(x_{t-1}|x_0)/q(x_t|x_0),三个高斯指数配方):

    q(x_{t-1} | x_t, x_0) = N( μ̃_t ,  β̃_t·I )

    μ̃_t = [ √ᾱ_{t-1}·β_t / (1−ᾱ_t) ] · x_0  +  [ √α_t·(1−ᾱ_{t-1}) / (1−ᾱ_t) ] · x_t
    β̃_t = (1−ᾱ_{t-1})/(1−ᾱ_t) · β_t        (注意恒有 β̃_t < β_t)

**这一步转移核就是"bridge"**:它连接两个边缘固定的端点(x_t 与 x_{t-1} 的边缘都由前向过程钉死),中间的高斯条件分布像一座桥;桥面晃动幅度 = σ_t。DDPM 与 DDIM 的一切差别,只是**这座桥的修法**不同。

把 x_0 用 x_t,ε 反解代入(x_0 = (x_t − √(1−ᾱ_t)ε)/√ᾱ_t),μ̃_t 化简为【恒等,可逐项验证】:

    μ̃_t = ( x_t − β_t/√(1−ᾱ_t) · ε ) / √α_t

### 2.4 训练目标【变分 → 代理】

- 对 −log p_θ(x_0) 变分上界(ELBO),链式分解成 T 个 KL 项【变分】:
  `−log p_θ(x_0) ≤ L = Σ_t E_q D_KL( q(x_{t-1}|x_t,x_0) ‖ p_θ(x_{t-1}|x_t) ) + const`
- 取 p_θ 为对角高斯、方差固定不学【构造】→ KL 闭式 = 加权平方误差【恒等】
- 用 2.2 的重参数化把 μ̃ 写成 ε 的函数【恒等】→ 误差变成 ‖ε − ε_θ(x_t,t)‖² 乘一个权重
- **丢掉权重** → 简单 MSE【代理】(丢了 β_t²/(2β̃_tᾱ_t(1−ᾱ_t));这是 Ho et al. 的经验选择,不是严格等价):

      L_simple = E_{t,x_0,ε} ‖ ε − ε_θ( √ᾱ_t·x_0 + √(1−ᾱ_t)·ε , t ) ‖²

  代码:`compute_loss` 的 `F.mse_loss(pred, target)`,target=noise。**我们和 SOE 这部分逐字同构。**

### 2.5 DDPM 采样(ancestral sampling)【代理:真 x_0 未知 → 用 x̂_0 替】

真 x_0 采样时不知道 → 拿网络预测 **x̂_0 = (x_t − √(1−ᾱ_t)·ε_θ(x_t,t))/√ᾱ_t** 代进 2.3 的桥【代理】,再加一步噪声:

    x_{t-1} = c0 · x̂_0 + c1 · x_t + σ_t · z ,   z ~ N(0,I)
    c0 = √ᾱ_{t-1}·β_t/(1−ᾱ_t),   c1 = √α_t·(1−ᾱ_{t-1})/(1−ᾱ_t)
    σ_t² ∈ { β̃_t (diffusers 默认 fixed_small), β_t (fixed_large) }

- 等价的纯 ε 形式【恒等】:`x_{t-1} = (x_t − β_t/√(1−ᾱ_t)·ε_θ)/√α_t + σ_t·z`(diffusers 用的是 x̂₀ 形式 + clip,见 §6)
- σ_t 两种选法不改训练目标,只改采样方差;diffusers 默认取更小的 β̃_t
- **关键性质:每个 t 都独立抽一个新 z。** 所以 DDPM 链即使 x_T 完全相同,两次采样也给出不同 x_0——随机性遍布整条链,不只入口。

---

## 3. DDIM(Song et al. 2021, arXiv:2010.02502)

### 3.1 核心观察【变分分解】

2.4 的 ELBO 里每一项 KL **只涉及边缘分布** q(x_t|x_0)(因为 KL 双方都被 x_0 条件化后都是单步高斯)。⇒ 训练根本不关心前向"过程"是不是马尔可夫链,只钉死了每个**边缘**。⇒ 可以在保持全部边缘 q(x_t|x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I) 不变的前提下,把前向过程**整个换掉**,再反解出对应的新采样器——**一个训练,一族采样器**。

### 3.2 非马尔可夫前向族 + DDIM 桥核【构造 + 恒等验证】

构造一个参数化的反向一步核(DDIM 论文式 12 的 q 侧):

    q_σ(x_{t-1} | x_t, x_0) = N( μ_σ ,  σ_t²·I )
    μ_σ = √ᾱ_{t-1}·x_0 + √(1−ᾱ_{t-1}−σ_t²) · ( x_t − √ᾱ_t·x_0 ) / √(1−ᾱ_t)

**恒等验证(这就是"桥"的含义)**:设 x_t = √ᾱ_t x_0 + √(1−ᾱ_t)·ε,代入 μ_σ 得
x_{t-1} = √ᾱ_{t-1}·x_0 + √(1−ᾱ_{t-1}−σ²)·ε + σ·z′,方差合并 = (1−ᾱ_{t-1}−σ²)+σ² = **1−ᾱ_{t-1}**。
⇒ 无论 σ 取多少,复合后的边缘 q_σ(x_t|x_0) 与 DDPM 前向**严格一致**。桥的两端被焊死,只有桥面晃动幅度 σ 是自由参数。σ² ≤ 1−ᾱ_{t-1} 内任取;论文取

    σ_t = η_DDIM · √β̃_t      (β̃_t 即 DDPM 后验方差,§2.3)

- **η_DDIM = 1**:σ² = β̃_t → 采样器 ≈ DDPM(论文证明此时两个更新式统计等价)
- **η_DDIM = 0**:σ = 0 → 核变成**确定性映射**:

      x_{t-1} = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1})·ε_θ(x_t, t)
            = √ᾱ_{t-1}·x̂_0 + √(1−ᾱ_{t-1}) · (x_t − √ᾱ_t·x̂_0)/√(1−ᾱ_t)   ← "direction pointing to x_t"

  整条采样链从马尔可夫随机链退化成一个 (x_T, 条件) → x_0 的**确定函数**。论文 §4.1【引用】:这恰是概率流 ODE(probability-flow ODE)的一阶 Euler 离散——同一模型还隐含一个把噪声域和数据域连续连起来的常微分方程,η=0 就是在数值解它。
- 中间 η:确定性偏移 + 部分噪声的插值。

### 3.3 跳步(为什么 SOE 只走 20 步)

既然每步核只依赖**边缘量 ᾱ_t**(而非相邻关系),完全不必一步步走:任取递减子序列 99 = τ_S > … > τ_1 = 0,把公式里的 ᾱ_{t-1} 换成 ᾱ_{τ_{i-1}}、σ 同理,直接 x_{τ_i} → x_{τ_{i-1}}。1000→10/100 步、100→20 步都成立。这是 DDIM 比 DDPM 快一个量级的真正原因(DDPM 链理论上也能跳,但那是外推近似,DDIM 是精确保持边缘的构造)。
diffusers 的 `set_timesteps` 生成网格:我们实测 SOE 配置(num_train=100, 20 步, leading, steps_offset=0)= **[95,90,85,…,5,0]**;我们 DDPM 配置(100/100)= [99,98,…,0]。

---

## 4. "bridge" 到底指什么(两个所指,本文主线是前者)

1. **数学的 bridge(主线)**:两端边缘被钉死时,连接 x_t 与 x_{t-1}(或 x_0)的条件高斯转移核。扩散采样器 = 一串首尾相接的桥。DDPM 的桥 = 后验核(σ²=β̃_t,每步必晃);DDIM 的桥 = σ 自由族(σ=η√β̃;σ=0 时绷成直线 = 确定性 ODE)。DDIM §3 的原话就是 "these distributions can be seen as … a bridge" 语义下的 subsequence/桥式推理;diffusers 源码注释逐字标注了公式(12)/(16) 的 "direction pointing to x_t"。
2. **代码字面的 bridge(附注)**:`scout/eval/rollout.py:99 make_action_bridge` + `scout/normalizer.py` 的 `ActionNormalizerBridge`/`UnnormalizeOnlyBridge`——把 DP 归一化空间里的 x̂₀ 可微仿射映回原始动作空间供 VIB cost 用(梯度可穿过);旋转表示 6d→axis-angle 在 env adapter。与采样器无关,一行指针即可。

---

## 5. 确定性 vs 不确定性

### 5.1 随机源清单(一条 guided action chunk 的全部随机性)

| # | 随机源 | 我们(DDPM) | SOE(DDIM η=0) |
|---|---|---|---|
| ① | x_T ~ N(0,I)(`torch.randn`,走 `generator`) | 有 | 有 |
| ② | 每步方差噪声(t>0,共 99 次) | **有** | 无(η=0 分支整段跳过) |
| ③ | VIB 的 z(独立 RNG 流,刻意不占 `generator`) | 有(explore 每 chunk 新抽或 locked) | 有(SOE 每 chunk 重抽 z) |
| ④ | 条件特征扰动(仅探索模式) | 无 | 有(`apply_modal_level_exploration`,扰动 global_cond) |

### 5.2 差别推到底

- **SOE**:采样器本身确定 ⇒ p_θ(·|s) 实现为确定映射 f(x_T, s)。同一场景下 retry 的多样性**全部来自换 x_T(和 z)**;固定住这两者,动作逐位复现。其探索 = 在条件空间(z/特征)撒点,采样器只是确定投影。
- **我们**:即使 x_T 相同,链上 99 次新噪声使两次调用 = p_θ(·|s) 的两个 i.i.d. 样本 ⇒ "同场景重试"天然就是重采样;retry 多样性在**采样器内部**。
- **对 guidance 的语义差别**:我们的引导 = 在随机马尔可夫链上逐步把均值往 cost 下降方向扳(类 Langevin 动力学,注入的均值偏移随后被方差噪声部分冲淡);η=0 下的引导 = 确定性流的连续变形(每步的注入沿后续确定链 100% 保留)。两边 guidance 即使公式相同,作用于随机对象不同——这就是"guidance 机制不同源"的准确含义。
- **协议后果**:SR 绝对值不可直接比(随机重试 vs 确定映射的抽样性质不同);我们若要做"固定 x_T 消融",DDPM 下必须把 99 步方差噪声也一并钉住(同一个 `generator` 一路传到底,或给 `DDIMScheduler.step(variance_noise=…)` 风格的外部供噪——DDPM 的 `step` 只收 `generator`)。

### 5.3 复现性与一个真实工程坑

- 顺序:`x_T` 与每步方差噪声共用 `generator`(seed 控),z 走独立流。scale=0 时 guided 轨迹与 unguided **逐位相同**(verify 过的不变量)。
- 坑(已修,`scout/guidance/policy.py:189-201`):读 x̂₀ 那次 `scheduler.step(model_output, t, trajectory)`(不传 generator)在 DDPM 下会**顺手从全局 RNG 抽一次方差噪声**(只有 `prev_sample` 用到,x̂₀ 本身确定)→ 会打乱主随机流。修法 = 一次性 throwaway generator `_gate_gen` 接住这个副作用抽签。

---

## 6. 实现方法(逐行)

### 6.1 我们这一侧(scout / exploit-dev)

**配置** `configs/base_dp_can_image.yaml:62-75`:
`DDPMScheduler, num_train_timesteps=100, beta_schedule=squaredcos_cap_v2, variance_type=fixed_small, clip_sample=true, prediction_type=epsilon`;`num_inference_steps: 100` → `set_timesteps(100)` = 全链 [99…0](DDPM 只允许 ≤T)。

**采样循环** `scout/guidance/policy.py:110-325`(`guided_conditional_sample`,循环体逐字承 LPB `diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:212-271`,只换 cost、删/改门):

| 行 | 代码 | 数学 |
|---|---|---|
| 179-184 | `trajectory = torch.randn(..., generator=generator)` | x_T ~ N(0,I)(源①) |
| 187 | `scheduler.set_timesteps(self.num_inference_steps)` | 生成 [99…0] |
| 249 | `trajectory[condition_mask] = condition_data[...]` | inpaint 条件(obs_as_global_cond 下 mask 全空,实际不动) |
| 250 | `trajectory.detach().requires_grad_()` | 让 cost 梯度能对 x_t 求导 |
| 253-255 | `model_output = model(trajectory, t, ...)` | ε_θ(x_t, t) |
| 262-263 | 门:`t < guidance_start_timestep`(t 从 99 降 → gst=100 = 全程)+ exploit 的 OOD 软门 `_ood_w` | 引导窗口/力度门 |
| 267-269 | `x0_hat = scheduler.step(model_output, t, trajectory, generator=_gate_gen).pred_original_sample` | x̂₀ = (x_t − √(1−ᾱ_t)ε)/√ᾱ_t(已 clip 到 ±1,见下) |
| 284-286 | `loss = planner.compute_loss(x0_hat, obs, reduction="sum")` | cost 在 x̂₀ 上;sum 归约修 1/B bug(`idea/guidance_batch_scaling_bug.md`) |
| 287 | `cond_grad = -autograd.grad(loss, trajectory)[0]` | cost 对 x_t 的下降方向 |
| 290-293 | `grad_scale = guidance_scale · √(1−ᾱ_t) · w`;`trajectory = trajectory.detach() + grad_scale·cond_grad` | 均值注入;√(1−ᾱ_t) 让注入量与该步噪声 std 同阶(自 t≈99 的 ~1 衰减到 t=0 的 ~0.025) |
| 316-318 | `trajectory = scheduler.step(model_output, t, trajectory, generator=generator).prev_sample` | §2.5 的桥 + 方差噪声(源②) |
| 321 | 最终 pin 条件 | 收尾 |

LPB 原版的门 `current_cost > self.threshold`(OOD 才引导)在 exploit 分支以 `ood_threshold`/`gate_weight`(软门 `min(slope·(cost−thr)/thr, cap)`,commit 38fb92a)形式回归;纯探索 planner 无此属性 → 门恒开。

**diffusers 0.27.2 `DDPMScheduler.step` 公式↔源码映射**(服务器 venv 实读):

    pred_original_sample = (x_t − √(1−ᾱ_t)·ε̂) / √ᾱ_t        # "epsilon" 分支
    pred_original_sample = clamp(pred_original_sample, ±1)    # clip_sample=true → 我们 guidance 看到的 x̂₀ ∈ [−1,1]^d
    pred_prev_sample   = c0·x̂₀(clipped) + c1·x_t             # c0=√ᾱ_{t-1}β_t/(1−ᾱ_t), c1=√α_t(1−ᾱ_{t-1})/(1−ᾱ_t) = Ho eq.(7)
    variance           = √β̃_t · z   (fixed_small, clamp 1e-20; t>0 才抽)   # 源②

### 6.2 SOE 那一侧(soe-scout-align campaign)

- `SOE/src/policy/diffusion.py:43` 默认 `noise_scheduler_type="ddim"` → 66-76 行构造 `DDIMScheduler(num_train_timesteps=100, cosine, clip_sample=True, set_alpha_to_one=True, steps_offset=0, epsilon)`;`:37` 默认 `num_inference_steps=20`。
- 采样 `:135-213`:x_T=randn(:146) → set_timesteps(20) → 网格 [95,90,…,0] → 循环 `scheduler.step(model_output, t, trajectory, generator=..., **kwargs)`(:197-201)。**kwargs 里从来没有 eta** —— `SOE/simulation/rollout_utils.py:304-306` 只在 `args.eta is not None` 时才把 `{"eta": ...}` 塞进 kwargs,而 campaign 入口 `run_scout_align.py:53` 默认 `eta=None` ⇒ diffusers 签名默认 `eta=0.0` 生效 ⇒ **整条链零步间噪声、20 步确定映射**。这就是"diffusion.py:43 默认未覆盖"的完整链条。
- t=0 那步的 ᾱ_{prev} 用 `final_alpha_cumprod = 1.0`(`set_alpha_to_one=True`;若 False 则用 ᾱ_0≈0.9994,差异 ~6e-4,可忽略)。
- SOE 自己的探索噪声不走采样器:`diffusion.py:170-183` 在**条件特征**上按 `tau1/tau2` 窗口加噪(`apply_modal_level_exploration`)。训练侧 `:309-344` 与我们同构的 ε-MSE。

### 6.3 两种 step 的公式对照(同一行意义的两套写法)

| 步骤 | DDPM step(diffusers) | DDIM step(η=0) |
|---|---|---|
| x̂₀ | (x_t − √(1−ᾱ_t)ε̂)/√ᾱ_t,再 clip | 同左,同 clip |
| 均值 | c0·x̂₀ + c1·x_t(Ho eq.7 后验均值) | √ᾱ_{t-1}·x̂₀ + √(1−ᾱ_{t-1})·ε̂(DDIM eq.12) |
| 两者关系 | 【恒等】把 x̂₀ 定义代入 DDIM 均值,严格等于 DDPM 后验均值(σ²=β̃ 时的 μ̃) | 同左 |
| 噪声 | + √β̃_t·z,每步必抽(t>0) | η=0:+0;η>0:+ η√β̃_t·z |
| 步长 | 固定 −1 | 任意跳(只看 ᾱ 网格) |

(表第三行是关键:两个均值公式是同一个东西的两种参数化,差的**只有方差项**——DDPM 强制 σ²=β̃_t,DDIM 把 σ 拿出来当 η_DDIM·√β̃_t。)

---

## 7. 常见追问速答

- **同一个 ckpt 为什么两边都能采样?** 训练目标(ε-MSE)只钉边缘,不含任何采样器信息;scheduler 是纯推理期选择。
- **为什么是 100 而不是论文的 1000?** Chi et al. Diffusion Policy 的约定(T=100 + cosine);两边一致,公式不含 T 特异性。
- **fixed_small vs fixed_large 差多少?** β̃_t = β_t·(1−ᾱ_{t-1})/(1−ᾱ_t) < β_t;diffusers 默认取小的(Ho 论文两选皆可,只影响采样方差)。
- **clip_sample 对我们意味着什么?** guidance 的 cost 永远看到 clip 后的 x̂₀ ∈ [−1,1]^d(DP 归一化动作空间),动作分量不会把梯度引到物理外区域。
- **DDPM 想钉死轨迹怎么办?** 只能整条 `generator` 一个 seed 贯穿(x_T+99 步全由它派生);或换 DDIM η=0 跑消融(改 config `noise_scheduler._target_` 为 DDIMScheduler + num_inference_steps,无需重训)。
- **η_g·√(1−ᾱ_t) 的 √(1−ᾱ_t) 是哪来的?** LPB 承 classifier guidance(Dhariwal & Nichol 2021)的标定:让注入量与该步采样噪声 std 同阶,大 t 强、小 t 衰减;与 η_DDIM 无关(重名,见 §1)。

## 8. 术语表

- **前向/反向过程**:加噪链 / 去噪链。
- **bridge(桥)**:两端边缘固定的条件高斯转移核;"桥面晃动"= σ。
- **ancestral sampling**:DDPM 式逐步采样,每步从 N(μ̃, β̃) 抽一个新样本。
- **x̂₀ / pred_original_sample**:当前 x_t 一步反解出的干净动作序列估计。
- **probability-flow ODE**:与 SDE 同边缘的确定性问题;DDIM η=0 是它的 Euler 离散。
- **leading spacing / steps_offset**:diffusers 生成推理时间网格的方式;我们全部 offset=0。
- **placebo / gate 等 SCOUT 术语**:见 `soe_scripts/rand_ideas/NAMES.md`,不在本文展开。

## 参考

- Ho et al., *Denoising Diffusion Probabilistic Models*, arXiv:2006.11239(后验闭式 eq.6-7、L_simple)
- Song et al., *Denoising Diffusion Implicit Models*, arXiv:2010.02502(式 12/16、非马尔可夫族、η、ODE 观点)
- Dhariwal & Nichol, *Diffusion Models Beat GANs*(classifier guidance,LPB 注入机制的出处)
- Nichol & Dhariwal, *Improved DDPM*(cosine schedule)
- Chi et al., *Diffusion Policy*(T=100、ε-prediction、clip_sample 约定)
- diffusers 0.27.2 源码 `scheduling_ddpm.py` / `scheduling_ddim.py`(服务器 venv 实读,本文所有"源码行为"以此为准)
- 网络资料(交叉验证):[AI Summer diffusion math](https://theaisummer.com/diffusion-models/)、[LearnOpenCV DDPM](https://learnopencv.com/denoising-diffusion-probabilistic-models/)、[apxml DDIM sampling](https://apxml.com/courses/intro-diffusion-models/chapter-5-sampling-generation-process/ddim-sampling-algorithm)、[DDIM non-Markovian 解读](https://medium.com/@kdk199604/ddim-redefining-diffusion-sampling-with-non-markovian-dynamics-39faf2dbef6b)、[DDPM math(Medium)](https://joydeep31415.medium.com/the-math-behind-diffusion-models-ddpm-9fabe9c9f1d9)
