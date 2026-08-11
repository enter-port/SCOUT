# 阶段 1 计划｜SCOUT 端到端单轮验证(image + proprio)

> 本文是 [`scout_design.md`](scout_design.md)(权威设计)的**执行计划**。方法 / 公式 / 架构 / 代码来源以 `scout_design.md` 为准;冲突处以 `scout_design.md` 为准。
> **管线永远 image + proprio 同时输入**(LPB 式,**无 low_dim/image mode 之分、无 stage1/stage2 之分**)。
> 实现遵循「代码须经用户审核」(memory `code-requires-user-approval`)。
>
> **本次修订(2026-08-11)**:去掉原 E1 的 β 扫描 + E2/E3 action 闸门,改成 **Step 0→4 线性端到端**,先用 **lift** 跑通一轮(can/square 同配方跟进)。

---

## 总览(5 步线性)

```
Step 0  base DP(冻结)              ← 已在跑(3 task)
Step 1  训 dynamics(全量数据, β=1e-3)
Step 2  采样 z + classifier guidance → 100 trajectory
Step 3  回灌(成功 rollout + core)→ 训 exploreDP
Step 4  baseDP vs exploreDP 同环境对比(SOE 指标)
```

**任务**:lift **首发**验证;can / square **同配方跟进**(base DP 已在 3 task 并行训练,见 `experiments/experiment_log.md` Part 1)。

---

## Step 0 · base DP(已在跑)

- 3 task(lift / can / square),**core 数据**(lift core_10、can/square core_20),600 epoch,LPB 超参。
- 产物:每 task 选 **`test_mean_score` 最高的 ckpt → 冻结 = DP₀**;其 **ResNet 给 `E_s` 复用**(冻结)。
- 配置 / 超参 / 数据格式详见 [`experiments/experiment_log.md`](../experiments/experiment_log.md) Part 1。

---

## Step 1 · 训 dynamics(全量数据,单 β)

### 数据
- **该 task 全部 demos 的 transitions** `({image,proprio}_t, a_t, {image,proprio}_{t+1})`(**不是 core 子集**;lift/can/square 各 200 demos 全量)。
- target = `E_s(S_{t+1}).detach()`(**latent 级监督,无 state decoder**)。
- ⚠️ **相对 LPB 的偏离**:LPB 用 **base 策略 rollout** 训 dynamics;本计划用**全量 expert demo**(更干净、更易训,代价是少一层"策略分布外"覆盖)。按用户字面"全部 robomimic 数据"理解。
- 数据接入:LPB `RobomimicImageDynamicsModelDataset`(zarr 缓存 + `DataLoader` + `LinearNormalizer`);**不用** `ReplayBuffer`/`RunningStats`(见 `scout_design.md §3`)。

### 网络(详见 `scout_design.md §2`)
| 网络 | 结构 | 维度 |
|---|---|---|
| `E_s` | 冻结 base-DP ResNet(per-view)+ 训练 proprio embed → concat | `s̄_t = 512·n_views + proprio_emb_dim` |
| `VIB_enc` | `concat(s̄_t, a_t) → EncoderMLP → (μ,logvar)` | out = `2·style_dim = 32` |
| `D_s` | `concat(z, s̄_t) → EncoderMLP → ŝ̄_{t+1}`(**无 decode**) | out = `s_bar_dim` |
| `z` | reparam `z = μ + σ·ε` | `style_dim = 16` |

### loss(单链、单次 backward;base ResNet 冻结在线、不更新,无梯度隔离)
$$\mathcal L = \big\|\hat{s̄}_{t+1} - E_s(S_{t+1}).detach()\big\|^2 + \beta\,\mathrm{KL}[\mathcal N(\mu,\sigma^2)\,\|\,\mathcal N(0,I)],\quad \beta = 10^{-3}$$

### 产物
- **一个** dynamics ckpt(`VIB_enc` + `D_s` + proprio embed)。
- **不做** β 扫描、**不做** 生死诊断 `‖∂μ/∂a‖`(用户定:单 β=1e-3 直接进 Step 2)。

---

## Step 2 · 采样 z + classifier guidance → 100 trajectory

- **在线网络**:冻结 base DP(DP₀)+ `VIB_enc` 的 **μ**;`D_s` **下线**。
- `z ~ N(0,I)`:**每条 rollout 各采一个**,chunk 内定住。
- **guidance**:LPB `guided_conditional_sample`,cost = $\text{mean}_t\,\big\|z - \mu(\bar s_t, a_t)\big\|_2`(算在 $\hat x_0$ 上,梯度对 $x_t$ 求,改 $x_t$ 不改 ε;缩放 $\eta\sqrt{1-\bar\alpha_t}$;最后 K 步引导。详见 `scout_design.md §4`)。
- `guidance_scale`(**待定**,先沿用 LPB 默认起,可调)。
- **rollout**:robomimic sim,**100 初始态 × `try_times=5`**;成功即止(`env.is_success()["task"]`)。
- **记录**:每条 success/fail、**jerk**(动作三阶差分)、供 Step 4 的 Pass@5 / yield。

---

## Step 3 · 回灌训 exploreDP

- 筛 **success** rollout(`env.is_success()["task"]` 为真)→ 写**增强 hdf5**(原 core_10/20 + 成功 rollout),SOE `run_full_multi_round` 式;**不用** in-memory buffer。
- 用 **base DP 同构**(`DiffusionUnetHybridImagePolicy`)重训 **exploreDP**(= DP₁)。从 scratch 或 warm-start(= base ckpt)——**待定**。
- 产物:exploreDP ckpt。

---

## Step 4 · baseDP vs exploreDP 同环境对比

- **同一批测试初始态**(与 Step 2 同 100,或独立 test 集——待定)。
- 指标(参照 SOE,定义见 [`evaluation_plan.md`](evaluation_plan.md)):

| 指标 | 来源 | 论证 |
|---|---|---|
| **success rate** | base vs explore 各在同初始态 rollout | **头条**:exploreDP 是否更好 |
| exploration yield | Step 2 成功 rollout 数 | 探索高产 |
| Pass@5 | Step 2:100 初始态、5 次内可解比例 | 覆盖广(SCOUT 最该赢) |
| jerk | Step 2 rollout 动作三阶差分 | on-manifold / 平滑 |

- **目标**:`exploreDP > baseDP`(success rate 升)⇒ 机制成立。

---

## 五、go / no-go

- **Step 4 `exploreDP > baseDP`** ⇒ **GO**(机制成立;后续:6 轮 multi-round、scale 到 can/square、β/超参调、可定向探索)。
- **`exploreDP ≤ baseDP`** ⇒ **NO-GO**,重审;**第一嫌疑 = β=1e-3 太大 → guidance no-op** —— 此时回头补**生死诊断 `‖∂μ/∂a‖`** 确认(本计划 Step 1 跳过了它,NO-GO 时必须补)。

---

## 六、默认假设(可改 —— 我按你的 4 步推测,不对就指出)

1. **dynamics 数据 = 该 task 全部 demos**(lift/can/square 各 200),**非跨任务**。
2. **先单轮**跑通(你的 4 步正好一轮);**6 轮 multi-round 是后续 scale**,不在 stage1。
3. **去掉 E2/E3 action 闸门**(含生死诊断,随你 β 选项);NO-GO 时才回头补诊断。
4. **lift 首发**跑通;**跑通后 can / square 并行**(各等自己的 base DP 训练完成)。先证明机制,再 scale。
5. metrics:Step 4 headline = **success rate**(base vs explore 同初始态);yield / jerk / Pass@5 从 Step 2 rollout 算。
