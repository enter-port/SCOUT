# 阶段 1 详细计划｜robomimic `lift` low_dim 落地与验证

> 本文是 [`scout_design.md`](scout_design.md)(权威设计)在 **stage 1(robomimic `lift` low_dim)** 上的执行计划。方法 / 公式 / 架构 / 代码来源以 `scout_design.md` 为准,本文不重复,只写「stage 1 具体做什么、训什么、怎么评、过线条件」。
> 实现遵循「代码须经用户审核」(memory `code-requires-user-approval`)。

---

## 一、数据

| 项 | 取值 |
|---|---|
| 任务 | **主:`lift`**;备:`can` |
| 数据集 | robomimic ph low_dim hdf5:`SOE/simulation/datasets/<task>/ph/low_dim_v141.hdf5` |
| 观测 | 全 low_dim keys(`robot0_eef_pos` / `eef_quat` / `gripper_qpos` + `object`),维度从 hdf5 读;`state_dim = Σ`(lift ≈ 19) |
| 动作 | 从 hdf5 读;raw delta = 7 / abs_6drot = 10,**base DP 与 VIB 必须同维**;chunk horizon = 20 |
| 划分 | `mask/train`、`mask/valid`;**core demos(如 core_20)** 训 base DP |
| 转移 | `(S_t, A_t, S_{t+1})`,`action_offset=1`;`obs_keys = sorted(...)` 拼接 |
| 数据接入 | 走 `TransitionSource` 接口(= `ReplayBuffer`);**robomimic low_dim 后端先实现**,真机 / 其他仿真后端后续插(`scout_design.md §3`) |

---

## 二、需要训练的部分(stage 1 要训的组件)

| 组件 | 训法 | 说明 |
|---|---|---|
| **base DP**(SOE `DP`) | 用 core demos 单独训(E0),训完**冻结** | `MultiImageObsEncoder` + `bottleneck` + `DiffusionUNetPolicy`;之后全程冻结,VIB 训练时不在场 |
| **E_s**(编码器,LPB 式,无 AE) | low_dim:**identity,不训**;image(stage-2):冻结 ResNet + proprio embed | low_dim: `s̄_t = S_t` |
| **VIB encoder** | 联合训(E1) | `concat(s̄_t, a_t) → (μ, logvar)`;`style_dim = 16` |
| **D_s**(dynamics decoder) | 联合训(E1) | `concat(z, s̄_t) → ŝ̄_{t+1}` |
| **state decoder** | 联合训(E1) | `ŝ̄_{t+1} → Ŝ_{t+1}`(低维 env state);**推理下线** |

> 测试期在线的只有 **base DP(冻结)+ VIB encoder 的 μ**;D_s、state decoder 下线。MLP block 用 SOE `EncoderMLP`(hidden 128)。

---

## 三、训练流程

### E0 · base DP
- 从 hdf5 读 obs/action 维度生成 config → `train_single_gpu.py`(策略 `DP`,SOE 搬入的代码)→ 训到 loss 平台。
- 产物 `policy_last.ckpt`,后续**冻结**复用(= self-improvement loop 的 DP₀)。

### E1 · VIB 联合训练 + β 扫描 + 生死诊断(前置)
- 对 β ∈ {1e-4, 1e-3, 1e-2, 1e-1} 各**联合训**一个 (VIB_enc / D_s / state_dec;`E_s` low_dim=identity 无参):单链、单次 backward,base DP 不在场。
- loss = next-state MSE(`Ŝ_{t+1}`, `S_{t+1}`)+ β·KL(见 `scout_design.md §3`);**无 AE / 无重建 / 无 latent 级**。
- 数据:transitions 经 `TransitionSource` 采样。
- 逐 β 记录(画 vs β):next-state MSE、KL、μ 的 mean/std(≈ N(0,I)?)。
- **生死诊断** `‖∂μ/∂a‖`:`A_t.requires_grad_(True)` → `autograd.grad(μ.sum(), A_t)` → **敏感比 = `‖∂μ/∂a‖·σ_a / σ_μ`**("a 抖一个 std 时 μ 的相对移动")。
- **β 选择**:敏感比仍 ≥ ~0.3(先定后校准)的**最大** β,且 next-state MSE 未显著升高、μ ≈ N(0,I)。

---

## 四、评估流程

### 前置 action 级闸门(便宜,跑 loop 前过)
**E2 · guidance 三判据**:选定 β 的 VIB + E0 base DP,classifier-guided 采样(LPB 注入,`scout_design.md §4`),sweep `guidance_scale` ∈ {0, 1, 5, 10, 20},固定初始噪声 + 固定一组 z。
- ① **多样性**:跨 z 的动作 std(scale=0 时 ≈ 0、随 scale 单调升、最大 scale 显著 > 0);
- ② **一致性**:`‖z − μ(s̄_t, a_guided)‖` 随 scale 降;
- ③ **Cost 方向**:去噪步上 `‖z − μ‖` 降;若升 ⇒ 翻转 `guidance_scale` 符号。

**E3 · on-manifold**:E2 最大 scale 的引导动作 chunk。**jerk** + 到 demo 动作分布的 **Mahalanobis**,均应与 base DP **同量级**。

### 主 eval · self-improvement loop(`scout_design.md §5`)
**E4 · 5 步闭环**:`DP₀`(core_20)→ 训 VIB(E1)→ 冻结 `DP₀`、采样 `z` 引导生成 exploration rollouts(robomimic sim)→ 筛成功 rollout → `buffer.add` 合 core → 训 `DP₁` → 多轮滚。
- **metric(参照 SOE)**:success rate / round、Pass@5、exploration yield、jerk。
- 默认:core_20、N=100、try_times=5、6 轮;成功判定 sim `env.is_success()["task"]`。

---

## 五、go/no-go

E0 过 + E1 找到合格 β + E2/E3 闸门过 + **E4 显示 `DP_new > DP_old`(success rate / round 升)** ⇒ **GO**,进 stage 2(图像观测)。
E1 全空间无合格 β ⇒ 机制不通 ⇒ **NO-GO**,重审 idea。

> **开销**:单卡;1 base DP + 4 VIB(β 扫)+ 闸门 + loop。迭代快,符合 stage 1「想清楚 + 证机制」的定位。
