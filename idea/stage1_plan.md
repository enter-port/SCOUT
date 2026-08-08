# 阶段 1 详细计划｜robomimic `lift` 落地与验证(image + proprio)

> 本文是 [`scout_design.md`](scout_design.md)(权威设计)在 **stage 1(robomimic `lift`)** 上的执行计划。方法 / 公式 / 架构 / 代码来源以 `scout_design.md` 为准。
> **管线永远 image + proprio 同时输入**(LPB 式,**无 low_dim/image mode 之分、无 stage1/stage2 之分**);数据用 lift **image** 数据集。
> 实现遵循「代码须经用户审核」(memory `code-requires-user-approval`)。

---

## 一、数据

| 项 | 取值 |
|---|---|
| 任务 | **主:`lift`**;备:`can` |
| 数据集 | robomimic ph **image** hdf5:`SOE/simulation/datasets/<task>/ph/image_v141.hdf5`(含 `obs/<images>` + `obs/<proprio>` + `actions`;`states` 不再需要) |
| 观测 | **image**(per-view,如 `agentview` + `robot0_eye_in_hand`)+ **proprio**(`robot0_eef_pos`/`eef_quat`/`gripper_qpos`);n_views / 维度从 hdf5 读 |
| env state | **不再需要**(latent 级监督,target = `E_s(S_{t+1})` = 下一观测再编码) |
| 动作 | 从 hdf5 读;**base DP 与 VIB 同维**;chunk horizon = 20 |
| 划分 | `mask/train`、`mask/valid`;**core demos(core_20)** 训 base DP |
| 转移 | `({image_t, proprio_t}, a_t, {image_{t+1}, proprio_{t+1}})`,`action_offset=1`;target = `E_s(S_{t+1}).detach()` |
| 数据接入 | `TransitionSource`(=`ReplayBuffer`);**robomimic image 后端**先实现(图像重 → buffer 存索引/路径,不内联像素);真机 / 其他仿真后端后续插(`scout_design.md §3`) |

---

## 二、需要训练的部分

| 组件 | 训法 | 说明 |
|---|---|---|
| **base DP**(SOE **image** `DP`) | core demos 单独训(E0),训完**冻结** | `MultiImageObsEncoder`(per-key ResNet-18)+ `bottleneck` + `DiffusionUNetPolicy`;**其 ResNet 冻结复用给 `E_s`** |
| **E_s**(LPB 式双输入,无 AE) | **冻结 ResNet**(复用 base DP)+ **训练 proprio embed** | `{image, proprio} → s̄_t`(永远两个同时进) |
| **VIB encoder** | 联合训(E1) | `concat(s̄_t, a_t) → (μ, logvar)`;`style_dim = 16` |
| **D_s**(dynamics decoder) | 联合训(E1) | `concat(z, s̄_t) → ŝ̄_{t+1}`(**到此为止,无 decode**) |

> 测试期在线:base DP(冻结)+ VIB encoder 的 μ;D_s 下线。MLP block 用 SOE `EncoderMLP`(hidden 128)。VIB 训练时 base DP 的 **ResNet 冻结在线**(LPB 式,见 `scout_design.md §0`)。

---

## 三、训练流程

### E0 · base DP(image)
- 从 hdf5 读维度生成 **image** config → `train_base_dp`(SOE DP 搬入)→ 训到 loss 平台。
- 产物 `policy_last.ckpt`,后续**冻结**复用(= self-improvement loop 的 DP₀),且其 **ResNet 给 `E_s` 复用**。

### E1 · VIB 联合训练 + β 扫描 + 生死诊断(前置)
- 对 β ∈ {1e-4, 1e-3, 1e-2, 1e-1} 各**联合训**一个 (VIB_enc / D_s / proprio_embed;`E_s` 的 ResNet 冻结):单链、单次 backward,base DP 的 ResNet **冻结在线**。
- loss = latent MSE(`ŝ̄_{t+1}`, `E_s(S_{t+1}).detach()`)+ β·KL(latent 级监督 = LPB,见 `scout_design.md §3`);**无 AE / 无 state decoder / 无重建**。
- 数据:transitions `({image,proprio}_t, a, {image,proprio}_{t+1})` 经 `TransitionSource` 采样。
- 逐 β 记录(画 vs β):latent MSE、KL、μ 的 mean/std(≈ N(0,I)?)。
- **生死诊断** `‖∂μ/∂a‖`:`A_t.requires_grad_(True)` → `autograd.grad(μ.sum(), A_t)` → **敏感比 = `‖∂μ/∂a‖·σ_a / σ_μ`**。
- **β 选择**:敏感比 ≥ ~0.3(先定后校准)的**最大** β,且 next-state MSE 未显著升高、μ ≈ N(0,I)。

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

E0 过 + E1 找到合格 β + E2/E3 闸门过 + **E4 显示 `DP_new > DP_old`(success rate / round 升)** ⇒ **GO**(机制成立,后续 scale 到更多任务 / 真机)。
E1 全空间无合格 β ⇒ 机制不通 ⇒ **NO-GO**,重审 idea。

> **开销**:单卡(图像比 low_dim 重);1 image base DP + 4 VIB(β 扫)+ 闸门 + loop。
