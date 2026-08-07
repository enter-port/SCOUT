# SCOUT stage-1(low_dim)实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: 用 superpowers:subagent-driven-development(推荐)或 superpowers:executing-plans 逐任务实现。步骤用 `- [ ]` 跟踪。
> **权威设计**:[`scout_design.md`](scout_design.md);**实验口径**:[`stage1_plan.md`](stage1_plan.md)。本文档冲突处以设计文档为准。
> **约束**:遵循「代码须经用户审核」(memory `code-requires-user-approval`)——本计划是规划、不是代码;真正落代码每个任务都要可审核、可回退,且整体执行前需用户点头。

**Goal:** 在 robomimic `lift` low_dim 上实现并验证 SCOUT(VIB 潜空间动力学 + LPB 式 classifier guidance + self-improvement loop),跑到 go/no-go。

**Architecture:** 冻结 base DP(SOE `DP` 移植)+ 自训 VIB(`E_s/D_s` 自编码器 + VIB enc/dec,联合训练)+ 测试期 LPB 式 guidance 注入 + 5 步 self-improvement 闭环。见 `scout_design.md §1`。

**Tech Stack:** PyTorch(SOE 用 1.13,沿用)、diffusers(DDPM scheduler)、robomimic、h5py。无 pytest 基建 → 验证 = 形状/前向 sanity check + 小训练跑(loss 下降)+ E0–E4 闸门。

**范围与接口约定(关键)**:本计划只落 **low_dim**;但下列位置留好 **image 可替换接口**,后续上图像只改这些、不动主干:
- `StateAE`(E_s/D_s):MLP(low_dim)/ CNN(image)由 config 切换(Phase 1)。
- `TransitionSource`:robomimic low_dim 后端先实现;image 后端后续插入(Phase 1)。
- base DP 的 `MultiImageObsEncoder` 本就模态无关(SOE 原生),图像只是换 `obs_shape_meta`。
- VIB enc/dec、guidance、cost 都只吃 `s̄_t` 向量,与模态无关 → 图像无需改。
- 图像路径的 #1 风险(像素重建)留到 stage-2,接口已为其预留(见 Phase 1 `StateAE` 注释 + `scout_design.md §6`)。

---

## 文件结构(fresh 代码库 `scout/`)

```
scout/
├── scout/
│   ├── data/
│   │   ├── transition_source.py   # TransitionSource 接口 + ReplayBuffer
│   │   ├── running_stats.py       # Welford 在线归一化
│   │   └── robomimic_lowdim.py    # robomimic low_dim 后端(产 transitions)
│   ├── model/
│   │   ├── mlp.py                 # EncoderMLP(从 SOE vqvae.py:12 移植)
│   │   ├── state_ae.py            # E_s/D_s —— 【image 接口点】MLP/CNN 可切换
│   │   ├── vib.py                 # VIB encoder + decoder
│   │   └── scout_vib.py           # 组合:前向 + 联合 loss(AE+next-latent+KL)
│   ├── policy/                    # base DP:从 SOE 整体移植
│   │   ├── dp.py                  # SOE DP(MultiImageObsEncoder + DiffusionUNetPolicy)
│   │   ├── diffusion.py           # DiffusionUNetPolicy + conditional_sample(guidance 注入点)
│   │   ├── multi_image_obs_encoder.py
│   │   ├── conditional_unet1d.py
│   │   ├── crop_randomizer.py
│   │   └── pytorch_util.py
│   ├── guidance/
│   │   └── cost.py                # SCOUT cost ‖z−μ(s̄_t,a)‖ + 归一化桥
│   ├── diagnose.py                # 生死诊断 ‖∂μ/∂a‖ + 敏感比
│   ├── train_base_dp.py           # E0:训 base DP(调移植来的 DP)
│   ├── train_vib.py               # E1:VIB 联合训练 + β 扫描
│   ├── eval/
│   │   ├── rollout.py             # robomimic rollout(从 SOE 移植)
│   │   ├── metrics.py             # success rate/Pass@5/yield/jerk
│   │   └── self_improvement.py    # E4:5 步闭环编排
│   └── normalizer.py              # 动作/状态 normalizer(归一化桥用)
├── configs/
│   ├── base_dp_lift_lowdim.yaml
│   ├── vib_lift_lowdim.yaml
│   └── eval_lift.yaml
├── scripts/                       # 数据下载/转换(参照 SOE simulation/)
└── README.md
```

**职责边界**:每个文件单一职责;`data/` 只管 transition 供给;`model/` 只管网络与 loss;`policy/` 是冻结的 base DP(不反向、不改);`guidance/` 只管 cost 与注入;`eval/` 只管 rollout+metric+编排。文件按职责分,不按技术层分。

---

## Phase 1 · 脚手架 + 数据模块

### Task 1.1:仓库脚手架
**Files:** Create `scout/` 目录树(见上)、`scout/README.md`、`setup.py`(或 `pyproject.toml`)、`.gitignore`。
- [ ] 建目录树;`README.md` 写一句话目的 + 指向 `scout_design.md`。
- [ ] `git init`(若尚未)、首提交 `chore: scaffold scout repo`。
- [ ] **验证**:`python -c "import scout"` 不报错(空 `__init__.py`)。

### Task 1.2:`EncoderMLP`(从 SOE 移植)
**Files:** Create `scout/model/mlp.py`;源 `SOE/src/policy/vqvae_modules/vqvae.py:12-44`。
- [ ] 把 SOE 的 `EncoderMLP`(含 `weights_init_encoder`)原样拷入 `mlp.py`,改 import 路径。
- [ ] **验证**:`python -c "from scout.model.mlp import EncoderMLP; import torch; m=EncoderMLP(19,32); print(m(torch.randn(4,19)).shape)"` → `torch.Size([4, 32])`。
- [ ] 提交 `feat: port EncoderMLP from SOE`。

### Task 1.3:`RunningStats`(Welford 在线归一化)
**Files:** Create `scout/data/running_stats.py`。
- [ ] 实现 `RunningStats`:`update(x)`(批量增量)、`mean`/`std` 属性、`normalize(x)`/`unnormalize(x)`。用 Welford 数值稳定累积。
- [ ] **验证**:小脚本——喂已知分布数据,断言 `mean≈真值`、`std≈真值`、`normalize(x)` 均值≈0。
- [ ] 提交 `feat: running stats normalizer`。

### Task 1.4:`TransitionSource` 接口 + `ReplayBuffer`
**Files:** Create `scout/data/transition_source.py`。
- [ ] 定义接口与默认实现(=ReplayBuffer,内存张量):
```python
class TransitionSource:
    def sample(self, batch_size: int) -> dict: ...      # {S_t, A_t, S_{t+1}}
    def add(self, transitions: dict) -> None: ...        # online 回灌
    def __len__(self) -> int: ...
    def stats(self) -> dict: ...                         # {S_t,A_t,S_{t+1}} 的 RunningStats

class ReplayBuffer(TransitionSource):       # 内存实现
    def __init__(self, state_dim, action_dim, capacity=int(1e6)): ...
    # add: 把 transitions 拼进预分配张量环形写
    # sample: 随机抽 batch,返回 dict(每项 (B, dim))
```
- [ ] **验证**:小脚本——`add` 100 条、`sample(8)` 形状对、`stats()` 不报错。
- [ ] 提交 `feat: TransitionSource + ReplayBuffer`。

### Task 1.5:robomimic low_dim 后端
**Files:** Create `scout/data/robomimic_lowdim.py`;参照 `SOE/src/dataset/robomimic_v2.py`。
- [ ] `RobomimicLowdimSource(TransitionSource)`:读 `low_dim_v141.hdf5`,`obs_keys = sorted(...)` 拼成 `S_t`(state_dim = Σ);action 从 hdf5 读(`action_offset=1` 对齐 `S_{t+1}`);按 `mask/train`/`mask/valid` 过滤;一次性灌进一个 `ReplayBuffer`。
- [ ] 暴露 `state_dim`/`action_dim` 属性(供后续网络构造)。
- [ ] **【image 接口点】** 注释标明:image 后端将来把 `S_t` 换成图像/特征张量,接口不变。
- [ ] **验证**:对 lift hdf5 跑 `RobomimicLowdimSource(path)`,打印 `len`、`state_dim`、`action_dim`、`sample(4)` 形状。
- [ ] 提交 `feat: robomimic low_dim transition backend`。

---

## Phase 2 · base DP 移植(E0)

### Task 2.1:移植 SOE 的 DP 相关文件
**Files:** Copy 进 `scout/policy/`:`dp.py`、`diffusion.py`、`multi_image_obs_encoder.py`、`conditional_unet1d.py`、`crop_randomizer.py`、`common/pytorch_util.py`(对应 SOE `SOE/src/policy/...`)。
- [ ] 逐文件拷贝,统一改 import 前缀(`policy.xxx` → `scout.policy.xxx` 或本包相对 import)。
- [ ] **不要改逻辑**,只改 import。`diffusion.py` 的 `conditional_sample` 保持原样(guidance 注入在 Phase 4 单独加,可继承/重写,不改原方法)。
- [ ] **验证**:`python -c "from scout.policy.dp import DP"` 不报错。
- [ ] 提交 `feat: port SOE base DP (DP + DiffusionUNetPolicy)`。

### Task 2.2:E0 训练 base DP
**Files:** Create `scout/train_base_dp.py`、`configs/base_dp_lift_lowdim.yaml`;参照 `SOE/src/train_single_gpu.py`(精简版,单卡、无 DDP)。
- [ ] config:`policy.name=DP`、`obs_shape_meta`(lift low_dim keys)、`action_dim`、`num_action=20`、`num_epochs`、`lr`、`save_epochs`、core 数据 `low_dim_v141.hdf5`(或 core_20 子集)。
- [ ] `train_base_dp.py`:从 config 构造 `DP`,用 `RobomimicLowdimSource`(或直接 SOE robomimic_v2 加载,二选一,保持一致)喂 `(obs_dict, actions)`,调用 `DP.forward(obs, actions)` 的 `compute_weighted_loss`,AdamW + cosine LR,按 `save_epochs` 存 `ckpt` + loss PNG。
- [ ] **验证**:在 lift 上训到 loss 平台;unguided 动作在 valid 上到 demo 动作均值的 Mahalanobis 距离低(写到 `eval_results/`)。
- [ ] **过线**:loss 收敛 + 动作像 demo(= DP₀)。提交 `feat: E0 base DP training`。

---

## Phase 3 · VIB 模型 + 联合训练(E1)

### Task 3.1:`StateAE`(E_s/D_s)—— 【image 接口点】
**Files:** Create `scout/model/state_ae.py`。
- [ ] 定义可切换接口 + low_dim 实现:
```python
class StateAE(nn.Module):
    """E_s: state->latent; D_s: latent->state. low_dim=MLP, image=CNN(后续)。"""
    @staticmethod
    def from_config(cfg, state_dim, latent_dim=32, hidden_dim=128):
        # cfg.modality == 'low_dim' -> StateMLPAE;  == 'image' -> (stage-2) StateCnnAE
        ...
class StateMLPAE(StateAE):
    def __init__(self, state_dim, latent_dim=32, hidden_dim=128):
        self.encoder = EncoderMLP(state_dim, latent_dim, hidden_dim)
        self.decoder = EncoderMLP(latent_dim, state_dim, hidden_dim)
    def encode(self, s):  return self.encoder(s)          # s̄
    def decode(self, z):  return self.decoder(z)           # Ŝ
```
- [ ] **【image 接口点】** 注释:`StateCnnAE`(stage-2)实现 `encode`/`decode` 同签名;stage-2 在此加 CNN + 决定像素重建策略(`scout_design.md §6`)。
- [ ] **验证**:`StateMLPAE(19,32)` 前后向形状对;小 AE-only 训练几个 step,recon loss 下降。
- [ ] 提交 `feat: StateAE (E_s/D_s) with image swap point`。

### Task 3.2:VIB encoder + decoder
**Files:** Create `scout/model/vib.py`。
- [ ] 实现(均 `EncoderMLP`):
```python
class VIBEncoder(nn.Module):       # (s̄_t, a_t) -> (μ, logvar)
    def __init__(self, latent_dim, action_dim, s_latent_dim=32, hidden_dim=128):
        self.net = EncoderMLP(s_latent_dim + action_dim, 2*latent_dim, hidden_dim)
    def forward(self, s_bar, a):
        mu, logvar = self.net(torch.cat([s_bar, a], -1)).chunk(2, -1)
        return mu, logvar

class VIBDecoder(nn.Module):       # (z, s̄_t) -> ŝ̄_{t+1}
    def __init__(self, s_latent_dim=32, latent_dim=16, hidden_dim=128):
        self.net = EncoderMLP(latent_dim + s_latent_dim, s_latent_dim, hidden_dim)
    def forward(self, z, s_bar):
        return self.net(torch.cat([z, s_bar], -1))
```
- [ ] **验证**:形状对;`reparam(mu,logvar)` 实现 `z = mu + exp(0.5 logvar)*ε`。
- [ ] 提交 `feat: VIB encoder/decoder`。

### Task 3.3:`ScoutVIB` 组合 + 联合 loss
**Files:** Create `scout/model/scout_vib.py`。
- [ ] 组合 `StateAE` + `VIBEncoder` + `VIBDecoder`;`forward(S_t, A_t, S_{t+1})` 返回 loss dict:
```python
# 联合训练(不 detach next-latent target)
s_bar_t = self.ae.encode(S_t)
s_bar_tp1 = self.ae.encode(S_{t+1})          # 非 detach
mu, logvar = self.vib_enc(s_bar_t, A_t)
z = reparam(mu, logvar)
s_bar_pred = self.vib_dec(z, s_bar_t)
ae_loss = MSE(self.ae.decode(s_bar_t), S_t) + MSE(self.ae.decode(s_bar_tp1), S_{t+1})   # 锚
dyn_loss = MSE(s_bar_pred, s_bar_tp1)         # 不 detach
kl = 0.5 * (mu**2 + logvar.exp() - 1 - logvar).sum(-1).mean()
loss = ae_loss + dyn_loss + beta * kl
return {"loss": loss, "ae": ae_loss, "dyn": dyn_loss, "kl": kl, "mu": mu, "logvar": logvar}
```
- [ ] base DP **不在场**(此模块完全不引用 base DP)。单链、单次 `loss.backward()`、无梯度隔离。
- [ ] **验证**:dummy 数据前向 + backward,梯度只到 `ae`/`vib_enc`/`vib_dec`;loss 各项 finite。
- [ ] 提交 `feat: ScoutVIB joint loss`。

### Task 3.4:E1 训练 + β 扫描 + 生死诊断
**Files:** Create `scout/train_vib.py`、`scout/diagnose.py`、`configs/vib_lift_lowdim.yaml`。
- [ ] `diagnose.py`:`sensitivity_ratio(model, batch)` —— `A_t.requires_grad_(True)`;`mu,_ = model.vib_enc(model.ae.encode(S_t), A_t)`;`g = autograd.grad(mu.sum(), A_t)`;`σ_a,σ_μ` 从 `buffer.stats()`;返回 `‖g‖·σ_a/σ_μ`。
- [ ] `train_vib.py`:对 β ∈ {1e-4,1e-3,1e-2,1e-1} 各训一个 `ScoutVIB`(`TransitionSource.sample` 喂 transitions,AdamW,存 ckpt + 逐 loss PNG);每个 ckpt 跑 `sensitivity_ratio`,记录 AE/dyn/KL/μ 的 mean·std。
- [ ] **β 选择**:画 敏感比 vs β;取敏感比 ≥ ~0.3 的最大 β(且 dyn/KL 未爆、μ≈N(0,I))。
- [ ] **验证/过线**:存在合格 β;否则 NO-GO。提交 `feat: E1 VIB training + beta scan + sensitivity`。

---

## Phase 4 · guidance 注入(E2/E3 闸门)

### Task 4.1:SCOUT cost + 归一化桥
**Files:** Create `scout/guidance/cost.py`、`scout/normalizer.py`。
- [ ] `normalizer.py`:DP 动作 normalizer 与 VIB 动作空间之间的桥(`policy_action_normalizer.unnormalize` → VIB `normalize`,参照 LPB `planner.py:211-213`)。
- [ ] `cost.py`:
```python
def scout_cost(x0_hat, s_bar_t, z, vib_enc, action_normalizer_bridge):
    a = action_normalizer_bridge(x0_hat)        # DP 动作 -> VIB 动作空间
    mu = vib_enc(s_bar_t.unsqueeze(...), a)[0]  # 逐 chunk 步;广播 s_bar_t、z
    return ((z - mu)**2).sum(-1).mean()         # mean_t ‖z-μ‖²
```
- [ ] **验证**:dummy `x0_hat`,cost 标量、对 `x0_hat` 可微。
- [ ] 提交 `feat: SCOUT guidance cost + normalizer bridge`。

### Task 4.2:LPB 式 guidance 注入(改 trajectory)
**Files:** Modify/继承 `scout/policy/diffusion.py` 的 `conditional_sample`;参照 LPB `diffusion_unet_hybrid_image_policy.py:244-266`。
- [ ] 新增 `guided_conditional_sample(...)`(不改原 `conditional_sample`),逐字照 LPB 的循环:
  - `trajectory = trajectory.detach().requires_grad_()`;
  - `model_output = model(trajectory, t, local_cond, global_cond)`;
  - 门控:**只留 (a)** `if classifier_guidance and t < guidance_start_timestep:`(**去掉 LPB 的 OOD 门 (b)**);
  - `x0_hat = scheduler.step(model_output, t, trajectory).pred_original_sample`;
  - `loss = scout_cost(x0_hat, s_bar_t, z, vib_enc, bridge)`;
  - `cond = -autograd.grad(loss, trajectory)[0]`;
  - `scale = guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()`;
  - `trajectory = trajectory.detach() + scale * cond`;
  - `trajectory = scheduler.step(model_output, t, trajectory, ...).prev_sample`;
- [ ] `s_bar_t`、`z` 在调用前算好传入(`s_bar_t = E_s(S_t)`,`z ~ N(0,I)` 整段定住)。
- [ ] **验证**:固定 s、z、初始噪声,scale=0 时输出 = 原始 DP 动作;scale>0 时动作改变。
- [ ] 提交 `feat: LPB-style guided conditional sample (SCOUT cost)`。

### Task 4.3:E2 guidance 三判据 + E3 on-manifold
**Files:** Create `scout/eval/guidance_checks.py`(或在 `diagnose.py` 扩展)。
- [ ] **E2**:选定 β 的 VIB + E0 base DP;sweep `guidance_scale`∈{0,1,5,10,20};固定初始噪声 + 一组 z。算:① 多样性(跨 z 动作 std);② 一致性 `‖z-μ(s̄_t,a_guided)‖`;③ Cost 方向(去噪步曲线)。过线:scale=0 ≈0、单调升;一致性降;Cost 降(升则翻符号)。
- [ ] **E3**:最大 scale 的引导 chunk 的 jerk + 到 demo 分布 Mahalanobis vs base DP → 同量级。
- [ ] 提交 `feat: E2/E3 guidance + on-manifold checks`。

---

## Phase 5 · eval / self-improvement 闭环(E4)

### Task 5.1:robomimic rollout 移植
**Files:** Create `scout/eval/rollout.py`;从 SOE `simulation/rollout_utils.py` 移植 robomimic env 执行 + `is_success` 判定。
- [ ] 用 robomimic env 跑一个 policy(可传 `guidance_scale` 走 guided 路径)N 个初始态、try_times 次;返回 rollout(成功与否、动作序列、状态序列)。
- [ ] **验证**:用 E0 base DP(unguided)在 lift 上跑,成功率合理。
- [ ] 提交 `feat: robomimic rollout harness`。

### Task 5.2:metrics
**Files:** Create `scout/eval/metrics.py`。
- [ ] 实现 SOE 四指标:`success_rate_per_round`、`pass_at_k`(try_times=5)、`exploration_yield`、`jerk`(动作三阶差分范数)。定义见 `evaluation_plan.md §一.4`。
- [ ] **验证**:dummy rollout 数据,各指标数值合理。
- [ ] 提交 `feat: SOE metrics`。

### Task 5.3:E4 self-improvement 闭环
**Files:** Create `scout/eval/self_improvement.py`、`configs/eval_lift.yaml`。
- [ ] 编排 5 步 loop(多轮):
  1. `DP₀` 已由 Phase 2 训好;
  2. `ScoutVIB` 已由 Phase 3 训好(选定 β);
  3. 冻结 `DP₀`,`z~N(0,I)` 引导 rollout 探索(Phase 4 路径),robomimic sim;
  4. 筛成功 rollout(`is_success`)→ `RobomimicLowdimSource`/`ReplayBuffer.add`(回灌接口)→ 合 core → 训 `DP₁`(复用 `train_base_dp.py`);
  5. `DP₁ vs DP₀` metric 对比;多轮滚。
- [ ] config:core_20、N=100、try_times=5、6 轮。
- [ ] **验证/过线**:`success_rate/round` 单调升(`DP_new > DP_old`)。
- [ ] 提交 `feat: E4 self-improvement loop`。

---

## Go/No-Go(对照 stage1_plan §五)

E0 过 + E1 找到合格 β + E2/E3 闸门过 + **E4 显示 DP_new > DP_old** ⇒ **GO**(进 stage-2 图像)。
E1 全空间无合格 β ⇒ **NO-GO**,重审 idea。

---

## 自审(写完后过一遍)

- **spec 覆盖**:`scout_design.md §1–§6` → Phase 1(data/§3 接口)、Phase 2(base DP/§2)、Phase 3(VIB 联合训练/§3)、Phase 4(guidance/§4)、Phase 5(eval loop/§5)。§6 图像 = 各处「image 接口点」注释,不另起 phase(stage-2)。✓
- **占位符**:无 TBD;SOE/LPB 移植项均给了确切源文件 + 行号 + 改动点。
- **类型一致**:`TransitionSource.sample→{S_t,A_t,S_{t+1}}`、`StateAE.encode/decode`、`VIBEncoder(s̄,a)→(μ,logvar)`、`scout_cost(x0_hat,s̄_t,z,...)→scalar` 各任务签名一致。✓
- **image 接口**:StateAE、TransitionSource 两处明确标注;VIB/guidance/cost 模态无关。✓
