# SCOUT 实验日志

> 记录 SCOUT 各阶段实验的设置、数据、结果。新阶段续写新 Part。

---

## Part 1 — Base DP (E0) 训练

**目标**:为 SCOUT 训练冻结的 base Diffusion Policy(robomimic lift / can / square,image)。这是后续 E1(VIB dynamics)/ E4(multi-round loop)的起点。
**日期**:2026-08-11 · **服务器**:106.14.2.243(8× H20)· **代码**:branch `impl/scout-stage1`,LPB `TrainDiffusionUnetHybridWorkspace`。

### 1.1 输入数据格式

数据来自 robomimic **v0.1 PH**(`low_dim_v141.hdf5`,200 demos),按 SOE 方式处理(详见 `memory/scout-data-pipeline`):下 low_dim → 从 state 重渲染图像 → DP 转换加 abs_actions → 提 core 子集。

**观测(policy 输入)**

| 类型 | key | 维度 | 说明 |
|---|---|---|---|
| rgb | `agentview_image` | (3, 84, 84) uint8 | 第三人称视角 |
| rgb | `robot0_eye_in_hand_image` | (3, 84, 84) uint8 | 手眼视角 |
| low_dim | `robot0_eef_pos` | (3,) | 末端位置 |
| low_dim | `robot0_eef_quat` | (4,) | 末端姿态(quaternion) |
| low_dim | `robot0_gripper_qpos` | (2,) | 夹爪关节角 |

→ 2 路 84×84 RGB + 9 维 low_dim。

**动作**

- 表征:**abs_6drot,10 维** = 3 平移 + 6 rotation_6d + 1 夹爪。
- hdf5 里存 `abs_actions`(7 维:3 pos + 3 axis_angle + 1 gripper);`RobomimicReplayImageDataset(abs_action=true, rotation_rep=rotation_6d)` 加载时**在线**把 3 维 axis_angle 转成 6d → 10 维。
- 原始 `actions`(7 维 delta rel)保留但训练不用。

**序列**:`horizon=16`,`n_obs_steps=2`,`n_action_steps=8`,`n_latency_steps=0`,`pad_before=1`,`pad_after=7`,`obs_as_global_cond=true`。

**hdf5 结构**(`image_v141_abs_coreN.hdf5`):`data/demo_{i}/{actions(7 rel), abs_actions(7 abs axis_angle), states, obs/...}`;`data` attrs 含 `env_args`(env_name + controller_configs);`mask/`(filter keys,LPB loader 不用,已物化成 core-only 文件)。

**core 子集(SOE 约定 `ind % interval == 0`,LPB loader 加载全部 demo 故物化)**

| task | 子集 | demos | 总步数 | 文件 |
|---|---|---|---|---|
| lift | core_10 (interval 20) | 10 | 486 | `lift/ph/image_v141_abs_core10.hdf5` |
| can | core_20 (interval 10) | 20 | 2301 | `can/ph/image_v141_abs_core20.hdf5` |
| square | core_20 (interval 10) | 20 | 3047 | `square/ph/image_v141_abs_core20.hdf5` |

### 1.2 三个 task 的训练超参

**超参来源**:LPB 论文 Appendix B(Table 5/6:square=600 epoch)+ LPB 官方 ckpt 配置(rollout/checkpoint=20);horizon/n_action/batch/lr 等与 LPB square 一致。**数据量(core)和图片(84×84)按 SOE/DP 标准,未跟 LPB 的 40 demos / 140×140**(见 §1.4)。

**共用超参**

| 项 | 值 |
|---|---|
| policy | `DiffusionUnetHybridImagePolicy` |
| diffusion | DDPM,100 train/inference 步,beta [1e-4, 0.02] squaredcos_cap_v2,predict epsilon,clip_sample |
| UNet | down_dims [512,1024,2048],step_embed 128,kernel 5,n_groups 8,cond_predict_scale,obs_encoder GroupNorm |
| crop | 76×76(`eval_fixed_crop=true`) |
| horizon / n_obs / n_action | 16 / 2 / 8 |
| batch / num_workers | 64 / **0**(torch shm 限制,见 memory) |
| optimizer | AdamW lr 1e-4,betas [0.95,0.999],wd 1e-6 |
| lr_scheduler | cosine,warmup 500 步 |
| ema | inv_gamma 1,power 0.75,max 0.9999 |
| **num_epochs** | **600**(LPB square;lift/can 按 square 推断) |
| rollout_every / checkpoint_every | **20 / 20**(LPB 官方) |
| sample_every / val_every | 5 / 1 |
| seed | 42(单 seed) |
| normalizer | abs_action 用 `robomimic_abs_action_only_normalizer_from_stat`;图像 [0,1] |
| eval pool | n_train=6, n_test=50, n_envs=28, test_start_seed=100000 |

**逐 task 差异**

| task | dataset(core) | max_steps | wandb run | GPU |
|---|---|---|---|---|
| lift | core_10(10 demos) | 400 | SCOUT-baseDP-lift | 0 |
| can | core_20(20 demos) | 300 | SCOUT-baseDP-can | 1 |
| square | core_20(20 demos) | 500 | SCOUT-baseDP-square | 2 |

checkpoint:`save_last_ckpt=true`(每 20 epoch 存 `epoch=N.ckpt`,共 ~30 个/task;topk 代码在本 workspace 注释掉了,不自动清理)。

### 1.3 运行信息

- wandb project:`scout-base-dp`(entity `jiachunbao-sjtu`),3 个 run:SCOUT-baseDP-{lift,can,square}。
- 启动:`soe_scripts/launch_trainings.sh`(3 tmux 会话 `train_{lift,can,square}`,各 `CUDA_VISIBLE_DEVICES=0/1/2`,隔离 wandb key)。
- 配置:`scout/configs/base_dp_{lift,can,square}_image.yaml`。
- 估计时长:~30 min/task(600 epoch + 30 次 eval,并行)。

### 1.4 与 LPB 的偏差(有意为之)

| 项 | 本实验 | LPB Appendix B | 原因 |
|---|---|---|---|
| #demos | core_10/20(5–10%) | 40(20%) | SCOUT 沿用 SOE 的**少样本**设定(自我改进的卖点) |
| 图像 / crop | 84×84 / 76 | 140×140 / 128 | 沿用 SOE/DP 标准,避免重渲染 |
| seed | 42(单) | — | 先出第一版;多 seed 后补 |

base DP 训完后,选成功率最高的 checkpoint 作为 E1(VIB dynamics)的冻结 base。

---

## Part 2 — 单轮自我改进 (SOE protocol) 完整管线 + 指标

**目标**:把 SCOUT 单轮自我改进(exploration → 回灌 → retrain)的完整流程记录到「任何人拿到这份 log 就能复刻实验、或照着跑一个新的 robomimic 任务」的颗粒度。
**日期**:2026-08-11 ~ 2026-08-12 · **代码**:branch `impl/scout-stage1` · **服务器**:106.14.2.243(`.venv`:py3.10 / torch2.4.1+cu121)。

### 2.1 管线总览 (5 个 stage)

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  Stage A 数据准备        Stage B base DP(E0)      Stage C VIB dynamics(E1)   │
│  low_dim.hdf5            train.py base_dp_*      scout.train_vib            │
│   ├ download ─────────►  ├ 600 epoch DP₀         ├ β=1e-3, 300 epoch        │
│   ├ re-render 84² img ─► │  (frozen base)         ├ encoder p̄(z|S,A)         │
│   ├ +abs_actions key ──► └ DP₀ ckpt(580.ckpt)     └ scout_vib.ckpt          │
│   └ core_N subset                                  │                         │
│                                                    ▼                         │
│  Stage D 探索 rollout + classifier guidance    Stage E 合并 + retrain(E2)   │
│  scout.eval.run_round                          同一个 run_round 自动接:     │
│   ├ 100 random init × (1 baseline              ├ augmented.hdf5            │
│   │   + ≤5 guided retry)                       │   = core ∪ 成功 rollout     │
│   ├ z-per-trajectory(N(0,1),锁)                 └ retrain DP 600 epoch       │
│   ├ ∇_a cost 引导(frozen DP₀)                     → DP₁ checkpoints/        │
│   └ 4 指标(success_rate/Pass@5/yield/jerk)         (wandb SCOUT-baseDP-*-exp1)│
└──────────────────────────────────────────────────────────────────────────────┘
```

> Stage D 与 E 由 `scout/eval/run_round.py` 一次驱动(`SelfImprovementLoop.run(num_rounds=1)` + 手调 retrain_fn)。`run_round` 在探索收集到 **0** 条成功时会 **abort**(guard),不进入 retrain。

### 2.2 Stage A:数据准备(下载 → 重渲染 → abs_actions → core)

环境激活:`source /root/workspace/baojiachun/.venv/bin/activate`(或直接 `.venv/bin/python`)。cwd = `/root/workspace/baojiachun`(repo 根在 `scout/`,数据在 `data/robomimic/<task>/ph/`)。

```bash
cd /root/workspace/baojiachun/data/robomimic/<task>/ph

# A0. 下载 robomimic v0.1 PH low_dim(200 demos,只有 state,无图像)
curl -C - --retry 3 -o low_dim_v141.hdf5 \
  http://downloads.cs.stanford.edu/downloads/rt_benchmark/<task>/ph/low_dim_v141.hdf5

# A1. 图像复原:从 state 重渲染 84×84 双摄/多摄(MUJOCO_GL=egl,egl_probe stub 已就位)
MUJOCO_GL=egl python -m robomimic.scripts.dataset_states_to_obs \
  --done_mode 0 --dataset low_dim_v141.hdf5 --output_name image_v141.hdf5 \
  --camera_names <cams> --camera_height 84 --camera_width 84

# A2. DP 转换:加 abs_actions 键(7 维 aa 单臂 / 14 维 aa 双臂),保留原 actions(rel)
python -m diffusion_policy.scripts.robomimic_dataset_conversion \
  -i image_v141.hdf5 -o image_v141_abs.hdf5 -n 4

# A3. core 子集:SOE 约定 ind % interval == 0(LPB loader 加载全部 demo,故物化成 core-only 文件)
python experiments/scripts/extract_core_subset.py \
  image_v141_abs.hdf5  image_v141_abs_core<N>.hdf5  <interval>
```

**逐 task 参数表**(interval 决定 core 大小;abs_actions 维度 = 单臂 7 / 双臂 14):

| task | cameras | interval | core | abs_actions | max_steps |
|---|---|---|---|---|---|
| lift | `agentview robot0_eye_in_hand` | 20 | core_10 | 7 | 400 |
| can | `agentview robot0_eye_in_hand` | 10 | core_20 | 7 | 300 |
| square | `agentview robot0_eye_in_hand` | 10 | core_20 | 7 | 500 |
| transport | `shouldercamera0 robot0_eye_in_hand shouldercamera1 robot1_eye_in_hand` | 10 | core_20 | 14 | 700 |

产物:`data/robomimic/<task>/ph/image_v141_abs_core<N>.hdf5`(`data/demo_i/{actions(7/14 rel), abs_actions(7/14 abs aa), states, obs/...}`,HWC uint8 图像;`data` attrs 含 `env_args`)。

### 2.3 Stage B:Base DP 训练 (E0)

超参全表见 Part 1(§1.2)。命令(每个 task 一张 GPU,wandb project `scout-base-dp`):

```bash
cd /root/workspace/baojiachun/scout
CUDA_VISIBLE_DEVICES=<g> .venv/bin/python train.py \
  --config-path configs --config-name base_dp_<task>_image \
  task.dataset_path=$(pwd)/data/robomimic/<task>/ph/image_v141_abs_core<N>.hdf5 \
  logging.name=SCOUT-baseDP-<task> logging.project=scout-base-dp \
  training.num_epochs=600
```

产物:`scout/data/outputs/<ts>_base_dp_<task>_image_<task>/checkpoints/{<epoch>.ckpt, latest.ckpt}`。选 `test/mean_score` 最高的 ckpt 作 DP₀(SCOUT 用 last `580.ckpt`)。

### 2.4 Stage C:VIB dynamics 训练 (E1)

```bash
cd /root/workspace/baojiachun/scout
.venv/bin/python -m scout.train_vib --config configs/vib_<task>_image.yaml
```

`configs/vib_<task>_image.yaml` 关键键:`base_dp_ckpt=<DP₀>`、`beta=1e-3`(= kl_weight,**最关键旋钮**)、`frameskip=8`(action chunk=80 维=8×10,**encoder 是 chunk-encoder 不是 per-step**)、`style_dim=16`、`hidden_dim=128`、`num_epochs=300`。
产物:`data/outputs/vib_<task>/<ts>/scout_vib.ckpt`(wandb project `scout-dynamics`)。预期:latent_mse≈val_mse(非过拟合)、kl→0、|μ|→~0(z 压向 N(0,I) 先验,**是 VIB 预期目标,非塌缩 bug**)。

### 2.5 Stage D+E:探索 + 合并 + retrain(一次 `run_round`)

```bash
cd /root/workspace/baojiachun/scout
CUDA_VISIBLE_DEVICES=<g> .venv/bin/python -m scout.eval.run_round \
  --config configs/eval_<task>.yaml --task <task> \
  --base-dp-ckpt <…/base_dp_<task>…/checkpoints/580.ckpt> \
  --vib-ckpt      <…/vib_<task>/<ts>/scout_vib.ckpt> \
  --core-hdf5     <…/image_v141_abs_core<N>.hdf5> \
  --num-epochs 600 --cuda-visible-devices <g>
```

`run_round` 内部做的(SOE 单轮协议):
1. **collect init states**:100 个随机初始状态(跨 round 固定)。
2. **baseline**:每 init 跑 1 次 frozen DP₀(不开 guidance)→ `success_rate`。
3. **exploration**:对 baseline 失败的 init,最多 `try_times=5` 次 **guided** retry。每条 rollout **采样 1 个 z~N(0,1) 并跨 chunk 锁定**(planner.set_z,异于 SOE 的 per-chunk);guidance = DDPM 去噪循环内对 x̂₀ 加 `guidance_scale·∇_a[−cost]`,`cost = ‖z − μ(s̄_t, a_chunk)‖²`。
4. **指标**:`success_rate` / `Pass@5` / `exploration_yield` / `jerk`(`scout/eval/metrics.py`)。
5. **合并**:全部成功 rollout 写进 `augmented.hdf5`(core ∩ rollout 的 obs keys;HWC uint8 图;7/14 维 aa abs_actions)。
6. **retrain**:`train.py` 在 augmented 上训 600 epoch,wandb `SCOUT-baseDP-<task>-exp1`(project `scout-base-dp`)。

### 2.6 SOE 指标定义(`scout/eval/metrics.py`)

| 指标 | 定义 |
|---|---|
| `success_rate` | baseline 在 100 个 init 上各 1 次的成功率 = `baseline_solved / 100` |
| `pass_at_k` (k=5) | 「baseline 成功」或「baseline 失败但在 ≤5 次 guided retry 内成功」的 init 占比。baseline 成功算 0 次尝试 |
| `exploration_yield` | 本轮成功 guided rollout 总数(first-success stop,每个 init 0 或 1)→ 这些被回灌 |
| `jerk` | `mean_t ‖a[t+3] − 3a[t+2] + 3a[t+1] − a[t]‖₂`,只在**成功** rollout 上算(baseline vs exploration 两组) |

### 2.7 lift / can / square 结果(单轮)

| task | baseline | success_rate | Pass@5 | yield | jerk_base | jerk_expl | retrain 结果 |
|---|---|---|---|---|---|---|---|
| lift | 100/100 | **1.00** | 1.00 | 0 | 0.329 | — | **ABORTED**(0 成功,base 已饱和,guidance 无可恢复) |
| can | 67/100 | **0.67** | **0.74** | **7** | 0.210 | 0.283 | **DONE**(ep599,train_loss 0.0017)→ `scout_round_can/round_0/checkpoints/latest.ckpt` |
| square | 37/100 | **0.37** | **0.46** | **9** | 0.199 | 0.129 | **DONE**(ep599,train_loss 0.0024)→ `scout_round_square/round_0/checkpoints/` |

> square 的 success_rate/Pass@5/jerk 最初卡在父进程 stdout buffer,等 retrain 跑完、`run_round` flush 后拿到(见 §2.11)。can 的完整 metrics 训练一结束就落盘。lift 0 成功触发了 `run_round` 的 guard,符合预期(lift 在 core_10 上已达 1.0,没有 exploration 余地)。
> **can:Pass@5 0.74 > success_rate 0.67 → guidance 在 7 个 baseline 失败的 init 上恢复了成功;exploration 有效。** 这是 SCOUNT stage-1 的关键正面信号。

### 2.8 跑一个新的 robomimic 任务 — 通用清单

1. **看数据**:`.venv/bin/python -c "import h5py; …"` 打开 `low_dim_v141.hdf5`,读 `data/demo_0/obs` keys、`actions` shape、`data.attrs['env_args']`(env_name、controller `control_delta`、camera_names)、`model_file` 是否非空。
2. **选相机**:从 `model_file` XML 里 `<camera name=…>` 挑(或参照 `diffusion_policy/config/task/<task>_image_abs.yaml` 的 `shape_meta`)。改 A1 的 `--camera_names`。
3. **Stage A**:跑 A0–A3(改 `<task>`、cameras、interval)。验证:`RobomimicReplayImageDataset` 能加载,`abs_action` 维度对。
4. **Stage B config**:复制 `configs/base_dp_square_image.yaml` → `base_dp_<task>_image.yaml`,改 `task_name`、`shape_meta`(cameras + proprio keys + action shape = 单臂[10]/双臂[20])、`max_steps`、`dataset_path`、`render_obs_key`。
5. **Stage B train**:照 §2.3 启动。
6. **Stage C config**:`configs/vib_<task>_image.yaml`(改 `base_dp_ckpt`、相机名、`proprio_dim`)→ 照 §2.4。
7. **Stage D config**:`configs/eval_<task>.yaml`(改 camera/proprio/horizon/core_filter_key)→ 照 §2.5。

### 2.9 已知坑(都已修,这里列全方便复刻)

**robomimic 源码 patch(服务器上,未入 git;每个新环境都要重打)**:
- `egl_probe.py` 写 stub `get_available_devices()->[0]`(robomimic 只用它挑 render gpu)。
- `diffusion_policy/common/robomimic_util.py` 的 model-None bug:Lift/Can 无自定义 model → `get_robomimic_model_file` 返 None → `edit_model_xml(None)` 崩。补丁:仅 `if model_file is not None` 才传 model。
- **(新,2026-08-12)** `robomimic/scripts/dataset_states_to_obs.py:get_camera_info` 硬编码 `assert cam_name.startswith("robot0")` + `robots[0]`,双臂 transport 的 `robot1_eye_in_hand` 会崩。补丁:从 cam 名前缀解析 `robot_idx = int(cam_name[len("robot")])`,`robots[robot_idx]`(原文件备份在 `.orig.bak`)。

**数据/环境坑**:`h5py` + `imagecodecs` 要 `uv pip install`;`gym==0.21.0`(LPB eval 是 0.21 风格);`num_workers=0`(torch shm 限制);`MUJOCO_GL=egl`;`diffusion_policy/env/` 从本地拷到服务器。
**SCOUT 集成坑(9 个,a991048..3a07131)**:VIB encoder 吃 chunk 不吃 per-step(最关键);DP factory 要建 ScoutPolicy 不是父类;ckpt 键是 `state_dicts["model"]`;env obs 要手 stack n_obs_steps;abs_action 10维6d ↔ 7维aa 转换;`control_delta=False`;normalizer 用 `params_dict` 不要 `in`;exploration rollout 要 `record_obs=True`;augmented hdf5 必须写 `abs_actions`(loader 读它训练)。详见 memory `scout-step1-progress`。

### 2.10 服务器环境一览
venv `/root/workspace/baojiachun/.venv`(uv,py3.10 / torch2.4.1+cu121);robomimic 源码 `dependencies/robomimic/`(@9273f9c);diffusion_policy 是 repo 内 vendored(`scout/diffusion_policy/`);wandb key 走 `baojiachun/.secrets/wandb.env`(`WANDB_API_KEY`,entity `jiachunbao-sjtu`)。GPU 8×H20,用前先 `nvidia-smi` 查占用。完整复刻见 README「环境配置(服务器 · uv)」。

### 2.11 square 完整 metrics(2026-08-12 补)

square retrain 跑完后 metrics 已 flush:`baseline 37/100`,`success_rate=0.37`,`pass_at_k=0.46`,`yield=9`,`jerk_baseline=0.1985`,`jerk_exploration=0.1288`,retrain 到 ep599(train_loss 0.0024)。

**小结(三任务单轮)**:
- **两个 meaningful 任务(can/square)guidance 都正向**:Pass@5 > success_rate(can +0.07,square +0.09),exploration 在 baseline 失败的 init 上各恢复 7 / 9 条成功。
- **lift abort**:core_10 上 base 已 1.0,没有 exploration 余地(符合预期;不是 bug)。
- square 的 `jerk_exploration 0.129 < jerk_baseline 0.199` —— guided 成功 rollout 反而比 baseline 更平滑(can 则相反 0.283 > 0.210);两个 task 的 jerk 方向不一致,jerk 不是 stage-1 的主指标(Pass@5/success_rate 才是)。
- **stage-1 go-signal**:can/square 都满足「exploration yield > 0 且 Pass@5 > success_rate」→ classifier guidance 在 frozen base DP 上能恢复成功,SCOUT stage-1 成立。Step 4(baseDP vs exploreDP 同环境对比 success rate)是下一步。

### 2.12 Step 4 — baseDP (DP₀) vs DP-exp1 对比(纯 DP success rate,SOE 对齐)

**方法(对齐 SOE `plot_multi_round_average`)**:success rate 用**纯 DP**(不开 exploration / 不用 VIB),同一批 100 个 init state(seed 固定 → DP₀ 与 DP-exp1 见到相同的 100 个 init)。SOE 每次 eval 的前 100 条 rollout(index 0–99)就是纯 DP 段,success_rate = 这 100 条的成功率;exploration 段(index ≥100)只用于 Pass@5 和回灌。所以这里只比纯 DP success rate(策略本身的进步),不比 Pass@5(DP-exp1 的 Pass@5 需要为它单独训 VIB,见 §2.12 末)。

**实现**:`scout.eval.run_round --try-times 0`(`evaluate_exploration` 在 try_times=0 时 retry 循环不执行 → yield=0 → run_round 的 "0 successful" guard 在 retrain 前 abort)。这样只跑 baseline 段(DP-exp1 当纯 DP 在 100 个 init 上各 1 次),拿到 success_rate + jerk_baseline,VIB 加载但不使用。两个 ckpt 都取 **580.ckpt**(SCOUT 约定用最高 epoch ckpt;`latest.ckpt` 不存在,与 DP₀ 一致),epoch 对齐 → 公平。

**结果**:

| task | DP₀ success_rate | **DP-exp1 success_rate** | Δ | DP₀ jerk | DP-exp1 jerk | 回灌 rollout 数 |
|---|---|---|---|---|---|---|
| can | 0.67 | **0.75** | **+0.08 ↑** | 0.210 | 0.210 | 7 |
| square | 0.37 | **0.34** | **−0.03 ↓** | 0.199 | 0.188 | 9 |

> 命令:`CUDA_VISIBLE_DEVICES=<g> python -m scout.eval.run_round --config configs/eval_<task>.yaml --task <task> --base-dp-ckpt <round_0/checkpoints/580.ckpt> --vib-ckpt <vib> --core-hdf5 <core> --try-times 0 --log-root data/outputs/eval_dpexp1_<task>`。日志 `data/logs/eval_dpexp1_<task>.log`。

**解读**:
- **can:self-improvement 起作用**。回灌 7 条 exploration 成功轨迹后,纯 DP 成功率 0.67 → 0.75(+12% 相对),jerk 不变。这是 SCOUNT 单轮自我改进的正结果。
- **square:没起作用(甚至略降)**。0.37 → 0.34,在噪声范围内,但绝不是提升。square 更难、DP₀ 成功率本就低(0.37),9 条 rollout 的回灌量相对 core_20(3047 步)太少,不足以把策略推向更好的解;且这 9 条 guided rollout 的质量(在低成功率任务上)可能不够。
- **jerk**:can 持平;square 略降(0.199→0.188,DP-exp1 的成功轨迹稍平滑),但 jerk 不是主指标。
- **结论**:单轮、单 seed 下,**SCOUNT 自我改进在 can 上有效、在 square 上无效**。要下定论需多 seed + 多 round;square 的瓶颈可能是回灌数据量/质量,而非 guidance 本身(guidance 在 square 上确实恢复了 9 个 baseline 失败的 init,见 §2.7/§2.11)。

**Pass@5 for DP-exp1(未做)**:VIB 的 `E_s` 只存了 `proprio_embed`,图像 ResNet encoder 是冻结借用 base DP 的(VIB ckpt 仅 14 keys),所以 DP₀ 训的 VIB 与 DP-exp1(重训过、encoder 不同)不匹配 —— 在 DP-exp1 上跑 guided Pass@5 需要先为 DP-exp1 单独训一个 VIB(同 §2.4,~300 epoch)。本轮只比纯 DP success rate(SOE 主指标)。
