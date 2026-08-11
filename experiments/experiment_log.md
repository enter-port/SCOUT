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
