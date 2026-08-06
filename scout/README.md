# SCOUT — low_dim 可行性验证(g o/no-go)

本目录是 classifier-guided exploration idea 的**最小可行性验证**(对标 `idea/long_term_plan.md`
阶段 1 的硬里程碑):在 robomimic **low_dim** 单任务上,证明 classifier guidance 能把冻结的
base Diffusion Policy 的动作推向不同 skill、产生有意义的探索。

代码风格与学长 SOE 对齐,复用 `SOE/src` 的基类(`EncoderMLP`、`DiffusionUNetPolicy`、
`MultiImageObsEncoder`、robomimic hdf5 工具),通过 `sys.path` 引入。`SOE/` 被本仓库
`.gitignore` 排除,故 SCOUT 代码独立放在 `scout/`(进版本库)。

## 目录结构

```
scout/
├── README.md                       # 本文件(runbook)
├── src/
│   ├── transition_dataset.py       # (S_t, A_t, S_{t+1}) 转移数据集
│   ├── scout_policy.py             # VIB 编码器 + 动力学解码器 + VIB loss
│   ├── make_configs.py             # 从 hdf5 自动生成 base-DP / SCOUT config
│   ├── train_scout.py              # SCOUT 训练入口(对齐 SOE train_single_gpu)
│   ├── guided_sampler.py           # classifier-guided 去噪采样(测试期核心)
│   └── validate_scout.py           # go/no-go 验证(多样性 / 一致性 / Cost 曲线)
├── configs/                        # make_configs 生成的 json 落这里
└── out/                            # 训练 / 验证输出(被 .gitignore 忽略)
```

## 方法速览(详见仓库根 README 与 `idea/idea_notes.md`)

- **训练**:base DP 冻结、不在场。编码器 `p̄_θ(z|S_t,A_t)→(μ,logvar)`、动力学解码器
  `q_φ(S_{t+1}|S_t,z)→Ŝ_{t+1}`(第一版**确定性回归**)。
  `loss = next-state MSE + β·KL[p̄‖N(0,I)]`,单次 backward,无需梯度隔离。
- **测试**:`z~N(0,I)` 定住,在 base DP 去噪每步注入 `model_output += −guidance_scale·∇_a Cost`,
  `Cost(a,z|s)=‖z−μ(s,a)‖₂`。

## Runbook

> 前置:已按 `SOE/README.md` 装好 SOE 环境(conda py3.8 + `SOE/requirements.txt` +
> pytorch3d + robomimic)。以下命令在仓库根目录执行(`cd "D:\博士\classifier-guided exploration"`)。

### 1. 下载 low_dim 数据(磁盘上目前没有)

```bash
cd SOE/simulation && python download_datasets.py \
    --tasks sim --dataset_types ph --hdf5_types low_dim --download_dir datasets
# 得到 SOE/simulation/datasets/{lift,can,square,transport}/ph/low_dim_v141.hdf5
```

(可选)确认观测维度(填 config 用,本流程已自动从 hdf5 读,无需手填):
```bash
cd SOE/simulation && python get_dataset_info.py --dataset datasets/lift/ph/low_dim_v141.hdf5
```

### 2. 生成 config(自动从 hdf5 读 obs 维度,免猜)

```bash
python scout/src/make_configs.py \
    --dataset SOE/simulation/datasets/lift/ph/low_dim_v141.hdf5 \
    --task lift --out_dir scout/configs
# => scout/configs/dp_lift_lowdim.json   (base DP)
# => scout/configs/scout_lift.json       (SCOUT VIB)
```

### 3. 训练冻结的 base DP(low_dim)

```bash
cd SOE/src && python train_single_gpu.py --config <仓库根的绝对路径>/scout/configs/dp_lift_lowdim.json
# checkpoint 落在 scout/out/dp_lift_lowdim/<时间戳>/ckpt/policy_last.ckpt
```

### 4. 训练 SCOUT VIB 动力学模型

```bash
python scout/src/train_scout.py --config <仓库根的绝对路径>/scout/configs/scout_lift.json
# checkpoint 落在 scout/out/scout_lift/<时间戳>/ckpt/policy_last.ckpt
```
> 关键旋钮 **β = `kl_weight`**(make-or-break):可在 `scout_lift.json` 里改,或用
> `make_configs.py --kl_weight` 扫 `1e-4 … 1e-1`。β 太大→z 与动作脱钩(探索失败),
> 太小→退化成普通动力学模型。

### 5. go/no-go 验证

```bash
python scout/src/validate_scout.py \
    --dp_config   scout/configs/dp_lift_lowdim.json \
    --dp_ckpt     scout/out/dp_lift_lowdim/<时间戳>/ckpt/policy_last.ckpt \
    --scout_config scout/configs/scout_lift.json \
    --scout_ckpt  scout/out/scout_lift/<时间戳>/ckpt/policy_last.ckpt \
    --guidance_scales "0,1,5,10,20" \
    --out_dir     scout/out/validate
```

输出(`scout/out/validate/`):`metrics.json` + 4 张图。

### 6. 判读(go/no-go)

三条同时满足 => **GO**,值得往下做到图像 / multi-round self-improvement:

| 判据 | 图 / 指标 | 期望 |
|---|---|---|
| 多样性 | `diversity_vs_scale.png` | 随 `guidance_scale` 上升;`scale=0` 时≈0 |
| 一致性 | `consistency_vs_scale.png` | 随 `scale` 下降(动作编码回去≈z) |
| Cost 方向 | `cost_over_steps.png` | 随去噪步下降;**若上升 → `guidance_scale` 取负号** |

## 与 SOE 的关系 / 后续

- **本目录 = 阶段 1** 的 low_dim toy。跑通后,阶段 2(对标 `SOE/src/`)会把 SCOUT 策略类
  接进 SOE 的 `train_single_gpu.py` / `diffusion.py` 注册与去噪循环,做图像观测、
  multi-round self-improvement。
- 第一版解码器是**确定性回归**;后续可换成扩散式 next-state 去噪(更接近最终图像设定)。
- 动作对齐 SOE 约定:`action_offset=1`(`actions[t+1]` 驱动 `obs[t]→obs[t+1]`)。
