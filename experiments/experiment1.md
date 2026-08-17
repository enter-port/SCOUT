# Experiment 1 归档记录(2026-08-14 → 2026-08-17)

本目录归档了 SCOUT 第一阶段自提升实验的全部数据(can / square / lift / logs)。
该阶段的完整结论与当前(修复后)配置如下。

## 1. 阶段结论

1. **VIB 死区事故(最重要)**:E_s 的 post-ReLU 特征(非负、大均值,‖s̄‖≈33)落在
   VIB 编码器第一层随机初始化权重的 ReLU 全死区(实测 0/128 单元存活)→ 编码器为
   常数函数(KL≈3e-6 nats,μ 对 s̄/a 的敏感度精确为 0)→ **guidance 梯度精确为 0,
   所有"SCOUT"实验实际 ≡ DP 链 + RNG 重排**。本阶段全部 SCOUT-vs-DP 差异均为
   噪声 + 回灌反馈环放大,不构成 guidance 有效性证据。
2. 修复后(LayerNorm 输入 + free-bits + β=1e-4 + failure_weight=5)在 can-exp6
   数据上验证:KL 稳定 0.67-0.72(修复前 epoch 13 即 <0.01)、latent_mse 0.0047-49 +
   val 0.0020(优于死模型平台 0.0078)、guidance |dNLL/da|≈0.13-0.15(修复前 0)。
3. 管线级基线数据(修复前,固定 seed 42)仍有效:can 六轮 DP/SCOUT 基线
   0.64→0.71/0.64、square .38→.45/.41;纯 DP 链"探索产量上升但基线不涨"的现象记录在案。

## 2. 当前 Setting(新实验默认)

### 评估 / rollout 协议(SOE 对齐)
- **固定 seed 42**(init 场景 42..141 跨轮不变,逐轮成功率直接可比);
  n_init=100,try_times=5(失败 init 最多重试 5 次);horizon can=300 / square=500。
- step2 = 当前链最新 DP 的首试基线;step3 = 仅失败 init 重试;
  SCOUT 链 guided(VIB NLL guidance),DP 链纯重试。z 每条 rollout 采样一次、整段定住。
- 指标:success_rate / pass@5 / exploration_rescued / 轨迹产量 / avg_jerk(SOE 三阶差分)。

### 数据规则(跨轮累积,2026-08-15 用户定)
- **DP retrain**:`success_accum.hdf5` = core + 第 1..N 轮全部探索成功轨迹(跨轮累积);
  当轮 0 救回时用同一份累积数据重新训练(反死锁);链上无 DP 的轮次向前回溯续链。
- **DYN retrain**:`all_accum.hdf5` = core + 第 1..N 轮全部轨迹(含失败);每轮 [3/3] 重建。
- **DP 链(a=DP)不训 dyn**(baseline 永不加载 VIB);SCOUT 链 dyn 的 E_s = 当轮新 DP。

### VIB / dyn 配置(2026-08-17 修复后,λ=3/5/7 扫描选定)
- VIBEncoder 输入 **LayerNorm**(防 post-ReLU 特征死区);
- **free_bits = 0.05**(逐维 KL 地板,β 无法把 z 挤干);**β = 1e-4**;
- **failure_weight = 5**(非成功轨迹重建损失 ×5;val 恒不加权);
- style_dim=16, hidden=128, frameskip=8(chunk=80),batch 256;
- **steps_per_epoch=100**,num_epochs 300(dyn-base/core 测试用 120);
- lr:AdamW 1e-3 + warmup 5 epoch + cosine 衰减至 5% 峰值;
- 哨兵:relu_alive<0.01 中止、KL<0.01(@ep10)中止、训后真实 batch |dNLL/da|>0 检查;
  liveness 每 epoch 打点。

### DP retrain 配置
- 600 epochs,train_filter_key=scout_aug(core+累积成功全选),workers=8(TMPDIR=/tmp 已修),
  cudnn_benchmark=true,ckpt_every=200(199/399/599),rollout_every=0(无中期 eval)。

### wandb(2026-08-17 起)
- 命名 task 在最前:`{task}-{DP|SCOUT}-rollout-exp{N}`、`{task}-DP-{a}-exp{N}`、
  `{task}-dyn-{a}-exp{N}`、`{task}-dyn-base`、`{task}-eval`。
- 六个项目:`1-can-DP`、`1-can-dyn`、`1-can-eval`、`1-square-DP`、`1-square-dyn`、`1-square-eval`。

### 基础设施要点
- rollout 提速:adapter `get_state()` 绕过 robomimic 的模型 XML 序列化(4.4h→~1h/_square);
- TMPDIR 必须 `/tmp`(DSW 容器继承的 TMPDIR 指向 CPFS,AF_UNIX bind 即 EOPNOTSUPP,
  torch_shm_manager 必崩 → DataLoader workers>0 不可用,已修进 round.sh);
- 训练瓶颈=kernel-launch-bound(精度无关);cudnn_benchmark +7%;compile 不可用(无 triton)。

## 3. 归档位置

- 本文件:`data/experiment1/experiment.md`(repo 副本 `experiments/experiment1.md`);
- 数据:`data/experiment1/{can,square,lift,logs}`(含 DP-base/dyn-base ckpt、exp6 rollout);
- 新实验沿用 `data/{can,square}`(core hdf5 与 DP-base ckpt 已用硬链接重建,零额外空间)。
