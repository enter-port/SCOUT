# SCOUT

SCOUT 用 VIB 动力学编码器在 Diffusion Policy 去噪过程中提供 atypical guidance，
通过探索轨迹循环训练 DP 和 dyn。

## 当前标准入口

当前唯一的训练编排入口是根目录 `train.py`，原子操作位于 `scripts/atom/`。
标准 threading 配置为 [campaign_threading_p6_klmedian.json](configs/campaign_threading_p6_klmedian.json)。

```bash
python train.py --config configs/campaign_threading_p6_klmedian.json --dry-run
python train.py --config configs/campaign_threading_p6_klmedian.json
```

配置需要一个已经完成 split 和绝对动作转换的 core-only HDF5。运行顺序为：

```text
base DP → base dyn
round 1: P6 (eta*kappa=6, R=0.01) → eval+explore → DP train → dyn train
round 2-5: KL median + R=0.01 → eval+explore → DP train → dyn train
round 6: 沿用 round 5 dose → eval+explore
```

标定 atom 输出的实际 eta/kappa 会直接传给 `eval_explore`；rollout 配置不重复声明这些运行时剂量。

## 非标定训练口径

标准 threading 配置与最新 `round_thm3` round driver 保持一致：base DP 600 epoch、round DP 600 epoch、
DP batch 64、8 个 DataLoader workers；dyn 300 epoch、batch 256、100 steps/epoch、beta=1e-5；
固定 seed 42 的 100 个 eval 场景；8 个 rescue shard worker、每 worker 25 env、pass@5、
`stop-on-first-success` 和 `flush_every=100`。前五轮训练，最后一轮只做完整 eval+rescue。
唯一改变的是标定方法：P6 → KL median + R=0.01 → final carry-over。

## 文档

- [原子操作与标准配置说明](scripts/README.md)
- [当前模型、loss、数据口径](idea/idea_notes.md)
- [标定算法](scout/calib/README.md)
- [实验记录](experiments/README.md)

## 代码结构

```text
train.py             campaign 配置入口；无 --config 时保留 DP Hydra backend
scripts/atom/        base、grid、calib、eval+explore、DP、dyn 原子操作
configs/             threading 标准 campaign 与其 DP/dyn/eval backend 配置
scout/model/         状态编码器、VIB encoder、动力学 decoder
scout/guidance/      ScoutPolicy、KL/ORBIT guidance
scout/calib/         P6、KL median、R/C 标定算法
scout/eval/          rollout、轨迹选择、HDF5 合并和指标
experiments/         实验记录
idea/                当前模型笔记和历史归档
```

验证：

```bash
python -m unittest discover -s tests -v
python train.py --config configs/campaign_threading_p6_klmedian.json --dry-run
```
