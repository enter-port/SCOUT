# SCOUT 标准训练入口

当前训练只保留 `scripts/atom/` 中的原子操作，由根目录 `train.py` 根据 campaign 配置编排。
旧的 task-specific round、launch、probe 和 shell chain 不再作为执行入口。

## 标准 threading 配置

```bash
python train.py --config configs/campaign_threading_p6_klmedian.json --dry-run
python train.py --config configs/campaign_threading_p6_klmedian.json
```

配置要求输入已经完成 split、绝对动作转换的 core-only HDF5。运行阶段由 atom 产生并传递产物：

```text
base DP → base dyn
round 1: P6 (eta*kappa=6, R=0.01) → eval+explore → DP train → dyn train
round 2-5: KL median + R=0.01 → eval+explore → DP train → dyn train
round 6: 使用 round 5 dose → eval+explore
```

`calib` 返回的实际 `eta/kappa` 是 `eval_explore` 的输入，配置只声明标定方法、目标和测量预算，
不会在 rollout 配置中复制剂量。`round 6` 的 `train: false` 是显式配置，因此不会隐式执行 DP/dyn retrain。

## Atom 接口

| 模块 | 作用 |
|---|---|
| `base_train` | 在 core 上依次训练 base DP 和 base dyn |
| `grid_search` | 可选的固定 base 网格探索；标准 threading 配置关闭 |
| `calib` | 调用 P6、KL median、R、RC 或其他标定入口并返回 dose |
| `eval_explore` | 固定场景 eval、冻结失败集、8-worker rescue、合并 success/all HDF5 |
| `dp_train` | core 加累计成功轨迹，按本轮训练预算重训 DP |
| `dyn_train` | core 加累计探索轨迹，使用本轮 DP encoder 重训 dyn |
| `shard_rollout` | atom 内部的 Python worker/merge 实现，替代旧的 shell driver |

所有阶段都有 `done.json` receipt，输入或配置变化会拒绝复用旧产物。失败阶段会保留 attempt 日志，
重新运行时只重试失败阶段。

## threading 与最新 round_thm3 的非标定口径

标准配置逐项复用 `round_thm3.sh` 的非标定设置：

- base DP 600 epoch、batch 64、8 DataLoader workers、checkpoint 每 100 epoch；
- 每轮 DP 600 epoch、batch 64、8 workers、checkpoint 每 300 epoch；
- base/round dyn 300 epoch、batch 256、100 steps/epoch、`beta=1e-5`；
- 固定 seed 42 的 100 个 eval 场景，25 个 eval env；
- rescue 使用 8 个 shard worker、每 worker 25 env、pass@5、`flush_every=100`；
- `stop-on-first-success`，成功轨迹和失败场景首次探索轨迹的 HDF5 口径保持一致；
- round 1-5 执行 DP/dyn retrain，round 6 完整 eval+rescue 后停止；
- ATY 使用 raw eta/kappa 和 atypical guidance，`eval` 与 `explore` 使用同一 dose。

两者唯一有意不同的是标定：标准配置 round 1 使用 P6，round 2-5 使用 KL median + R=0.01，
round 6 不再标定；round_thm3 使用 R-anchored eta 加 C-anchored kappa。
