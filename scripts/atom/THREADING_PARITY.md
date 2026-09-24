# threading 标准链 parity 记录

对照基线：Git 当前历史中的 `scripts/threading/round_thm3.sh`。比较对象：
`configs/threading/campaign.json` 加 `scripts/atom/` 执行器。

以下非标定行为逐项一致：

| 项目 | 标准配置 | round_thm3 |
|---|---:|---:|
| base DP epoch / batch / workers | 600 / 64 / 8 | 600 / 64 / 8 |
| round DP epoch / batch / workers | 600 / 64 / 8 | 600 / 64 / 8 |
| base/round dyn epoch / batch | 300 / 256 | 300 / 256 |
| dyn steps/epoch / beta | 100 / 1e-5 | 100 / 1e-5 |
| fixed eval scenes / seed / envs | 100 / 42 / 25 | 100 / 42 / 25 |
| rescue workers / envs per worker | 8 / 25 | 8 / 25 |
| pass budget / flush | 5 / 100 | 5 / 100 |
| retry data policy | rescue + stop-on-first-success | 相同 |
| guidance | raw atypical eta/kappa | 相同 |
| rounds 1-5 | eval+explore → DP → dyn | 相同 |
| round 6 | full eval+explore, no retrain | eval-pk, no retrain |

唯一的算法差异是剂量标定：标准链 round 1 使用 P6，round 2-5 使用 KL median + R=0.01，
round 6 沿用 round 5 dose；round_thm3 使用 R-anchored eta + C-anchored kappa。

atom runner 将旧的 `shard_rollout.sh` 改为 Python 模块，worker 切片、合并、输出 HDF5 和
`stop-on-first-success` 参数保持相同。W&B heartbeat/backfill 属于监控与日志附加流程，不改变
rollout、数据选择或训练输入；标准链使用 eval 的 run id 继续 DP/dyn run（online 模式时）。

静态校验在 2026-09-24 通过：所有训练预算、数据预算、seed、guidance 和 final-round 条件均匹配。
