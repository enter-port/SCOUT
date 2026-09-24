# 原子脚本与标准链验证 — 2026-09-24

## 本地

`python -m unittest discover -s tests -v`：33 项通过。
新增测试涵盖六轮数据传递、末轮无训练、arm 数据隔离、core 去重、future-round 排除、
空成功集重训、grid 共用失败集及择优、P6/KL median 及其他标定模式、错误 PR 结果拒绝、失败重试与完成记录复用。
标准示例配置的完整六轮 dry-run 成功，测试确认无文件写入。
`git diff --check` 通过。

## 服务器真实 GPU 短验证

服务器：`106.14.2.243:1022`，GPU 0/1/2；使用任务环境 `.venv_mg/bin/python`。
隔离代码与产物：`/root/workspace/baojiachun/atom_validation_20260924/`。
未覆盖服务器主仓库的在用脚本或未提交修改。

输入来自 `COFFEE-MG-p5-s233` 的真实 core/base pair。验证用 core 仅保留 3 个 demo、
每个最多 48 帧；环境 horizon=16，2 个 scene，2 个 rescue worker，pass@1。
DP 预算为 1 epoch、1 train batch、1 val batch；dyn 为 1 epoch、1 train step。
这些预算仅用于接口和数据流验证，不能据此判断训练效果或选定正式剂量。

| 项目 | 真实执行结果 |
|---|---|
| base training | 从头完成 DP 与 dyn 短训练，均生成 checkpoint |
| grid search | η=2.8、5.6，κ=2.5；两个 cell 完成，复用同一失败集，生成 verdict |
| calib | 独立 R→C 完成且退出码 0；短链两轮各完成 R 标定并传递剂量 |
| eval + explore | 真实 Coffee 环境 eval、两个并行 rescue worker、JSON/HDF5 合并均成功 |
| DP retrain | 零新增成功时仍完成训练；累计数据为 3 条 core |
| dyn retrain | 完成训练；累计数据为 3 条 core + 2 条失败轨迹；encoder 确认为本轮新 DP |
| 标准链 | 缩短为 2 轮完整跑通；第 1 轮重训，第 2 轮只测量，生成 summary |

服务器独立校验脚本 `validation/audit.py` 断言了 core/DP/dyn 数据条数 `3/3/5`、
dyn encoder checkpoint 来源、RC 退出码和最后一轮没有 dp/dyn 训练目录。
权威检查输出为 `validation/audit.json`，两轮结果为 `validation/chain/summary.json`。
正式 6 轮预算没有运行。

最终版本重入检查通过：base 复用 2 个阶段，chain 复用 13 个阶段，两个退出码均为 0，
控制台无新的 `RUN:` 命令。服务器执行新增 11 项编排测试全部通过；收尾 GPU 进程列表为空。

运行命令（在隔离仓库根目录）：

```bash
PY=/root/workspace/baojiachun/.venv_mg/bin/python
"$PY" -m scripts.atom.base_train --config validation/base.json
python train.py --config configs/threading/campaign.json --dry-run
bash validation/calib_smoke.sh
"$PY" validation/audit.py
"$PY" validation/resume_check.py
```

完整产物、短验证配置、日志及 helper 都保留在服务器 `validation/` 目录。
本地证据副本位于 `experiments/2026_09_24_atom_refactor/`（实验记录目录不进 Git）。

隔离部署首次发现两项环境问题并已修复：Windows shell 换行需 LF（新增 `.gitattributes` 保证）；
`diffusion_policy/env` 被仓库现有 ignore 规则排除，因此隔离副本只读链接服务器已有环境包装代码。
失败 eval 没有留下完成记录，修复依赖后重入成功，已完成 base 自动复用。

本次 GPU 覆盖 Coffee 的 R/RC 路径；其他任务、PR/DP-KL/ORBIT 的分发及参数由本地测试/dry-run 检查，
未对每个任务和模式重复做 GPU 实验。恢复到失败阶段会重启该阶段，不续训其未完成 epoch。
