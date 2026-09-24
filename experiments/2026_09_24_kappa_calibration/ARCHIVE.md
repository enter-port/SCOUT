# Calib 实验与代码迁移归档

本目录按用户要求纳入 Git。仓库原有 `experiments/` 忽略规则仍适用于其他实验；
本次实验归档只提交 5 份 Markdown 记录和 5 份报告／历史编排 Python 脚本。

## 阅读顺序

1. [RESULTS.md](RESULTS.md)：已完成实验的最终结果、逐项 η/κ/R、救回数与 pass@5。
2. [METHODS.md](METHODS.md)：P6、自然 KL 分位数、C 标定和 DP KL 中位数的方法与适用边界。
3. [README.md](README.md)、[WORKING_STATE.md](WORKING_STATE.md)：按时间记录的实验过程，早期“运行中”状态保留作历史记录。
4. [当前标定代码用法](../../scout/calib/README.md)：迁移后的入口、参数和调用结构。

## 归档内容

- `ARCHIVE.md`、`METHODS.md`、`README.md`、`RESULTS.md`、`WORKING_STATE.md`：归档说明、方法、实验过程与最终报告。
- `build_report.py`：根据 snapshot JSON 重建最终报告；在仓库根目录运行
  `python experiments/2026_09_24_kappa_calibration/build_report.py --final`，会更新本目录报告。
- `snapshot/coffee_r2_fixed_k1_after_r25.py`、`snapshot/round_diversity_probes.py`、
  `snapshot/round_median_dose_probes.py`、`snapshot/sequential_c_probes.py`：当时使用的实验编排脚本。

JSON 明细、分片结果、manifest、checkpoint、HDF5、NPZ 和压缩包留在本地／服务器，
不纳入这次 Git 提交。报告中的这些路径是原始实验数据引用。
从 Git 新检出后，如需运行 `build_report.py`，须先从原实验目录补齐对应的 snapshot JSON。
服务器实验目录为 `/mnt/workspace/baojiachun/scout/experiments/2026_09_24_kappa_calibration/`。
本地保存的 NPZ 仍可用于 posterior 数值复算。历史记录描述的是当时执行的代码，
本次迁移没有重新运行实验。

## 本次代码迁移与验证

原 η、原 C、P6 和 DP KL 中位数实现集中到 `scout/calib/`。旧 Python 入口保留转调兼容，
coffee/threading 的 RC、PR 分支改用模块入口。原算法的观测选取、R 分母、随机种子、
KL 方向、迭代预算与收敛语义保持对应关系；固定 base κ 与当前轮 κ 起点分别传入。
完整接口说明见上面的标定文档。

- 25 项本地测试通过，覆盖标定协议、兼容入口、失败处理与 round 参数传递。
- 用 13 组已有 posterior NPZ 复算 KL 与中位数，均在 `rtol=atol=1e-12` 内一致；
  全部数组最大绝对差约 `2.3e-13`。
- 提交前重新检查 66 条配对重试记录：base 31 条、round1 16 条、round2 19 条，
  合计 13 个 DP 对照和 53 个 SCOUT 评估；与保存的结果表逐字段一致（路径字段除外）。
- 本次迁移仅在本地完成；没有重新运行 GPU 采样、环境评估或部署服务器。
