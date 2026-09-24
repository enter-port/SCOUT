# 设计与模型文档

当前实现从 [idea_notes.md](idea_notes.md) 开始阅读；方法细节以对应代码为准，
运行参数以 campaign 配置和实际生成的训练配置为准。

| 文档 | 定位 |
|---|---|
| [idea_notes.md](idea_notes.md) | 当前网络、loss、ATY/ORBIT、标定、六轮链及数据回灌口径 |
| [entropy_cost.md](entropy_cost.md) | 封顶 KL 的设计动机、实现计算和推导边界 |
| [kappa_pr_base_calibration.md](kappa_pr_base_calibration.md) | 已测 base 上的 P/R 经验结论及适用范围；不是通用剂量规范 |
| [research/drift_guide.md](research/drift_guide.md) | 尚未接入标准链的 Drifting/1-NFE 研究提案 |
| [archive/README.md](archive/README.md) | 早期设计、实现计划、历史参数、性能与事故记录 |

`figures/` 保留已有 ATY/ORBIT 示意图及生成代码。图是机制示意，
参数、soft feedback 和噪声开关应与当前代码核对，不作为实验结果。
原始 [idea.JPEG](idea.JPEG) 保留作为构思资料。

实际操作见 [训练脚本说明](../scripts/README.md) 和 [标定说明](../scout/calib/README.md)。
结果及运行过程统一放在 [experiments](../experiments/README.md)，不在这里复制逐轮数字或维护进程状态。

2026-09-24 整理时，将已过时的 stage-1/low_dim 计划、旧“权威设计”和早期运维记录移入归档；
保留原文追溯，并修正当前笔记中的 LayerNorm、动作切片、free-bits、ORBIT soft feedback、
标定接口和 rescue 数据选择等描述。
