# 历史归档

这些文档保留历史方案、参数及实证，不再决定当前实现。
正文中的“权威”“默认”“待实现”以及旧任务运行状态只属于记录时点。
当前模型见 [idea_notes.md](../idea_notes.md)，执行入口见 [scripts/README.md](../../scripts/README.md)。

| 记录 | 归档原因 |
|---|---|
| [idea.md](idea.md) | 原始构思及后续手工补记；由当前模型笔记统一说明实现 |
| [scout_design.md](scout_design.md) | 早期设计规范，混有旧 cost、评估和待定项；不再作为“权威设计” |
| [scout_impl_plan.md](scout_impl_plan.md) | low_dim、StateAE、像素重建路线等已不对应当前图像 latent dynamics 实现 |
| [stage1_plan.md](stage1_plan.md) | 单轮采样目标 z 的 stage-1 执行计划，已被当前标准链取代 |
| [evaluation_plan.md](evaluation_plan.md) | 初期比较方案；当前 rescue 指标和轨迹选择以代码及现行笔记为准 |
| [long_term_plan.md](long_term_plan.md) | 早期阶段里程碑，不能用来判断当前完成状态 |
| [speedup_report_2026-08-14.md](speedup_report_2026-08-14.md) | 当时环境和数据规模下的性能/磁盘快照 |
| [guidance_batch_scaling_bug.md](guidance_batch_scaling_bug.md) | 已修复的 1/B 事故；其中 0.01 等剂量不是当前默认值 |
| [entropy_cost_2026-08.md](entropy_cost_2026-08.md) | 早期推导；有限步采样分布、信任域和熵增结论需按现行说明限定 |
| [environment_2026-08.md](environment_2026-08.md) | 从旧 README 保存的环境版本和安装过程快照 |

历史链接已按移动后的路径调整；未随当前仓库保留的历史材料以普通文字注明，不伪造现有入口。
归档正文不逐条改写成现行结论，以免失去原始方案的可追溯性。
