# 三任务 base 的近似共同标定：P=ηκ≈6，R≈0.01

记录日期：2026-09-24。范围限定为 can、threading、coffee 的本次 seed233 base
DP/dyn checkpoint。两个指标允许±10%相对误差。

| 任务 | κ | η | ηκ | 实测R | SCOUT/DP救回数 | 判定 |
|---|---:|---:|---:|---:|---:|---|
| can | 2.5 | 2.299270 | 5.748175 | .010600 | 15/8 | 优秀 |
| threading | 1 | 5.992432 | 5.992432 | .009531 | 22/16 | 基本通过 |
| coffee | 7.071068 | .848528 | 6 | .009162 | 25/16 | 优秀 |

评估：seed42–141共100个场景；冻结初始DP失败集；DP和SCOUT各5次重试；
4个worker×25环境。严格超过DP为基本通过，严格超过1.5倍为优秀。
R分母为core原始abs_actions平均绝对值，不是noisy action。

结论：在上述三任务base上，P≈6可作为近似共同指标，三者全部超过DP、两者优秀。
尚未达到三个任务全部优秀，也未证明优于固定κ=2.5再标定η：后者分别为
15/8、26/16、23/16。本结论是在探索中提出的，不是独立留出集上的确认。
不能扩大成跨轮保证：coffee round2的P6为9/DP11；tool_hang base也不支持P6。

依据：仅当成本变化主要满足f_new=c*f_old时，κ_new=c*κ_old、η_new=η_old/c
保持η∇min(f,κ)不变，此时ηκ不变。不同模型成本形状变化时不保证成立。

实施：令η=P_target/κ，在固定128个core观测、固定RNG seed0上有界求解κ，
直到实测R进入目标区间。不是先标定η后直接计算6/η，也不是rollout成功率网格搜索。
直接checkpoint入口严格沿P=6求解；此前can/threading结果允许复用±10%带内的
已有core点，因此新求解器不保证返回表中完全相同的κ/η或相同救回数。

证据目录：`experiments/2026_09_24_kappa_calibration/`，其中`RESULTS.md`、
`METHODS.md`、`snapshot/potential_base_mapping.json`及各arm的summary保存完整记录。
算法、直接用法和调用结构见 [`scout/calib/README.md`](../scout/calib/README.md)；
当前标定算法见 [`scout/calib/README.md`](../scout/calib/README.md)。
