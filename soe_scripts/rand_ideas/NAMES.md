# 方法命名与实验对照表(权威,持续维护)

> 规则:方法名 = 简单易懂的英文短名,与代码 `--guide` / rand_costs 文件名一致;
> 旧文献里的"方案N"一律按下表替换。数据根:can = `scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can/rollout/`(正式链与探针)和 `scout-rand/data/rand/`(campaign);20 场景筛 = 前 20 个 seed42 场景、rescue×10、env20;100 场景 = seed42 全部、env50。

## 方法名

| 名字 | 旧称 | 一句话定义 | 代码 |
|---|---|---|---|
| **entropy cost** | 方案三 | −min(KL(q(z\|s̄,a)‖q(z\|s̄,a⁰)), κ),从自身意图定向逃逸(现行主线) | `--guide atypical`,entropy_costs.py |
| **NLL cost** | v0 | ‖z−μ(s̄,a)‖²,z 从先验采样(已弃) | cost.py(历史) |
| **novelty** | 方案二 | 对本场景已执行 code 的 KDE 斥力(已弃) | `--guide novelty` |
| **shell** | 方案A | 随机 κ-壳目标后验 q\*=N(μ⁰+√(2κ)σ⁰⊙u,σ⁰²),u 每重试随机方向 | `--guide shell`,ShellTargetCostPlanner |
| **pshell** | — | shell 但锚 (μ⁰,σ⁰) 整条重试冻结(持久外场;发现:比 chunk 重锚更窄) | `--guide rand_pshell` |
| **ushuffle** | — | shell 且 u 逐 chunk 重抽(测宽度来源,在跑) | `--guide rand_ushuffle` |
| **dose** | — | entropy cost × 每重试随机剂量乘子 w~logU | `--guide rand_dose` |
| **mjitter** | — | KL 参考精度随机旋转 τ²=σ⁰²/softmax(bξ) | `--guide rand_mjitter` |
| **rshell** | — | shell 但 u 只在探针实测可达子空间采样 | `--guide rand_rshell` |
| **smask** | — | entropy cost × 随机时间窗门控 | `--guide rand_smask` |
| **failanchor** | — | 锚=本场景失败重试的意图均值,推离已失败行为 | `--guide rand_failanchor` |
| **entropyseek** | — | 奖励随机选中维的 lnσ 增大(σ 轴)+entropy cost 锚 | `--guide rand_entropyseek` |
| **covtilt** | — | 动作空间专家协方差线性倾斜(违反 KL 形式约束,未实现即弃) | 无 |
| **combo** | — | novelty + entropy cost 求和(历史) | `--guide combo` |
| **expert-bank** | — | 引导向 core 数据最近邻 expert z(历史) | `--guide expert` |

## 关键实验 tag(按时间序)

| tag(数据目录名) | 方法 | 协议 | 关键结果 |
|---|---|---|---|
| SCOUT-exp1 | entropy cost | 100 场景 env12(r1 正式) | SR .62 / 救回 14 / pass@10 **0.76** |
| PROBE-base-pass10 | entropy cost | 100 场景 env50 重测 | SR .59 / 救回 19 / pass@10 **0.78** / PR 1.38 |
| DP-exp1(can 正式链 r1) | 无引导重试 | 100 场景(placebo 锚) | 救回 6 —— 100 场景运气底线 |
| PROBE-shellA-pass10-eta{0p115,0p35,0p7} | shell | 100 场景 | 救回 10/13/12,严格子集;eta0p35 最优 pass 0.72 |
| MINI-shell-* | shell | 10 场景扫 κ/η | κ1.25≈2.5>5;η3.0 inject 26× 过猛已停 |
| base3_screen | entropy cost | 20 场景(基线) | 14/20,救回 4/6 {3,9,12,18},pass 0.90,jerk 0.350 |
| base3_screen_cap10bug | entropy cost(κ=10 误跑) | 20 场景 | 作废留档(cap 漏传事故) |
| mjitter_b2 | (注入≡0) | 20 场景 placebo 对照 | 救回 {9,18} = 20 场景运气底线 |
| dose_wl*、mjitter_b*、rshell_*、smask_*、fa_*、es_rho* | 各方法 | 20 场景 | 见 RESULTS.md;全部=基线子集或平 |
| pshell_{r_k25,r_k5,c_k25} | pshell/chunk 对照 | 20 场景+宽度指标 | **c_k25(=shell 行为)最宽 PR 1.72/d_act 0.77;r 臂反而窄 25-30%** |
| (在跑)ushuffle 臂 | ushuffle | 20 场景+宽度 | 分辨宽度来源:锚追踪 vs 方向重抽 |

## 判读框架(2026-08-28 重定义)

主指标 = **重试分布宽度**(PR/μ̄ 两两距离/d_act/终态散布);SR = 护栏(≥ placebo {9,18});旧"救回数排序"框架作废。
