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
| **portfolio** | — | 同场景重试分工:前 K 次用 entropy cost 公式(保救回)、其余用 shell 公式(供宽度) | `--guide rand_portfolio` |
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

## 判读框架(2026-08-28 定稿)

主指标 = **重试分布宽度**(PR/μ̄ 两两距离/d_act/终态散布);**护栏(严格版,用户令 08-28)= 救回/pass@10 不得差于 entropy cost**(20 场景基线 4/6 {3,9,12,18}·pass 0.90;100 场景 19·0.76-0.78)——placebo 水平不再是合格线,只是运气底线参照。双目标:宽度↑ 且 SR 不降。

## 术语表(每个词首次出现都必须能在这里查到;新词入表)

| 术语 | 含义 |
|---|---|
| **placebo(安慰剂对照)** | 流程完全相同但引导注入恒为零的重试组 → 它救回的场景纯靠 DP 采样运气。20 场景运气底线 = {9,18}(mjitter_b2 意外实测);100 场景 = 6 个(DP 臂 r1)。作用:把"引导贡献"与"重试运气"分开 |
| **PR(参与率)** | 重试组宽度指标:把组内各轨迹的行为向量做主成分分解,PR=(Σλ)²/Σλ² ≈ 散布的有效维数。PR=1 全挤一条线(窄锥),PR=N 各方向均匀(喷雾) |
| **μ̄(行为摘要)** | 一条轨迹的所有 chunk 经冻结编码器的 μ 取平均 → 一个 16 维点;重试组宽度在这些点上量 |
| **d_act** | 动作空间宽度:两条重试的动作序列(重采样对齐 64 步)的归一化 RMS 距离,组内两两平均 |
| **teef / 终态散布** | 轨迹最后 10 步末端执行器位置均值;组内两两距离 = 各重试"最终到哪了"的散布 |
| **吸引子** | 12 个最难场景的全部 72 条重试(跨所有方法)最终都收敛到的同一个角落位置 eef=(0.218,−0.411,0.894)——策略像被磁铁吸住;成因=感知别名(观测分不清罐子在哪)+闭环 |
| **chunk(动作块)** | 策略每次规划 8 步 = 1 个 chunk;horizon 300 步 ≈ 37 个,逐块重规划 |
| **锚(anchor)** | cost 的参照点 (μ⁰,σ⁰)——"从哪里推离"。"锚追踪"=每 chunk 重捕当前意图作参照(shell 原行为);"持久锚"=整条重试只用首 chunk 参照(pshell) |
| **η(guidance_scale) / κ** | η=注入力度旋钮;κ=KL 偏离预算上限(nats,控制"最多推多远") |
| **护栏** | 新框架下 SR 不作优化目标、只要求不低于 placebo 水平 |
| **指纹** | 场景身份:轨迹初始状态 states[0] 的 md5;同指纹=同场景,用于跨 run 对齐救回集合 |
| **20 场景筛 / 100 场景** | 前 20 个固定场景的小实验(~30 分钟)vs 全部 100 个(~2 小时) |
| **never-wall / COLLAPSE** | never-wall=迄今所有方法都没救回过的场景;COLLAPSE=其中重试全部收敛到同一吸引子的子类(感知别名型) |
| **运气底线** | placebo 的救回数;20 场景 2/6(33%),100 场景 6/39(15%) |
| **宽度** | 主指标:同一场景重试行为分布的散布程度(PR/d_act/终态散布三口径) |
| **mean_inject** | 引导注入幅度遥测(每 5000 次注入的均值);注意幸存者偏差:早失败的低剂量轨迹贡献步数少 |

