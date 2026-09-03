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
| **exploit guide** | — | LPB 式吸引引导:min‖slice(D_s(μ(s̄,a),s̄)) − expert state-bank 最近邻‖₂,把 D_s 预测的下一状态 latent 拉向专家(训该 DP 的数据)访问过的状态流形——方向与 entropy cost 相反(2026-08-29,branch exploit-dev) | `--guide exploit --exploit-latent eye`,exploit_costs.py |
| **particle** | — | 并行粒子互斥:同场景 10 条重试锁进同一重规划批,生成期间在编码器 μ 空间加 RBF 斥力(中值带宽);η 径向 × 斥力角向(2026-08-30,Corso ICLR 2024 血统,branch entropy-random-dev) | `--guide particle --pg-lambda/--pg-h-scale/--pg-start`,particle_costs.py |
| **orbit** | — | 约束控制:κ−δ 以下照旧爬坡(argmax 控制),以上切「Newton 反馈扶 κ + 切向噪声」在逃逸壳上巡行——单轨迹换控制律,不靠系综耦合(2026-08-31,#math 会话定稿,branch entropy-random-dev) | `--guide orbit --orbit-lam/--orbit-delta/--orbit-sigma`,orbit_costs.py |
| **orbit sector** | — | orbit 切向噪声模式:iid=逐步独立采样(默认,逐位不变);det=逐(场景,重试)确定性方向缓存,各重试沿不同固定方向巡壳=角向分层覆盖(beat-SOE campaign B2,2026-08-31,subagent 审后 SHIP) | `--orbit-sector {iid,det} --orbit-sector-seed`,orbit_costs.py |
| **noise anneal(p)** | — | orbit 切向噪声退火指数:噪声携带 (1−ᾱ_t)^{p/2}(p=1 原行为逐位不变,p>1 晚期去噪步噪声压得更狠=jerk 杠杆;beat-SOE B3,复审 SHIP) | `--orbit-noise-anneal`,orbit_costs.py |
| **κ 分层** | — | 不同 --atypical-cap 的 orbit 进程池化=径向分层(κ1.5 近逃逸/2.5 标准/4.0 深逃逸) | 多进程池化,零新代码 |
| **混编(mixing)** | — | 同一失败集上多机制臂各自 --explore-try-times k 的池化 pass@10(每场景总试数=10);离线模拟可从 explore_detail 曲线直接算;确认 run 换 explore seed 防同流水偏差 | `soe_scripts/sq2_mix_sim.py` |

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
| particle G1/G2/G3(CAN,s233 base) | particle λ=0.25,pg_start 0/50/90 | CAN 41 失败定集 rescue×10 | 救回 21/19/22 vs 基线 19;pass@10 0.80/0.78/0.81 vs 0.78;PR 1.29/1.27/1.42 vs 1.38(**锥未展宽**);μ̄ 两两距 G1 +34%(0.406 vs 0.303)=沿同锥更远非新方向 |
| particle G1/G2(SQUARE,s233 base) | particle λ=0.25,pg_start 0/50 | SQUARE 62 失败定集 rescue×10 | **救回 33/31 vs atypical 30;pass@10 0.71/0.69 vs 0.68**(首个正信号,G1>G2 与 CAN 排序一致);jerk 0.336/0.296;PR 1.22/1.28 vs 1.28(未展宽,跨任务一致) |
| sq_orb_cal_{s010,s025,s050} | orbit λ=0.5 δ=0.25 σ=0.1/0.25/0.5 | SQUARE 6 场景探针 | 救回 4/3/5(无 λ=1.0 式崩溃);noise inject 0.79/1.98/3.90;fb 0.19-0.27 温和;phase-2 占比 ~48%(机制大量生效);jerk 0.452/0.458/0.649 |
| (完赛)sq_orb_{s025,s050} | orbit 同上 σ=0.25/0.5 | SQUARE 62 失败定集 rescue×10,双臂 GPU4/6 | 🏁**orb_s025 = 全场最佳:救回 36、pass@10 0.74**(atypical 30/0.68、G1 33/0.71);orb_s050 32/0.70 过量;jerk 0.471/0.647(高于 particle 0.30-0.35);**PR 1.38 全场最高且 μ̄ 距 0.188 最低 = 径向压缩+角向展开,orbit 机制签名**(particle 反向:距远 PR 平);重叠:orb_s025 保 atypical 27/30 只丢 3、新破 9;五臂并集 51/62,11 场景永不救(定理 C 带内) |
| (完赛,7/9 后用户停)sq_orb025_s{S}_r{R} | orbit σ=0.25 网格:3 seed(s233/2333/23333)× 链轮 r1(base)/r2(exp1)/r3(exp2) | SQUARE 各格自建失败集(eval 与历史逐位吻合)rescue×10,GPU1/4/6 | 🏁**轮次反转:r1 均值 +2.7(+6/+7/−5)、r2/r3 均值 −8(−8/+2/−17/−9),累计 −24——σ0.25 全链替代不成立,价值集中在 round1 冷策略**;机制=链回灌数据 atypical 型,后期 DP/VIB 长在 atypical 棱线上,切向噪声晃下棱线;s23333 全线负且 −17/−9 剂量带内(非剂量伪影);jerk 通胀 r2/r3 2-3×;详见 memory orbit-guidance-campaign |

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
| **逃逸壳(κ-shell)** | 等值面 {a: KL(q(z\|s̄,a)‖q(z\|s̄,a⁰)) = κ}——entropy cost 认定的"已逃逸足够远"的边界;orbit 在它上面巡行 |
| **切向噪声 ξ⊥** | orbit 术语:高斯噪声 ξ 减去其在 ∇KL 方向的分量(投影到等值面切平面)——方向覆盖的唯一来源,强度 σ_orb 定标 |
| **Newton 反馈** | orbit 术语:−λ(KL−κ)∇KL/‖∇KL‖²,把 KL 一阶精确拉回 κ(λ=1 一步到壳、<1 阻尼);λ 无量纲,不继承 η 的梯度尺度 |
| **相位切换缓冲 δ** | orbit 术语:KL ≥ κ−δ 即从爬坡切约束动力学——赶在 κ 封顶把梯度掐死之前交接 |
| **G1/G2/G3** | particle 斥力介入时序消融:去噪步 0/50/90 起开斥力(100 步循环) |
| **护栏** | 新框架下 SR 不作优化目标、只要求不低于 placebo 水平 |
| **指纹** | 场景身份:轨迹初始状态 states[0] 的 md5;同指纹=同场景,用于跨 run 对齐救回集合 |
| **20 场景筛 / 100 场景** | 前 20 个固定场景的小实验(~30 分钟)vs 全部 100 个(~2 小时) |
| **never-wall / COLLAPSE** | never-wall=迄今所有方法都没救回过的场景;COLLAPSE=其中重试全部收敛到同一吸引子的子类(感知别名型) |
| **运气底线** | placebo 的救回数;20 场景 2/6(33%),100 场景 6/39(15%) |
| **宽度** | 主指标:同一场景重试行为分布的散布程度(PR/d_act/终态散布三口径) |
| **mean_inject** | 引导注入幅度遥测(每 5000 次注入的均值);注意幸存者偏差:早失败的低剂量轨迹贡献步数少 |
| **state bank(状态库)** | exploit guide 的专家库:训练该 DP 的数据(success_accum)每一帧的 E_s(s) 编码;s̄ 切片预设 eye=手眼 view 512 维(LPB Square [...512:] 同款)/agentview/visual(双 view)/full |
| **kNN cost(k 近邻代价)** | exploit guide 的 cost 变体(2026-08-30,commit 980b4fd):k>1 时取 bank 全局 k 近邻距离的均值代替最近单点——拉向流形局部密度而非单个(可能错配的)邻居;`--exploit-knn`,默认 1=LPB hard-argmin 逐位不变 |
| **软门(soft gate)** | exploit guide 的 OOD 门变体(2026-08-30,commit 38fb92a):开门时的力度按超限程度加权 w=min(slope·(cost−thr)/thr, cap)——近门限温和、强离群重击;默认 slope=None=LPB 二值门逐位不变;`--exploit-gate-slope/--exploit-gate-cap` |
| **cracks / drops(实拉括注)** | 75% campaign 取证术语:crack=门真开过且该场景被引导救回;drop=门真开过且该场景被引导弄丢;门从没开过的场景翻转叫实现漂移(flips),不计入引导功劳/损伤 |

| **exploit-matrix(冻结迁移矩阵)** | 2026-08-30 CAN 实验:冠军 vis250 冻结为方法(visual 切片+OOD 门 p75 标定+kNN1),铺 can 3 seed × 6 轮 ckpt,bank=训该 ckpt 的数据;含两项部署标定=门限 p75(逐 bank)+剂量单位换算 η_rN=250×g_sq/g_rN(can VIB 梯度是 square 的 7-43×,原样 250 全灭);终局=链内 −5 无增益,详见 experiments/exploit_can_matrix.md |

| **sq2 gate / GATE / SPLIT(conf 协议)** | beat-SOE campaign(08-31)的确认协议:GATE=1=前 20 场景失败子集(s233 为 12 场景)做 20env 门;SPLIT="arm:k,..."=每场景 10 发重试按臂分配(如 orb025:6,att:3,plc:1),pooled pass@10 为测量;SEED=重试 RNG 种子(--rescue-seed,42=历史流/43=新鲜确认),场景集恒 42 |
| **臂名缩写(sq2)** | plc=placebo(--guide off 零引导重试);att=atypical(entropy cost κ2.5);par=particle(λ0.25);orb025/orb015=orbit σ;orbk15/orbk40=orbit κ=1.5/4.0;orbdet=orbit sector=det |
