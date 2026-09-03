# 全向逃逸(escape coverage):从「最远方向」到「覆盖所有逃逸方向」

- 日期:2026-08-30 | 状态:**研究 + 提案,未落代码,待用户拍板**
- 定位:`entropy-random-dev` idea 的下一代。上一代(rand-cost campaign,11 臂全记录见 `soe_scripts/rand_ideas/NAMES.md`)改的是「**单条重试的 cost 随机化**」;本代改的是「**重试系综与回灌数据的覆盖结构**」。
- 命名:方法名均为简单英文短名(与 NAMES.md 规范一致),落地时同步追加 NAMES.md;文中全部术语首现即白话解释。

---

## 0. 问题陈述

### 0.1 用户原话 → 形式化

> KL cost 相当于把轨迹往"和目前分布偏差最大"的方向 guide。"最大"意味着失去往别的方向探索的机会。要往**所有**能远离当前分布的方向走,而不是只走"最"远离的那一个。

现行 entropy cost 的引导后分布(推导恒等式,`idea/entropy_cost.md`):

  p\*(a|s̄) ∝ p_DP(a|s̄) · exp( min( KL( q(z|s̄,a) ‖ q(z|s̄,a⁰) ), κ ) )

**理论上 p\* 本身就是"全方向"的**:任何 KL>0 的方向都被 e^{KL} 加权,质量按偏离程度分配到所有逃逸方向上——这不是 argmax,是指数倾斜分布。

**实现缺口**:实际注入是确定性梯度上升(每去噪步 `−η√(1−ᾱ_t)·∇cost`)。梯度上升从 a⁰ 出发只收敛到 cost 景观的**一个局部峰**;cost 对 a 的曲率由 VIB 编码器雅可比的 JᵀΛJ 型结构主导 → 同一场景的全部重试(同 s̄、同景观、独立采样)收敛到**同一个主方向峰**。实测签名:窄锥 PR 1.30(重试系综行为分布的有效维数,≈1 即挤成一条线),重试共享逃逸主方向(2026-08-27 定量闭环)。

**一句话诊断:「只走最远方向」不是 cost 定理的错——定理说的是分布;是采样实现的错。我们从未从 p\* 采样,只做了 argmax 的贪心近似。** 因此本代改的不是 cost 公式,是**采样与分配机制**。

### 0.2 三个泄漏点(方案必须逐点对准)

| 泄漏点 | 层次 | 病理 | 优化文献的名字 |
|---|---|---|---|
| **L1 轨迹内** | 单条重试的去噪循环 | 确定性爬坡锁死单峰,从未体现 p\* 的多方向质量 | basin trapping / greedy argmax |
| **L2 重试间** | 同场景 rescue×10 | 10 条 i.i.d. 重试面对同一景观 → 同盆同峰(窄锥) | mode collapse / 同分布重复抽样 |
| **L3 跨轮** | 回灌数据组装 | 数据近重复(fresh 口径 86% 场景重合;rescue 口径同场景重试互为近重复)→ retrain 学不到新方向 | redundant samples / 无覆盖控制 |

### 0.3 「所有方向」的严格边界(自家三定理,必须先划清)

- **定理 A(h-变换保零测集)**:以上一切机制只能在 supp(p_DP) 内重分配质量。「所有能远离的方向」严格限于逃逸集 E_κ = {a : KL ≥ κ} ∩ 支撑集。**支撑外的新方向要绝对锚(exploit 线的 state bank)或支撑扩展,不在本代范围**——这是"相对锚稳定/绝对锚可外延"原理级 trade-off 的本代选择:本代 = 支撑内全覆盖。
- **定理 B(e^κ 质量放大上界)**:总逃逸质量 ≤ e^κ。摊到 K 个方向 → 每方向质量下降;某方向要"更远"需要更高剂量。⇒ **覆盖不是免费放大,是固定质量预算下的分配问题**(ergodic 观点:预算按目标密度比例分配是最优控制形式,见 §1C)。
- **定理 C(别名可测性不可能)**:12 个 COLLAPSE 场景(感知别名:观测分不清罐子位置)无论方向怎么铺都测不出差异 → **预期仍是 never-wall**。本代主战场 = 非别名的可逃逸场景(现基线救回集 4/6 的同类)。

---

## 1. 文献地图(五家族)

用户判断正确:这个问题在优化/采样领域被反复做过,而且**每个家族有不同的名字和不同的可借机制**。逐族给:代表工作 → 核心机制 → 映射到 SCOUT。

### A. 把 argmax 换成采样(温度化)

| 工作 | 机制 | 映射 |
|---|---|---|
| Thompson sampling(Russo et al. 2018, tutorial arXiv:1707.02038) | 不确定哪个臂最优 → 从后验抽一个样本当"当前最优" | L1:从 p\* 采样,不从 argmax |
| Parallel tempering / replica exchange(Swendsen-Wang 1986;快速混合条件 Woodard et al. 2009;综述 arXiv:2501.05908) | **温度梯子**:热副本自由跳盆、冷副本深挖,副本间交换 | L2:重试 = 不同温度(剂量)的副本系综 |
| Wang–Landau / multicanonical(综述见上 arXiv:2501.05908) | 按能态密度重加权,每个能级等时访问 | 理念来源:重试预算跨 KL 能级均匀分配 |
| 模拟退火 / softmax continuation(Kirkpatrick 1983;softmax 平滑 max) | 有限温度 = argmax 的软化,τ→0 才退化成 argmax | L1 的理论依据:温度化注入(M5) |

### B. 粒子斥力(一批样本互相排斥)

| 工作 | 机制 | 映射 |
|---|---|---|
| **Particle Guidance**(Corso et al., ICLR 2024, arXiv:2310.13102) | 扩散采样时对一组粒子加**联合斥力势**,非 i.i.d. 采样 → 样本多样且质量不降(分子构象任务) | **与我们的架构完全同构**:同场景重试=粒子,斥力可串行实现=排斥已执行重试 |
| SVGD(Liu & Wang, NeurIPS 2016) | drift(吸向密度)+ repulsion(核斥力)→ 模式覆盖 | 斥力形式来源;**诚实局限**:RBF 核斥力局域,模式间不传播(annealed SVGD arXiv:2101.09815 修)→ 斥力核带宽必须自适应 |
| DvD(Parker-Holder et al., NeurIPS 2020, arXiv:2002.00632) | 种群 RL;核矩阵**行列式**=多样性目标;**核带宽在线学习** | novelty 的 logKDE 带宽可直接抄它的在线适应 |
| DPP 家族:Batch BO(Kathuria et al., NeurIPS 2016)、DPP-BBO(Nava et al., AISTATS 2022, arXiv:2110.11665)、Determinantal Beam Search(Vijayakumar et al., ACL 2021, arXiv:2106.07400) | **批量选择不用 top-k,用"质量×多样性"联合采样**(DPP=行列式点过程:一个子集的概率 ∝ det(相似度核),向量近了行列式小=概率低) | L3:回灌子集精选 / bank 候选选择 |

### C. 覆盖式优化(illumination / 档案)

| 工作 | 机制 | 映射 |
|---|---|---|
| MAP-Elites(Mouret & Clune 2015;QD 论文列表 quality-diversity.github.io) | 行为空间分格,每格保最优解 → "照亮"整个行为空间而非找单点最优 | L3:回灌数据按 μ̄(16 维行为摘要,已有)分格精选 |
| Go-Explore(Ecoffet et al. 2019/2021) | 档案记住"到过的最远状态",**从档案态继续探索**(需 reset);解 Montezuma | rescue 协议本身就是"记住失败场景回同初态重试"——档案思想已在,缺的是**跨重试档案利用** |
| Ergodic control(Mathew & Belta, ACC 2012;Miller & Murphey 2016) | 控制律使**时间分配 ∝ 目标密度**(遍历度量 = 轨迹时空统计与目标分布的傅里叶系数差) | 重试预算分配的理论伞:M1(剂量梯)的定理化语言 |

### D. 确定性散布(运筹/数值分析)

| 工作 | 机制 | 映射 |
|---|---|---|
| p-dispersion(经典:Kuby 1987;EJOR 2016 IP 解法;EJOR 2021 综述) | 选 p 个点使**最小两两距离最大**(max-min 分散) | shell 随机方向 → 确定性准均匀格点(M3) |
| farthest-point sampling / k-center | 贪心:下一个点选离已选集最远 | M2 的确定性版:第 k 条重试 = 行为空间里离前 k−1 条最远 |
| 分层抽样 / 低差异序列(Neyman 1934;QMC) | 同预算下**分层抽样的方差严格低于 i.i.d.**(O(1/K)) | L2 的定理化:命题 1(§2) |

### E. 自家负结果(必须绕开的坑,上一代 11 臂的教训)

shell(随机 κ-壳方向)、pshell(持久锚)、ushuffle(逐 chunk 重抽)、dose(随机剂量乘子)、portfolio(异 cost 分工)、mjitter(精度随机旋转)、rshell/smask/failanchor/entropyseek/covtilt——**全部 = 基线子集或平**(NAMES.md 关键实验表)。

**共同病理**:它们都把随机性注入「**单条重试的 cost 实现**」,而系综层面仍然是 10 次**同分布**抽样——随机方向的 10 次抽样覆盖期望低(重复+锥内浪费),这是 i.i.d. 采样的固有性质,不是方向本身不好。portfolio 的判负还有剂量共享问题(η 共享 → shell 行 8.6× 过剂量)。

**本代的本质区别:随机性从「生成机制」移到「不影响重复性的地方」,系综层做显式分配**(梯子/斥力/格点/档案)。这是采样理论从 i.i.d. 到 stratified/repulsive/DPP 的同一条演进路。

---

## 2. 理论整合(三条命题草案,各配可证伪实验)

### 命题 1(分层覆盖优势)
同预算 K 条重试:按逃逸集的分层(方向扇区或剂量能级)做**确定性分配**,重试系综行为分布的 PR 期望 ≥ i.i.d. 抽样,且两两重复率严格更低。依据:分层 vs i.i.d. 抽样的经典方差结果(Neyman)迁移;parallel tempering 副本系综对单温度链的模式覆盖优势。
**可证伪**:20 场景,ladder(M1)vs 现行同剂量,读 PR 分布 + 两两 d_act。若 ladder 的 PR 不升 → 命题被否,且说明逃逸集有效维数 < 2(窄锥是几何必然而非采样缺陷)——这本身是重要结论。

### 命题 2(斥力 = 可行性约束下的换峰)
novelty 斥力项的最优化读法:argmax_a p_DP(a)·e^{β·KL(a)} − λ·Σ_j k(f(a), f(traj_j)) 的解 = "**在 DP 认可的行动里,离已执行行为最远的那个峰**"。每条重试的 DP 一致度预算不降(斥力是拉格朗日罚,不是放松),峰却互异 → 理论上同时满足「宽度↑」与「护栏(SR 不降)」。
**可证伪**:rescue×10 三臂(纯 novelty / novelty+entropy 求和 / 基线),计量每条重试的 DP 似然代理(jerk / mean_inject)与救回。若斥力臂救回 < 4/6 而 PR 升 → 覆盖-可行性权衡比预期紧,需要 λ 折算。

### 命题 3(e^κ 预算守恒 → 收益在数据集宽度,不在单轨迹)
覆盖 K 个方向**不增加总逃逸质量**(定理 B)。收益只能来自 retrain 输入的行为覆盖更宽(数据集宽度),不来自单轨迹更远。⇒ 主指标 = 重试系综宽度(08-28 判读框架正确),SR 是二阶间接量。
这同时解释 mjitter 悖论(宽了但 100 场景双 FAIL):它的宽以牺牲单轨迹可行性为代价;命题 2 的斥力式换峰在理论上规避这一点——**这也是对"为什么上一代全死"的统一解释:上一代全部在单轨迹上动刀,没动系综分配**。

---

## 3. 候选机制菜单(供拍板;按改动面从小到大)

### M1 `ladder`(重试剂量梯)
- **人话**:同场景 10 条重试不再用同一剂量,排成从温和到猛的梯子(第 k 条 η_k = η_0·ρ^k)。低档重试走"近方向"(容易成功、保救回),高档重试走"远方向"(覆盖远峰、供宽度)。
- **推导**:parallel tempering 的剂量轴离散化;对 π_β = e^{β·KL} 做指数族分层。
- **实现面**:rollout/round 编排层(rescue 循环给第 k 次重试传不同 scale);policy / cost / config 零改动。CLI 例:`--retry-scale-ladder "0.5,1,2,4"`。
- **风险**:梯顶进入已知过剂量区(η3.0 时 mean_inject 26×);梯顶要设 inject 遥测上限或截断。
- **与 portfolio 的区别**:portfolio = 不同 cost 族共享 η(判负);本机制 = 同一 cost、每重试剂量显式折算——正是 portfolio 判负时留下的修法方向(当时"待批")。
- **实验**:20 场景;判读 = PR 是否随档位单调↑、总救回 ≥ 基线 4/6(pass@10 0.90 护栏)。

### M2 `novelty-rev`(互斥重试;复活 novelty)
- **人话**:第 k 条重试的 cost 加一项"离前 k−1 条已执行行为越远越好"的斥力 → 强制换方向。**这是对用户原话最直接的实现:不重走别人走过的方向。**
- **推导**:Particle Guidance 的串行化(逐粒子加斥力势);farthest-point 采样的连续版;拉格朗日形式 = 命题 2。
- **实现面**:**代码已存在**(NoveltyCostPlanner,`--guide novelty`,logKDE 对同场景已执行编码的 inter-try 斥力)。缺口:① logKDE 带宽在线适应(DvD 式);② 与 entropy cost 的求和配比(历史 combo 起点 nov0.5+att1.0@s2.0);③ **从未在 08-28 宽度框架 + 严格护栏下评估过**——它被"已弃"是在更早的 SR-first 世代。
- **风险**:串行依赖(job gate,重试必须逐条);λ 过强 → DP 一致度崩(护栏盯)。
- **实验**:20 场景三臂(novelty / combo / 基线),读 PR + 救回 + jerk + mean_inject。

### M3 `shellgrid`(壳面确定性格点)
- **人话**:shell 的随机方向 u 换成**预排的准均匀格点**(球面 Fibonacci 格或正交向量组),10 条重试把 κ-壳均匀铺满,不再随机撞重复。
- **推导**:p-dispersion / 球面 t-design 的构造版;命题 1 的确定性实现。
- **实现面**:ShellTargetCostPlanner 内 u 生成器(单函数)+ 跨重试的格点索引传递(第 k 重试用第 k 格点)。
- **风险**:壳上均匀 ≠ 编码空间均匀(非线性);格点可能正对"不动峰"(命中吸引子方向)。
- **实验**:并入 M1/M2 同矩阵对比。

### M4 `elite`(QD 档案化回灌精选;数据侧,与引导机制正交)
- **人话**:回灌前,把每个场景的多条救回重试按行为描述子(μ̄ 16 维,已有)分格,每格只保最优/最远的 1–2 条 → **同样的数据量,行为覆盖更宽**。
- **推导**:MAP-Elites 的数据侧投影;直接治 L3(retrain 输入冗余,四种子方差根因诊断的直接推论)。
- **实现面**:hdf5 组装处(merge_accumulated / success 数据写回前)加过滤器;DP 臂同样可用 → 公平对照。
- **风险**:描述子网格分辨率是超参(可用 CVT 自适应);与现行数据累积规则的交互要先审。
- **实验**:同一份 rollout 数据 ± elite 过滤 → retrain 两条 DP → 下一轮 eval 对比。不占引导线,可并行。

### M5 `langevin`(温度化注入;动前向,最后做)
- **人话**:确定性梯度注入后加一点校准噪声(+√(2τ)·ε),τ=0 逐位还原现行 → 真正"从 p\* 采样"而非"爬一个峰"。
- **推导**:p\*(理论)与 argmax(实现)之差就是温度;ULA/MALA 的 τ→0 极限;Thompson 采样精神。
- **实现面**:`guided_conditional_sample` 注入行后加一项(policy 层);`--guide-temp τ`。
- **风险**:噪声破坏 DP 去噪一致性(等效动作抖动);τ 与 η 耦合需重标定;注入路径改动需回归测试(参照 1/B 修复的 check 体系)。
- **实验**:τ∈{0, τ\*} 两臂 20 场景;τ\* 从 mean_inject 遥测定标。

### M6 `dpp-select`(DPP 选择器;暂缓)
用途在 L3 / bank 候选:多个候选(救回轨迹 / 意图 / bank 邻居)按"质量 × 多样性"联合采样子集,代替 top-k。等 M1–M4 出结果再决定是否需要。

---

## 4. 路线(2026-08-30 用户拍板后修订)

**第一波 = `particle` 单机制,三组时序消融**(实现设计见 `idea/particle_design.md`,待批):斥力分别在去噪第 0/50/90 步介入(G1/G2/G3),CAN seed233 SCOUT round0 三件套,先跑 seed42-141 定失败集(~40 场景),三组(建议 +G0 纯 entropy 对照)同失败集测 pass@10。**ladder 被用户否决**(2026-08-30:"本质上没有改变方向而是改变步长"——沿同一条棱线摆 10 个半径,PR≈1,不回答方向问题);其剂量响应曲线副产品以后需要时再测。**串行 novelty-rev 降为备胎**(particle 的在线互斥是其正统并行版)。

| 后续波次 | 内容 | 条件 |
|---|---|---|
| 第二波 | M4 `elite`(QD 档案化回灌,数据侧,与引导正交) | 可随时并行 |
| 第三波 | M3 `shellgrid-whitened` + 混合编制;M5 `langevin` | 视 particle 结果;M5 仅当系综分配仍不够时 |

**预期管理(诚实版)**:12 个 COLLAPSE(感知别名)场景预期仍不救(定理 C)。主张的胜利条件 = PR / d_act / 终态散布 ↑ **且** 救回 / pass@10 不降(严格护栏:20 场景基线救回 4/6、pass 0.90),赢家再上 100 场景复验(基线救回 19、pass 0.76–0.78)+ placebo 锚 {9,18}。

**2026-08-31 用户令**:particle 完赛后续做 `orbit`(约束控制,见 §6;#math 会话定稿)——在 entropy-random-dev 上与 particle 共存(互不改动),先在 SQUARE seed233 上测。

---

## 6. 约束控制(orbit,2026-08-31 拍板;#math 会话 msg 42 定稿)

一句话:**爬坡是 argmax 控制(峰形奖励→单峰收敛);把奖励换成「到达 {f≥κ} 即满分」的平台形,最优控制就变成壳上巡行的约束动力学——法向反馈扶 κ + 切向噪声走方向**。单轨迹层面换控制律,不依赖粒子间耦合。

**推导链**(理论目标 → 实现公式;每步标注性质):

1. 【恒等|控制论】KL-control / path-integral control(Kappen 2005;Todorov 2006;Dvijotham UAI 2011):扩散过程在 KL 正则控制 min E[c + (1/η)KL(u‖u₀)] 下,HJB 经 log 变换 V=e^{−Ψ/η} 线性化,最优控制场 **u\*(x) = ησσᵀ∇log V(x)**——控制 = 势场的分数流。
2. 【代理|现有注入的定位】classifier guidance 注入的 ∇f(f = KL(q(z|s̄,a)‖q(z|s̄,a⁰)))即 ∇log V 的贪心一步近似 = **argmax 控制**;奖励 r=f(峰形,越大越好)⇒ 模式坍缩到 cost 景观单峰(窄锥三因子之①的正式说法)。
3. 【恒等|重定义奖励】换平台形奖励 r = 1{f≥κ}(到达即满分)⇒ 最优控制在壳 {f=κ} 邻域 = **约束保持**。与 entropy cost 推导同源(同一 KL 正则变分界),叙事 = 「从 argmax 控制到约束控制」。
4. 【文献|壳上巡行结构】约束流形采样:Zappa Holmes-Cerfon & Goodman《Monte Carlo on Manifolds》(2018,约束马尔可夫链);Holmes-Cerfon 2024(高维);约束 Langevin + Lagrange 乘子(Leimkuhler–Matthews;离散化血统 = 分子动力学的 SHAKE/RATTLE)。机制 = **法向反馈(把 f 扶回 κ)+ 切向自由扩散(方向覆盖的唯一来源)**。
5. 【代理|实现公式】每引导去噪步按行二选一(δ = 相位切换缓冲,默认 0.25κ):
   - f(x̂₀) < κ−δ:照旧爬坡 `η√(1−ᾱ_t)·∇f`(逐字节 = atypical);
   - f(x̂₀) ≥ κ−δ:`Δx_t = −λ·(f−κ)·g/‖g‖² + σ_orb·√(1−ᾱ_t)·ξ⊥`,其中 g=∂f/∂x_t(未封顶 KL),ξ⊥=ξ−(ξ·ĝ)ĝ。
6. 【恒等|标度性质】**Newton 项重参数不变**:x̂₀=(x_t−√(1−ᾱ)ε̂)/√ᾱ ⇒ ∂f/∂x_t = (∂f/∂x̂₀)/√ᾱ,代入得 −λ(f−κ)√ᾱ·g₀/‖g₀‖²——x_t 空间算出的 Newton 步恰是 x̂₀ 空间的 Newton 步,一阶把 f 精确投回 κ(步后 f′ ≈ f − λ(f−κ);λ=1 一步到壳,λ<1 阻尼)。**λ 无量纲**(0,1] 松弛因子,不继承 η 的 VIB 梯度尺度 ⇒ 跨任务免 η 型单位换算;σ_orb 仍按遥测定标)。切向噪声乘 √(1−ᾱ_t) = 与注入约定一致、随 DDPM 过程噪声退火。
7. 【机制|与 particle 的关系】phase 2 **替代而非叠加**爬坡/斥力:方向多样性来自约束采样结构,不来自系综互斥;**过剂量悬崖结构性消失**(feedback 永不推过 κ——超壳自动回拉,不存在「越推越远」的正反馈)。反馈项在 κ 下方仍沿 +g 爬(−λ(f−κ)>0)⇒ 交接无缺口、无双份剂量。

**可证伪签名**:重试终态沿等值面散布(f 终值方差小、方向方差大 → PR↑);救回 ≥ 基线(orbit 前阶段 ≡ 现行爬坡,下界 = atypical);COLLAPSE 12 场景(定理 C)仍不救。

**工程指针(一行)**:实现 = `scout/guidance/orbit_costs.py`(钩子 = `OrbitCostPlanner.orbit_update` 挂 `policy.py` 注入行;`orbit_displacement` 纯函数供单测);CLI = `--guide orbit --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25`(no-op 哨兵 (0,0,0) ≡ atypical 位同;验证 = `scout/guidance/_verify.py` check 10-12;commit `f639e4b`)。

---

## 7. 上升射线族(ray,2026-08-31 #math msg 59 定稿;beat-SOE 轮的备选迭代)

一句话:**爬升段(phase 1)多样性的根级修法:每条重试沿一条钉死的直线方向 û_k 爬 KL——rank-1 投影场保证 f 无条件单调,方向永不旋转 → 漏斗(最速流方向坍缩)无处作用;到 κ−δ 交接 orbit phase-2**。用户 2026-08-31 指示:本轮 beat-SOE campaign 有 idea 被毙后试这个。

**推导链**(理论目标 → 实现公式;每步标注性质):

1. 【问题|窄锥根源在 phase-1 的正式说法】orbit 只治了 phase-2(壳上巡行),phase-1 仍逐字节 = atypical:每条重试走同一条最速方向 ∇f/‖∇f‖(argmax 控制)。漏斗定理:最速流的方向坍缩率 e^{(λ₁−λ₂)t},任何 O(ε) 扰动场(wedge 倾斜 / deflect 重抽 / langevin 小噪)都被指数压死——这就是爬升段多样性方案屡毙的共性死因。
2. 【恒等|rank-1 投影场单调性】对任意固定单位方向 û,定义 v_û(x)=⟨∇f(x),û⟩·û ⇒ df/dt=(⟨∇f,û⟩)²≥0。对任何 C¹ 的 f、任何地形、任何 û 成立(方向错了只是停,不会降)。
3. 【恒等|幅度恢复】朴素投影的注入强度折损 |cos θ|;改 v=‖∇f‖·sgn(⟨∇f,û⟩)·û ⇒ df/dt=‖∇f‖·|⟨∇f,û⟩|≥0——满强度注入、单调性保持。
4. 【恒等|射线定理(漏斗免疫)】沿线 δ=s·û 二次地形 f(s)=½s²(ûᵀHû) 严格单调升;κ-壳穿越半径 r(û)=√(2κ/(ûᵀHû));方向钉死不旋转 ⇒ 坍缩机制没有可作用的对象。
5. 【代理|族构造】γ₀ = 现行 atypical(û=归一化 ∇f,保住 maximize 主路径 + 完备性护栏:K 条全废时下界=atypical);γ_k(k=1..K−1)方向 = 白化球面**确定性**设计(Fibonacci 格 / 球面 t-design / 归一化高斯+max-min 筛),种子与 TSEED 可复现约定一致。方向空间第一版 = **轨迹空间**(80 维/chunk),z 空间(16 维)备选。
6. 【恒等|高维红利】两随机单位向量 E[cos²]=1/d(80 维 ≈1.25%)⇒ 射线间天然近正交,代价 = 进展率 ×|cos θ|。
7. 【机制|与 orbit 分工】f≥κ−δ → orbit 接管(已实现,`orbit_costs.py`);ray 只改 phase-1 的方向分配,注入语法(guidance 行)不变。**两主旨**:爬升目标仍是 z 的 KL divergence(主旨①);K 条射线 ≈ K 个方向,PR 1.30→≈K(主旨②)。

**可证伪定理**(#math 原文):T1 单调(executed-KL 非降率 ≥70%;非单调源 = chunk 闭环漂移,用 orbit 已有 executed-KL 遥测监控);T3 分散(PR→≈K);T4 覆盖(K=10 时 max-min 角 covering radius 较粗——诚实账)。上升证书:注入幅度 = ‖∇f‖·|⟨∇f,û_k⟩| 逐射线遥测;离散步长的 Armijo 型条件由 DDPM √(1−ᾱ_t) 退火天然满足。

**文献血统**:Nesterov–Spokoiny 随机方向法(rank-1 方向导数);球面 t-design / Fibonacci 格(确定性铺方向);Torczon pattern search(坐标射线搜索);GAD/dimer(固定方向二阶鞍点动力学)。

**工程指针(一行)**:尚无实现(#math todo「拍板后文件级设计」未做);落点 = `OrbitCostPlanner` phase-1 的方向分配(tries k≥1 换 û_k,γ₀ 不动),钩子/语法与 orbit 同源。

---

## 5. 参考文献清单

**A 温度化采样**
- Thompson sampling tutorial: https://arxiv.org/abs/1707.02038
- Parallel tempering(Wikipedia): https://en.wikipedia.org/wiki/Parallel_tempering
- MCMC for multimodal targets 综述(tempering / mode jumping / Wang–Landau): https://arxiv.org/abs/2501.05908
- Woodard et al., rapid mixing conditions for PT: https://people.orie.cornell.edu/woodard/rapidMixTemper.pdf

**B 粒子斥力**
- Particle Guidance(Corso et al., ICLR 2024): https://arxiv.org/abs/2310.13102 | code: https://github.com/gcorso/particle-guidance
- Repulsive Score Distillation: https://huggingface.co/papers/2406.16683
- SVGD(Liu & Wang 2016);Annealed SVGD: https://arxiv.org/abs/2101.09815
- DvD(NeurIPS 2020): https://arxiv.org/abs/2002.00632 | code: https://github.com/jparkerholder/dvd_es
- DPP Batch BO(Kathuria et al., NeurIPS 2016): https://proceedings.neurips.cc/paper/2016/hash/a1d7311f2a312426d710e1c617fcbc8c-Abstract.html
- DPP-BBO(Nava et al., AISTATS 2022): https://arxiv.org/abs/2110.11665
- Determinantal Beam Search(ACL 2021): https://arxiv.org/abs/2106.07400

**C 覆盖式优化**
- MAP-Elites(Mouret & Clune 2015);QD 入口: https://members.loria.fr/jbmouret/qd.html ;论文列表: https://quality-diversity.github.io/papers.html
- Go-Explore: https://www.uber.com/us/en/blog/go-explore/
- Ergodic exploration(Mathew & Belta, ACC 2012): https://robotics.northwestern.edu/documents/publications/ACC2012.pdf ;tutorial: https://ergodiccontrol.github.io/

**D 确定性散布**
- p-dispersion IP 解法(EJOR 2016): https://www.sciencedirect.com/science/article/abs/pii/S0377221716300637
- 离散多样性/分散度综述(EJOR 2021): https://www.sciencedirect.com/science/article/pii/S0377221721006548

**E 本地记录**
- 上一代 11 臂方法与结果: `soe_scripts/rand_ideas/NAMES.md`
- 熵 cost 权威推导: `idea/entropy_cost.md`
- 窄锥 vs 喷雾定量闭环、宽度指标框架: memory `scout-vs-soe-exploration-diversity` / NAMES.md 判读框架(2026-08-28)
