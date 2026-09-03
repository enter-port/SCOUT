# rand ideas — SOE 机制忠实移植角度(agent 头脑风暴 2026-08-27,B=covtilt 已派测)

## SOE 六成分拆解(代码取证)
- C1 学习的潜空间几何:down-module 映到 (μ_θ(o),σ_θ(o)),σ 逐维由 KL+ext_loss 权衡训出(dp_ext.py:72-74,119-122)。
- C2 δ 采样点:z=μ_θ+α·σ_θ⊙ε,**是点不是分布**,每 forward 重采(131-143)。
- C3 decoder 可行化(最关键):z 经 up-module 回 readout 整体替换条件,diffusion 在扰动条件上解码;ext_loss 专门训过"带噪 z→demo 动作"→噪声方向天然落在行为流形切空间(149-153,168-170)。**全程无梯度、无 encoder Jacobian**。
- C4 CADS 去噪期模态噪声:每 denoise step 对条件重加噪 noise_scale·ε·(1−γ(t)),γ 线性退火(diffusion.py:170-185)。
- C5 动作尾巴噪声(206-207)。
- C6 维度门控/覆盖设计:std_mask 逐维开噪(vdp.py:123-127);uniform_exploration:linspace(−2,2) 沿 batch 主动铺开(dp_ext.py:135-137)。

**结构不对称 = 今天方案A 的死因**:我们的随机性以 cost 形式过 a→μ 的低秩 Jacobian J;SOE 噪声直接进条件、由被训过的 decoder 解码,不经过任何 J。**移植原则:随机对象要么放动作空间(cost 线性于 b(x̂₀)),要么用数据变异方向替代 σ⁰ 球面方向。**

## B covtilt(已派测,主推)
每 chunk 取 s̄ 的 K 近邻专家 chunk,C_e=Cov(近邻)+εI;每 (i,k) 采 ξ~N(0,C_e) 定住;cost=−ξᵀ·b(x̂₀)·1[KL(q(z|s̄,b(x̂₀))‖q⁰)≤κ]。梯度 −ξ 全在动作空间,绕开 J。VIB 从发散力改为刹车。倾斜方向=专家真实变异轴 → exp(ξᵀa)·p_DP 推向 DP 够得着但不常选的模态。扫 K(10/25/50)、ε、κ gate、η。失败:K 小→C_e 退化≈均匀噪声;gripper 维方差虚高;η 大出支持集。

## C anchor(第二波)
j(i,k) 在 top-M 近邻里按 (rand_seed,i,k) 选 a*=第 j 近邻的 chunk,整重试定住;cost=½‖b(x̂₀)−a*‖²_{C_e^{-1}}·1[KL(q(z|s̄,a*)‖q⁰)≥τ]。与旧 expert-zbank 的差别:专挑 atypical 邻居(非 NLL 最小),目标在动作空间。失败:状态失配不可达;can 20 demo 模态数少 → 3 次重试饱和。

## D drift(CADS 消融)
ξ_t=ρξ_{t−1}+√(1−ρ²)ε_t(OU),cost_t=−(1−γ_t)ξ_tᵀb(x̂₀),γ=cads 线性退火。退火 Langevin 式探索。ρ=0 净推力≈0。分布探索非模态瞄准,单重试成功率低、靠 pass@10。

## E cover(覆盖设计,包在 B/C 外)
C_e 谱分解前 m 维,ξ_k=Ē^{1/2}h_k/‖·‖,h_k=正交设计第 k 行(Hadamard/GS)。10 retry 系统性扫过数据子空间 10 个正交方向(SOE uniform_exploration 的 cost 化 + 修正 iid 碰撞)。m 由谱肘定;可留 1-2 retry 纯方案三保底。

## F dynsig(几何插件)
g_i=‖∂D_s(z,s̄)/∂z_i‖|_{μ⁰}(16 次 VJP,近似免费),σ*_i=σ⁰_i/ĝ_i,cost=−min(KL(q_a‖q⁰;Λ←diag(1/σ*²)),κ)。学习的"行为重要度"代餐;z 空间内重加权不改 J 的秩——独立跑=消融,主用途是给 B/E 提供逐维几何。

## 三个关键回答
1. σ⁰ 之外的流形几何:(i) 专家 chunk 协方差 C_e(动作空间,唯一绕开 J);(ii) D_s 敏感度 g(学习版前向模型几何);(iii) 专家 μ 协方差(仍过 J,仅消融)。主用 (i)。
2. δ 先于高斯:方案A 已证高斯目标的方差项无方向信息;SOE 本尊对象就是 δ 点,"分布"只在重试系综层;软目标(−logΣexp 混合邻居)留第二波。
3. 重试内连贯:默认逐重试固定随机对象("一次重试=一个行为假设");SOE 忠实版=逐 chunk 重采+OU(D 的消融);临近成功衰减调度还控制权给 DP。

**试验顺序:B 先行,B+E(cover)是冲 pass@10>0.85 主线;C 第二波;D 连贯性消融;F 度量升级。**
