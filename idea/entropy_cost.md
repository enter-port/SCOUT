# Entropy Cost：设计意义、实现计算与推导

> 组织方式仿 DIAYN（Eysenbach et al. 2019）第 3 节的推导体例：先给出目标的表达形式，再逐条说明为什么它不可直接计算、每一步做什么替换，最后落到可执行的计算式。每行公式后注释符号；每步标注【恒等式】/【变分界】/【代理】/【定义】。
> 实现：`scout/guidance/entropy_costs.py`（KLCostPlanner，2026-09-04 由 AtypicalCostPlanner 更名；orbit = 其子类 `scout/guidance/orbit_costs.py`）、注入 `scout/guidance/policy.py`。

---

## §1 设计意义与目标的形式表达

**问题**：失败场景的 ×10 重试采样中，冻结 DP 自身的动作分布在这些场景上已经收敛——直接重采样等价于重复同类失败。重试需要的是**策略自身不会做的动作**。

**度量的载体**：判断"这个动作与策略常规行为差多少"，需要有一个把动作映射为行为摘要的模型。SCOUT 的 VIB 动力学模型提供了它：编码器 $q_\phi(z\mid\bar s, a)$ 把（观测摘要，动作块）映射为技能潜变量 $z$ 的分布，且 $z$ 经训练能决定下一潜态（§2.1）。于是"行为差异"可以在 $z$ 空间中度量。

**目标的形式表达**（探索的信息论目标，与 DIAYN 同源）：

$$\max\; I(Z;\,S') \;=\; H(S') \;-\; H(S'\mid Z) \tag{1}$$

- $I(Z;S')$：技能潜变量与下一状态间的互信息；
- $H(S')$：下一状态的边缘熵——重试应访问**多样的未来**（探索的实质）；
- $H(S'\mid Z)$：给定技能潜变量的条件熵——未来仍由技能**决定**（行为的连贯性）；
- 【恒等式】互信息的熵分解。

式 (1) 的两半在系统里分工：$H(S'\mid Z)$ 由 **VIB 训练期**压低（解码器确定性回归，§2.1）；推理期要最大化的是 $H(S')$。下一节先给出我们最终实际计算的量，§3 再从式 (1) 走到那里。

## §2 实际计算的是什么

§1 的式 (1) 无法直接计算（原因见 §3 Step 1）。实际实现中，推理期在**每个动作块**上计算的是：

$$C(\hat x_0) \;=\; -\min\Big(\mathrm{KL}\big(q_\phi(z\mid\bar s,\,a)\;\big\|\;q_\phi(z\mid\bar s,\,a^0)\big),\;\kappa\Big),\qquad a=\mathrm{bridge}(\hat x_0) \tag{2}$$

- $\hat x_0$：去噪循环当前步的干净动作估计（DP 归一化空间，8 步 × 10 维）；
- $a$：经 bridge（归一化逆变换，可微）回到编码器训练时见过的原始动作向量（80 维）；
- $\bar s$：当前观测摘要 $\bar s=E_s(o)$（1088 维），本块内缓存复用；
- $a^0$：**本块第一步**、去噪轨迹尚未被修改时 DP 的无引导意图动作（基线，每块捕获一次）；
- $q_\phi(z\mid\bar s,\cdot)$：VIB 编码器，输出对角高斯 $\mathcal N(\mu_\phi,\mathrm{diag}\,\sigma_\phi^2)$，$z$ 为 16 维；
- $\kappa=2.5$ nats：KL 上限；
- 负号：注入路径统一做代价的梯度下降，即 KL 的梯度上升（封顶前）。

对角高斯间的 KL 有闭式解【恒等式】：

$$\mathrm{KL}(q_a\|q_{a^0}) = \frac12\sum_{i=1}^{16}\Big[ \frac{(\mu_i-\mu^0_i)^2}{\sigma^{0\,2}_i} + \frac{\sigma^2_i}{\sigma^{0\,2}_i} - 1 - (\mathrm{logvar}_i - \mathrm{logvar}^0_i) \Big] \tag{3}$$

- $(\mu_i,\sigma^2_i,\mathrm{logvar}_i)$：候选动作 $a$ 处后验的均值、方差、对数方差（第 $i$ 维）；
- $(\mu^0_i,\sigma^{0\,2}_i,\mathrm{logvar}^0_i)$：意图动作 $a^0$ 处后验的对应量；
- 四项分别来自一般高斯 KL 闭式解的迹项、均值二次项、维数常数、对数行列式项；
- 注意均值差按 $1/\sigma^{0\,2}_i$ 加权（马氏距离）：意图后验越确定的维度，偏离被罚得越重。

### 2.1 编码器 $q_\phi$ 怎么训出来的（cost 的信号来源）

$$\min_\phi\; \mathbb{E}\Big[\big\|D_\psi(z,\bar s_t)-E_s(o_{t+1})\big\|^2\Big] + \beta\,\mathrm{KL}\big(q_\phi(z\mid\bar s_t,a_t)\,\big\|\,\mathcal N(0,I)\big),\qquad z=\mu_\phi+\sigma_\phi\odot\epsilon,\ \epsilon\sim\mathcal N(0,I) \tag{4}$$

- $D_\psi$：解码器，从 $(z,\bar s_t)$ 预测下一观测摘要 $\bar s_{t+1}=E_s(o_{t+1})$（target 停梯度）；
- $E_s$：冻结的 base-DP 视觉前端；
- $\beta$：信息瓶颈权重（$3\times10^{-5}$，坍缩扫描定标）；
- $\epsilon$：重参数化噪声；
- 第一项使 $z$ 携带"决定未来"的信息（压低 $H(S'\mid Z)$，§1 的另一半）；第二项把后验拉向先验；
- **关键是第一项给了 $q_\phi$ 对 $a$ 的敏感性**：动作改变未来 ⇔ 编码器必须把该改变编码进 $z$ 的后验——这是式 (3) 能作为行为差异度量的前提，也是 ∂C/∂a 非零的来源。

### 2.2 cost 如何进入采样

$$x_t \leftarrow x_t-\eta\sqrt{1-\bar\alpha_t}\,\frac{\partial}{\partial x_t}\sum_{b=1}^{B}C\big(\hat x_0^{(b)}(x_t)\big),\qquad \eta=3.0 \tag{5}$$

- $x_t$：去噪中间变量；$\bar\alpha_t$：DDPM 累积噪声系数；
- $\sqrt{1-\bar\alpha_t}$：随去噪进行衰减到 0 的缩放（Dhariwal & Nichol 2021 引导项形式，LPB 实现模板）；
- $\sum_b$：批内各行求和（sum 归约，保证每行梯度不被并发数 $B$ 稀释）；
- 梯度路径 $x_t\to\hat x_0\to a\to(\mu,\mathrm{logvar})$，全程可微。

引导后的隐式采样分布【标准结果，见 §3 Step 6】：

$$p_{\text{guided}}(x)\ \propto\ p_{\text{DP}}(x)\cdot\exp\big(\min(\mathrm{KL}(x),\,\kappa)\big) \tag{6}$$

## §3 从式 (1) 到式 (2)：逐步推导

**Step 1｜式 (1) 为什么算不了。**
$H(S')=-\int p(s')\log p(s')\,ds'$ 需要未来状态的密度 $p(S')$。
- $p(S')$：下一状态的分布密度——在图像观测下这正是 world model 要学的东西，本设计刻意不建（项目论证过的首要风险）；
- $H(S'\mid Z)$ 同理需要条件密度。
结论：【代理的入口】推理期不能优化任何直接含 $p(S')$ 的量，必须找一个只经已训模型即可计算、且与 $H(S')$ 单调相关的替代量。

**Step 2｜影响必经后验——把"未来多样"换成"后验移动"。**
VIB 的生成链是

$$p(\hat s'\mid a,\bar s)=\int q_\phi(z\mid\bar s,a)\,\delta\big(\hat s'-D_\psi(z,\bar s)\big)\,dz \tag{7}$$

- $\delta$：Dirac 测度——解码器 $D_\psi$ 确定；
- 【恒等式】确定性解码下，未来分布 = 后验的推前（pushforward）。

由式 (7) 立即得【恒等式引理】：若 $q_\phi(z\mid\bar s,a)=q_\phi(z\mid\bar s,a^0)$，则 $p(\hat s'\mid a)=p(\hat s'\mid a^0)$。
即：**后验不动的动作不可能改变未来分布**，对 $H(S')$ 的增量为零。于是"最大化未来熵"的可计算代理是"最大化后验相对当前意图的移动量"。$a^0$ 取 DP 自身意图，因为目标是"做策略不会做的事"（§1）。

**Step 3｜把"移动量"定成信息量——DIAYN 引理。**
度量两个后验的差异，自然单位是 KL；KL 又恰是互信息的变分下界里的量（DIAYN Lemma 1 的形式）：

$$I(Z;X)\;\ge\;\mathbb E_x\,\mathbb E_{z\sim q_\phi(\cdot\mid x)}\Big[\log q_\phi(z\mid x)-\log p(z)\Big] \tag{8}$$

- $X$：被区分的对象，DIAYN 取 $x=s$（技能由状态判别）；此处取 $x=(\bar s,a)$——编码器的条件本就含动作；
- $p(z)=\mathcal N(0,I)$：先验（式 (4) 的 KL 项保证它是合法参照）；
- 【变分界】等号当 $q_\phi$ 为真后验时成立；$q_\phi$ 与真后验的 KL 是被丢掉的松驰量；
- 单个样本的判别奖励 $r(a,z):=\log q_\phi(z\mid\bar s,a)-\log p(z)$ 的期望 $\mathbb E_{z\sim q_a}[\,\cdot\,]=\mathrm{KL}(q_a\|p)$：该动作的后验离先验多远（nats）。

**Step 4｜相对化——参照系从先验换成自身意图（得到式 (2) 的 KL）。**
式 (8) 的绝对量不问语境：推向"先验典型动作"。救援语境要求的是相对差。对**同一个** $z$ 取两个动作的奖励之差：

$$\Delta r(z)=r(a,z)-r(a^0,z)=\log q_\phi(z\mid\bar s,a)-\log q_\phi(z\mid\bar s,a^0) \tag{9}$$

- 【恒等代数】$\log p(z)$ 两项相消——先验在差中不出现；
- $\Delta r(z)>0$：在编码器看来，这个 $z$ 支持"动作是 $a$ 而非 $a^0$"的证据量。

在 $z\sim q_a$ 下取期望【恒等式】：

$$\mathbb E_{z\sim q_a}[\Delta r(z)]=\mathrm{KL}\big(q_a\,\|\,q_{a^0}\big) \tag{10}$$

- 这正是式 (2) 中的 KL。**与"两个 KL 的差"的关系**：把式 (10) 的比值拆向先验得 $\mathrm{KL}(q_a\|q_{a^0})=\mathrm{KL}(q_a\|p)-\mathbb E_{z\sim q_a}[\log\frac{q_{a^0}}{p}]$，当两后验接近时第二项 ≈ $\mathrm{KL}(q_{a^0}\|p)$，故 $\mathrm{KL}(q_a\|q_{a^0})\approx\mathrm{KL}(q_a\|p)-\mathrm{KL}(q_{a^0}\|p)$（候选的绝对不典型度减意图的绝对不典型度）【近似，测度交换，未采用】。实现直接算式 (10)：它是恒等式，且式 (3) 的马氏加权梯度性质更好。

**Step 5｜封顶与先验——为什么是 $-\min(\cdot,\kappa)$ 而不是 $-\,\mathrm{KL}$。**
无界 KL 的风险：式 (3) 中 $1/\sigma^{0\,2}_i$ 加权在 $\sigma^0\!\to\!0$ 的维度上发散，且推力随偏离增大而不减，会把动作推出可行域。两个信任域：
- $\kappa=2.5$：单块 KL 达到 $\kappa$ 后梯度为零（式 (2)），对散度型目标设上界是 PPO（Schulman et al. 2017）clip 的结构先例；
- DP 先验 $p_{\text{DP}}$：式 (5)(6) 中 DP 始终是采样分布的主体，似然权重至多放大 $e^\kappa\approx 12.2$ 倍。

**Step 6｜落地为可微计算并注入。**
式 (3) 只依赖编码器一次前向（候选）+ 一次缓存前向（意图），闭式、可微；经 bridge 与 $\hat x_0(x_t)$ 链式反传（式 (5)）。加性引导项使反向过程渐近采样 $p_{\text{DP}}\cdot f$，其中 $f=\exp(\min(\mathrm{KL},\kappa))$（classifier guidance 的标准结果，Dhariwal & Nichol 2021），即式 (6)。推导完毕。

## §4 对照：三篇论文各自的推导写法与本推导的对应

| 论文 | 推导体例 | 与本文的对应 |
|---|---|---|
| **DIAYN**（Eysenbach 2019, §3） | 目标 $I(S;Z)=H(S)-H(S\mid Z)$ → 四条简化假设（固定先验 / 变分后验代真后验 / 不显式最大化策略熵 / 限制状态访问）→ 引理给出下界 $\mathbb E[\log q_\phi(z\mid s)-\log p(z)]$ → 实际目标 $J(\pi,q_\phi)$ | 本文 §1 式 (1) 即其目标式；Step 3 式 (8) 即其引理；差异：DIAYN 在**训练期**优化 $q_\phi$ 与策略，本文 $q_\phi$ 冻结、在**推理期**用同一下界的逐动作差分（Step 4）作探索信号 |
| **SOE**（arXiv:2509.19292） | 损失 $\mathcal L_{IL}+\mathcal L_{IB}$：模仿项 + 信息瓶颈项（$-\log q_\phi(a\mid z)$ 的扩散重建 + $\beta\,\mathrm{KL}[p_\theta(z\mid o)\|r(z)]$，后验只条件于观测）；测试期在潜空间加噪探索 | 对照点：SOE 的 KL 是**后验到先验**（训练正则）；本文的 KL 是**后验到意图后验**（推理探索信号）。SOE 扰动 $z$ 本身，本文经 $\partial q_\phi/\partial a$ 在动作空间引导 |
| **LPB**（Sun & Song 2025, arXiv:2508.05941） | 无 MI 推导：直接定义 cost（候选编码到 expert 潜流形的 NN 距离），在冻结 DP 去噪循环按 $\hat x_0$ 算 cost、对 $x_t$ 求梯度、乘 $\sqrt{1-\bar\alpha_t}$ 注入 | 本文 Step 6 的注入机制（式 (5) 的形式、sum 前的归约约定、缩放因子）完整沿用其实现；cost 定义换成式 (2) |

## §5 逐步性质总结

| 步 | 内容 | 性质 |
|---|---|---|
| (1) | $I=H(S')-H(S'\mid Z)$ | 恒等式 |
| Step 1 | $p(S')$ 不可得 → 需代理 | 代理入口 |
| (7) | 未来分布 = 后验的推前 | 恒等式（$D_\psi$ 确定） |
| Step 2 | 后验不动 ⇒ 未来分布不动 | 恒等式引理 |
| (8) | $I\ge\mathbb E[\log q_\phi-\log p]$ | 变分下界（DIAYN） |
| (9)(10) | 同一 $z$ 的奖励差取期望 = 后验 KL | 恒等代数 |
| (2) | $-\min(\mathrm{KL},\kappa)$ | 定义（封顶信任域） |
| (3) | 对角高斯闭式解 | 恒等式 |
| (5)(6) | $\sqrt{1-\bar\alpha_t}$ 注入 ⇒ $p_{\text{DP}}\cdot e^{\min(\mathrm{KL},\kappa)}$ | 标准结果 |

代码锚点：式 (2)(3) → `entropy_costs.py:188-208`；基线捕获式 (9) 的 $a^0$ → `entropy_costs.py:178-186`；式 (4) → `scout/train_vib.py`（VIB 训练）；式 (5) → `policy.py:236-265`；bridge → `scout/normalizer.py:57`。
