# Entropy Cost：从 I(Z;S′) = H(S′) − H(S′|Z) 到实现公式的推导

> 每行公式后给出符号说明；每一步标注它是**恒等式**、**变分界**还是**代理近似**。
> 实现代码：`scout/guidance/entropy_costs.py:163-208`（AtypicalCostPlanner）、注入 `scout/guidance/policy.py:236-265`。

## 0. 符号

| 符号 | 意义 |
|---|---|
| $S_{t+1}$（下文 $S'$） | 下一时刻状态（严格说是其潜表示 $\bar{s}_{t+1}=E_s(o_{t+1})$） |
| $Z$ | 技能潜变量，16 维 |
| $a$ | 一个动作块的原始动作向量（8 步 × 10 维，展平 80 维） |
| $a^0$ | 同一状态下冻结 DP 自身（无引导）意图动作块 |
| $q_\phi(z\mid \bar{s},a)$ | VIB 编码器，对角高斯 $\mathcal{N}(\mu_\phi,\mathrm{diag}\,\sigma_\phi^2)$ |
| $D_\psi$ | VIB 解码器，$D_\psi(z,\bar{s})\to\hat{s}'$，确定性映射 |
| $p(z)$ | $Z$ 的先验（训练目标收敛到 $q_\phi\!\to\!\mathcal{N}(0,I)$） |
| $\bar{s}=E_s(o)$ | 冻结视觉前端输出（1088 维） |

---

## 1. 目标：互信息及其两半

$$\max_{a}\; I(Z;\,S') = H(S') - H(S'\mid Z)$$

- $I(Z;S')$：$Z$ 与下一状态间的互信息——"技能潜变量对未来的掌控量"；
- $H(S')$：下一状态的边缘熵——访问多样的未来（探索要的东西）；
- $H(S'\mid Z)$：给定潜变量的条件熵——给定 $z$ 未来应可预测；
- 此式为**恒等式**（互信息的熵分解）。

**两半的分工**：
- $-H(S'\mid Z)$ 半边由 VIB **训练期**处理：$D_\psi$ 对 $E_s(S_{t+1})$ 做确定性回归，给定 $z$ 的未来方差被压低。推理期不再触碰。
- $H(S')$ 半边是推理期要最大化的——但 $H(S')$ 需要未来状态的密度 $p(S')$，**不可得**（有它就等于有了 world model，正是本设计刻意绕开的）。因此第 2 步起全部是对 $H(S')$ 的代理。

## 2. 影响必经后验：$a \to q(z|\bar s,a) \to z \to D_\psi \to \hat s'$

生成链为

$$p(\hat{S}'\mid a,\bar{s}) = \int q_\phi(z\mid \bar{s},a)\,\delta\!\big(\hat{s}'-D_\psi(z,\bar{s})\big)\,dz$$

- $q_\phi(z\mid\bar s,a)$：编码器在动作 $a$ 处的（对角高斯）后验；
- $D_\psi(z,\bar{s})$：确定性解码器；
- $\delta$：Dirac 测度——$z$ 确定地决定 $\hat s'$；
- 此式为**恒等式**（确定性解码下未来分布 = 后验的推前）。

推论（恒等）：若 $q_\phi(z\mid\bar s,a)=q_\phi(z\mid\bar s,a^0)$，则 $p(\hat S'\mid a)=p(\hat S'\mid a^0)$。
即：**不移动后验的动作不可能改变引导后的未来分布**。所以"该动作对 $H(S')$ 有任何贡献"的必要条件是它移动后验；移动量就是可计算的代理量。这一步是**代理近似**的入口：用"后验移动量"代替不可得的 $H(S')$ 增量。

## 3. DIAYN 变分下界：判别奖励

互信息的变分下界（DIAYN 引理）：

$$I(Z;X) \;\ge\; \mathbb{E}_{x}\,\mathbb{E}_{z\sim q_\phi(z|x)}\Big[\log q_\phi(z\mid x) - \log p(z)\Big]$$

- $X$：这里取 $x=(\bar s,a)$（动作与状态一起作为被判别对象）；
- $\log q_\phi(z\mid x)-\log p(z)$：单个样本的判别项——后验对数似然减先验对数似然；
- 不等号来源：$q_\phi$ 是真后验的变分近似，其与真后验的 KL 作为松驰量被丢掉（**变分界**）；
- 期望下每一项 $\mathbb{E}_{z\sim q_\phi(\cdot|x)}[\log q_\phi(z|x)-\log p(z)] = \mathrm{KL}\big(q_\phi(\cdot|x)\,\|\,p\big)$，即"该动作的后验离先验多远"（nats）。

定义**每动作判别奖励**：

$$r(a,z) \;=\; \log q_\phi(z\mid \bar{s},a) - \log p(z)$$

- $z$：从 $q_\phi(\cdot\mid\bar s,a)$ 采样的潜变量；
- $r$：DIAYN 目标的单样本形式——"这个动作的 $z$ 有多不先验典型"。

## 4. 相对化：以策略自身意图为参照（关键一步）

绝对判别（直接最大化 $\mathrm{KL}(q_\phi(\cdot|\bar s,a)\|p)$）与当前语境无关：它推向"先验典型动作"，不问 DP 在此块本会做什么。救援场景需要的是"**做 DP 在这里不会做的事**"。因此把同一个 $z$ 下的两个奖励作差：

$$\Delta r(z) \;=\; r(a,z) - r(a^0,z) \;=\; \log q_\phi(z\mid \bar{s},a) - \log q_\phi(z\mid \bar{s},a^0)$$

- $a$：候选动作（去噪中的 $\hat x_0$ 经 bridge 映回原始动作空间）；
- $a^0$：DP 无引导意图动作（该块去噪第一步、未被修改时捕获）；
- $\log p(z)$：两项相消——先验在差中不出现，参照系从先验换成了意图后验；
- $\Delta r(z)$：同一 $z$ 下，候选动作相对意图动作的判别优势。

在 $z\sim q_\phi(\cdot\mid\bar s,a)$ 下取期望（**恒等代数**）：

$$\mathbb{E}_{z}[\Delta r(z)] \;=\; \mathbb{E}_{z}\Big[\log\frac{q_\phi(z\mid\bar s,a)}{q_\phi(z\mid\bar s,a^0)}\Big] \;=\; \mathrm{KL}\big(q_\phi(z\mid\bar s,a)\,\big\|\,q_\phi(z\mid\bar s,a^0)\big)$$

- 期望测度：候选动作自己的后验 $q_\phi(\cdot\mid\bar s,a)$；
- 右端即**实现中的那个 KL**（`entropy_costs.py:200-201`）。

**与"两个 KL 散度之差"的关系**。把比值拆向先验：

$$\mathrm{KL}(q_a\|q_{a^0}) \;=\; \underbrace{\mathbb{E}_{z\sim q_a}\big[\log\tfrac{q_a(z)}{p(z)}\big]}_{=\ \mathrm{KL}(q_a\|p)\ \text{(候选后验到先验)}} \;-\; \underbrace{\mathbb{E}_{z\sim q_a}\big[\log\tfrac{q_{a^0}(z)}{p(z)}\big]}_{\text{意图后验对数比的错测期望}}$$

- $q_a := q_\phi(\cdot|\bar s,a)$，$q_{a^0} := q_\phi(\cdot|\bar s,a^0)$；
- 第一项：候选后验到先验的 KL（DIAYN 绝对量）；
- 第二项：意图后验的对数比在**候选**测度下的期望——若两个后验接近，可用它自己的测度近似 $\mathbb{E}_{z\sim q_{a^0}}[\log\frac{q_{a^0}}{p}]=\mathrm{KL}(q_{a^0}\|p)$，得到

$$\mathrm{KL}(q_a\|q_{a^0}) \;\approx\; \mathrm{KL}(q_a\|p) - \mathrm{KL}(q_{a^0}\|p)$$

- 即"两个 KL-to-先验之差"：候选的绝对不典型度减意图的绝对不典型度 = **相对不典型度**；
- 该式是**近似**（把 $\mathbb{E}_{q_a}$ 换成 $\mathbb{E}_{q_{a^0}}$），两后验重合时取等；
- **实现不采用右端**而直接计算左端（后验对后验的 KL），原因：(i) 左端对期望测度是恒等式、无近似；(ii) 高斯闭式解中 $\mu$-偏移按 $1/\sigma^{0\,2}$ 加权（马氏距离，意图后验越确定的维度惩罚越重），梯度性质优于两个绝对 KL 的差（其 $\mu$-项是未加权平方差）。

## 5. 高斯闭式解与封顶

对角高斯下（**恒等式**，标准结果）：

$$\mathrm{KL}(q_a\|q_{a^0}) = \frac{1}{2}\sum_{i=1}^{16}\Big[ \frac{(\mu_i-\mu^0_i)^2}{\sigma^{0\,2}_i} + \frac{\sigma^2_i}{\sigma^{0\,2}_i} - 1 - (\mathrm{logvar}_i - \mathrm{logvar}^0_i) \Big]$$

- $\mu_i,\sigma^2_i,\mathrm{logvar}_i$：候选后验第 $i$ 维均值、方差、对数方差；
- $\mu^0_i,\sigma^{0\,2}_i,\mathrm{logvar}^0_i$：意图后验对应量；
- 四项分别来自迹、均值二次型、维数常数、对数行列式比（一般高斯 KL 闭式解的对角化）。

代价（**定义**）：

$$C(\hat x_0) = -\min\big(\mathrm{KL},\ \kappa\big),\qquad \kappa=2.5\ \text{nats}$$

- $\kappa$：封顶——$\mathrm{KL}\ge\kappa$ 时代价常数、梯度为 0，单块推力饱和；DP 先验是另一层信任域（见第 6 步的乘积形式）。

## 6. 注入与整体语义（压缩，细节见代码）

去噪每步（$t<100$，全部步）：

$$x_t \leftarrow x_t - \eta\,\sqrt{1-\bar\alpha_t}\;\frac{\partial}{\partial x_t}\sum_{b=1}^{B} C\big(\hat x_0^{(b)}(x_t)\big)$$

- $\eta=3.0$：引导步长；$\sqrt{1-\bar\alpha_t}$：随去噪衰减的缩放（Dhariwal & Nichol 2021 的引导项形式）；
- $\sum_b$：sum 归约（每行梯度不被 $B$ 稀释，2026-08-21 修复）；
- 梯度经 $x_t\to\hat x_0\to a\to(\mu,\mathrm{logvar})$ 链式反传，全部环节可微。

隐式采样分布（引导采样的标准结果，Dhariwal & Nichol 2021）：

$$p_{\text{guided}}(x)\ \propto\ p_{\text{DP}}(x)\cdot\exp\big(\min(\mathrm{KL}(x),\kappa)\big)$$

- $p_{\text{DP}}$：冻结 DP 先验（信任域主体）；似然权重至多放大 $e^{\kappa}\approx 12.2$ 倍后饱和。

## 7. 推导链总结：每一步的性质

| 步 | 内容 | 性质 |
|---|---|---|
| 1 | $I(Z;S')=H(S')-H(S'\mid Z)$ | 恒等式 |
| 1 | $-H(S'\mid Z)$ 交给 VIB 训练（$D_\psi$ 回归） | 训练期机制 |
| 2 | $H(S')$ 不可直接算 ⇒ 用"后验移动量"代理 | 代理近似（入口） |
| 2 | 后验不动 ⇒ 未来分布不动（$D_\psi$ 确定性） | 恒等式 |
| 3 | $I \ge \mathbb{E}[\log q_\phi - \log p]$ | 变分下界（DIAYN） |
| 4 | $\Delta r$ 相对化，先验相消 | 恒等代数 |
| 4 | $\mathbb{E}_{q_a}[\Delta r]=\mathrm{KL}(q_a\|q_{a^0})$ | 恒等式（实现量） |
| 4 | $\mathrm{KL}(q_a\|q_{a^0})\approx\mathrm{KL}(q_a\|p)-\mathrm{KL}(q_{a^0}\|p)$ | 近似（未采用，仅解释语义） |
| 5 | 对角高斯闭式解；$-\min(\mathrm{KL},\kappa)$ | 恒等式 + 定义 |
| 6 | $\sqrt{1-\bar\alpha_t}$ 缩放注入 ⇒ $p_{\text{DP}}\cdot f$ 采样 | 标准结果（classifier guidance） |

## 8. 论文依据

| 步骤 | 出处 | 用了它的什么 |
|---|---|---|
| 3 | Eysenbach et al. 2019, DIAYN, arXiv:1802.06070 | 变分下界 $I\ge\mathbb{E}[\log q_\phi(z\mid x)-\log p(z)]$ 与判别奖励形式 |
| 1 | Alemi et al. 2017, Deep VIB, arXiv:1612.00410 | $q_\phi$ 对角高斯参数化与训练期 KL-to-先验（使 $p(z)=\mathcal N(0,I)$ 成为合法参照） |
| 6 | Dhariwal & Nichol 2021, arXiv:2105.05233 | $\sqrt{1-\bar\alpha_t}$ 缩放的加性引导项 ⇒ 采样 $p\cdot f$ |
| 6 | Sun & Song 2025, LPB, arXiv:2508.05941 | 冻结 DP 去噪循环内注入代价梯度的实现模板 |
| 5 | Schulman et al. 2017, PPO, arXiv:1707.06347 | 对散度型目标设上界（$\kappa$）的结构先例 |
| 6 | Ho & Salimans 2022, arXiv:2207.12598 | 引导权重 $>1$（$\eta=3$）的采样实践先例 |
| 背景 | Chi et al. 2023, Diffusion Policy, arXiv:2303.04137 | 被引导的冻结基座 |
| 背景 | SOE, arXiv:2509.19292 | 潜空间加噪探索的对照路线（SOE 扰动 $z$，本文经 $\partial q_\phi/\partial a$ 扰动 $a$） |
