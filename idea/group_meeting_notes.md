# 组会汇报提纲：Classifier-Guided Exploration

> 背景：上次组会未定明确目标，本次任务是「汇报对 idea 的理解」。
> 隐含目标：证明已吃透 idea，并暴露值得导师 / 学长拍板的关键问题。

## 1. 复述 idea + 讲清动机（建立基线）
- 一句话：**冻结的 base DP** + **VIB 动力学模型**学 skill 潜空间，测试时从先验采样 skill，用 classifier guidance 引导 DP 探索。
- 训练：编码器 $\bar p_\theta(z\mid S_t,A_t)\to z$；解码器 $q_\phi(S_{t+1}\mid S_t,z)$ 预测下一状态；目标 $\max\ I(Z;S_{t+1}\mid S_t)-\beta\,I(Z;A_t\mid S_t)$。
- 测试：$z\sim\mathcal N(0,I)$；$\nabla_a\log p=\nabla_a\log\bar p_{DP}(a\mid s)+\nabla_a[-Cost]$，$Cost=\|z-\mu(s,a)\|_2$。
- **与 SOE 的本质区别**（学长在场，重点讲）：SOE = 观测压缩潜空间 + 潜空间加噪；本 idea = $(s,a)\to$skill 语义潜空间 + 显式 guidance。

## 2. 把数学讲透（显示深度）
- 目标两项含义：$I(Z;S_{t+1}\mid S_t)$ 逼 $z$ 抓「要把状态带成什么样」；$-\beta I(Z;A_t\mid S_t)$ 逼 $z$ 少依赖「具体怎么动」。
- VIB 上界落地：next-state 重建项（扩散去噪 MSE 或确定性回归）+ 解析 KL。
- score 分解 + 去噪循环的每一步细节。

## 3. ⭐ 关键开放问题（最有价值，拿来讨论）
- **(a) next-state prediction 可行性**：图像观测下预测下一帧是 world-model 级难题，SOE 靠 action reconstruction 绕开。**计划先用低维状态验证**——请导师确认取舍。*(头号风险)*
- **(b) guidance 与训练目标的内在矛盾**：guidance 要 $\mu(s,a)$ 对 $a$ 敏感，但 $-\beta I(Z;A_t)$ 恰恰压制它。β 一边压制、一边 guidance 又依赖——这正是「β 太大探索失败」的根因。
- **(c) 测试 / 训练分布 gap**：训练 $z$ 从 $(S_t,A_t)$ 编码（有信息），测试从 $\mathcal N(0,I)$ 采样（无信息），靠 KL 拉近；测试期编码器吃**带噪** $a$，训练见干净 $a$——近似，可能影响 guidance 质量。
- **(d) Cost 选择**：为何用 $\|z-\mu\|_2$ 而非 $-\log\bar p_\theta(z\mid s,a)$？后者才是严格 classifier guidance 形式，但要多算一次 KL。

## 4. 下一步 + 请导师拍板
- 提议：阶段 1 的 **low_dim toy demo** 作为近期目标。
- 请确认：(1) 状态空间取舍；(2) 是否复用 SOE base DP 与数据管线；(3) β 初始量级。

## 附：可带去组会的关键对比表
|  | SOE | 本 idea |
|---|---|---|
| 信息瓶颈 | action reconstruction | next-state prediction |
| 探索方式 | 潜空间加噪 $z=\mu+\sigma\varepsilon\alpha$ | classifier guidance（加 Cost 梯度） |
| base DP | 训练时与插件共享 | 冻结、训练时不在场 |
| 头号风险 | — | next-state prediction 难度 |
