# Idea 流程梳理

## 1. 输入 / 输出
- 训练：输入转移 $(S_t,A_t,S_{t+1})$（当前状态、动作、下一状态）；产出 $\hat S_{t+1}$ 与真实 $S_{t+1}$ 比
- 测试：输入观测 $s$ + $z\sim\mathcal N(0,I)$；输出动作 $a$（被引导的 DP 生成）

## 2. 涉及的网络
- 编码器 $\bar p_\theta(z\mid S_t,A_t)$：$(S_t,A_t)\to z$ 的高斯 $(\mu,\sigma)$ —— 训练
- 动力学解码器 $q_\phi(S_{t+1}\mid S_t,z)$：$(S_t,z)\to\hat S_{t+1}$ —— 训练
- base DP $\bar p_{DP}(a\mid s)$：预训练、冻结；训练阶段不出现，仅测试提供 score

## 3. 训练目标与 loss
- 信息论目标：$\max\ I(Z;\,S_{t+1}\mid S_t)-\beta\, I(Z;\,A_t\mid S_t)$
  - 第一项：$z$ 能预测 next state（给定 $S_t$）→ 抓住"要把状态带成什么样"
  - 第二项：$z$ 尽量少依赖 action → 是"目标/结果"，非"具体怎么动"
- VIB 变分上界：
  - $\mathcal L=\underbrace{-\mathbb E_{z\sim\bar p_\theta(z\mid S_t,A_t)}\log q_\phi(S_{t+1}\mid S_t,z)}_{\text{① 下一状态重建}}+\underbrace{\beta\,KL[\bar p_\theta(z\mid S_t,A_t)\,\|\,r(z)]}_{\text{② KL 正则}}$
- 第①项怎么算（下一状态重建）：
  - $(S_t,A_t)\xrightarrow{\text{编码器}}z=\mu+\sigma\varepsilon$，$(S_t,z)\xrightarrow{\text{解码器}}\hat S_{t+1}$
  - 扩散模型 → 下一状态去噪 MSE（$S_{t+1}$ 加噪、解码器在 $(S_t,z)$ 下预测噪声）
  - 确定性回归 → $\|S_{t+1}-\hat S_{t+1}\|^2$
- 第②项怎么算（KL，解析）：$\beta\,KL=-\tfrac{\beta}{2}\sum_i\big(1+\log\sigma_i^2-\mu_i^2-\sigma_i^2\big)$

## 4. 训练前向链路与梯度回传
- 前向一条链：$(S_t,A_t)\to$编码器$\to z\to$动力学解码器$\to\hat S_{t+1}$
  - $\hat S_{t+1}$ vs $S_{t+1}$ → 第①项；$(\mu,\sigma)$ vs $\mathcal N(0,I)$ → 第②项
- 梯度回传：一次 `loss.backward()`
  - 第①项：穿解码器（更新 $\phi$）→ 穿 $z$/重参数化 → 回编码器（更新 $\theta$）
  - 第②项：只更新 $\theta$
  - 直链，无需额外隔离

## 5. 测试阶段：classifier-guided 探索
- 用编码器的 $\mu$ + 预训练 base DP；动力学解码器测试时不用
- score 分解：$\nabla_a\log p(a\mid s)=\nabla_a\log\bar p_{DP}(a\mid s)+\nabla_a[-Cost(a,z\mid s)]$
  - 第一项：base DP 自身的去噪方向
  - 第二项：Cost 的梯度，把动作往"能命中 $z$"方向推
- $Cost(a,z\mid s)=\big\|z-\mu(s,a)\big\|_2$
  - $z\sim\mathcal N(0,I)$ 整段生成定住；Cost 越小 → 动作"编码回去正好等于 $z$"
- 去噪循环（$t=T\to0$）：
  - DP 看（带噪 $a$、步 $t$、$s$）→ 去噪方向
  - 算 Cost，取对 $a$ 的梯度，乘缩放加到去噪方向
  - 沿调整方向擦一步（加随机噪声）→ 下一步 $a$
  - 走完得：既在 DP 支集上、又朝着 $z$ 走的动作

## 6. 采样 $z$ 的意义
- base DP 自跑只给默认动作（无探索）；采 $z$ = 在"合法 skill 空间"挑目标，逼 DP 生成奔向它的动作 → 探索多样性
- 从 $\mathcal N(0,I)$ 采：训练 KL 已把真实 skill 压到标准正态附近 → 采到的是"合法 skill"

## 附：注意点
- $\beta$ 权衡：太大 → $z$ 与动作脱钩 → 换 $z$ 动作不变（探索失败）；太小 → 退化为普通动力学模型（好在重建项逼编码器读 $A_t$，依赖被软化而非切断）
- 测试期编码器看到带噪 $a_t$（在干净数据上训），是近似
