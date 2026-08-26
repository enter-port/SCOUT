# Idea

## Train Phase

```
St → Encoder → Z → Decoder → St+1
At → Encoder
St → Decoder
```

- skill, 和 N(0; I) 对齐。
- End-to-End Dynamics Model

$$
q_\phi(S_{t+1} | S_t, z)
$$

**优化目标:**

$$
\max I(Z; S_{t+1} | S_t) - \beta I(Z; A_t | S_t)
$$

**VIB:**

$$
\min_{\theta, \phi} -\mathbb{E}_{z \sim \bar{p}(z|S_t, A_t)} \lg q_\phi(S_{t+1} | S_t, z) + \beta \, KL[\bar{p}_\theta(z|S_t, A_t) \, \| \, r(z)]
$$

- Prior, 标准高斯。

---

## Test Phase

$$
a^0 := \text{DP 无引导意图（本块基线）}, \qquad
Cost(a_t \mid S_t) := -\min\Big( \mathrm{KL}\big( q_\phi(z \mid \bar s_t, a_t) \,\big\|\, q_\phi(z \mid \bar s_t, a^0) \big),\ \kappa \Big)
$$

- **entropy cost**（2026-08-24 定稿，方案三）：把候选动作的编码后验推离 DP 自身无引导意图的后验——「做策略自己不会做的事」；$\kappa$ 为 KL 封顶（信任域）。推导见 [`entropy_cost.md`](entropy_cost.md)。
- **v0 原始 cost（2026-08-06，已被上式取代）**：$Z \sim r(z)=\mathcal N(0;I)$，$Cost(a_t,z\mid S_t) := \| z - \bar p_\theta(z \mid S_t, A_t) \|_2$（从先验采样目标 skill，把动作推向「编码回去 $\approx z$」）。

**Diffusion Denoising:**

$$
\nabla_a \lg p(a | s) = \nabla_a \lg [\bar{p}_{DP}(a | s) \, e^{Q(a, s)}]
$$

$$
= \nabla_a \lg \bar{p}_{DP}(a | s) + \nabla_a [-Cost(a | s)]
$$

- 原始 DP 输出
- Classifier Guidance.
