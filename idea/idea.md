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
Z \sim r(z) = \mathcal{N}(0; I), \quad Cost(a_t, z | S_t) := \left\| z - \bar{p}_\theta(z | S_t, A_t) \right\|_2
$$

**Diffusion Denoising:**

$$
\nabla_a \lg p(a | s) = \nabla_a \lg [\bar{p}_{DP}(a | s) \, e^{Q(a, s)}]
$$

$$
= \nabla_a \lg \bar{p}_{DP}(a | s) + \nabla_a [-Cost(a, z | s)]
$$

- 原始 DP 输出
- Classifier Guidance.
