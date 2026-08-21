# Guidance 的 1/B 缩放 bug(2026-08-21 发现,记录在案、暂不修)

> **状态**:已确认 + 已实证;修复方案已定(一行级),**经用户决定先不修**。
> 关联分析:`experiments/e2_scout_guidance_gradient_analysis.md`(梯度膨胀);
> 触发场景记录:`experiments/vis_final_summary.md`(exp1 四宫格排查)。

## 1. 机制

guided rollout 的注入路径(`scout/guidance/policy.py::guided_conditional_sample`):

```
loss = planner.compute_loss(x̂₀, current_obs)      # 标量
cond_grad = -autograd.grad(loss, trajectory)
trajectory = trajectory + guidance_scale·√(1−ᾱ_t)·cond_grad
```

而 `scout/guidance/planner.py::compute_loss` 的 cost 是
**"mean-reduced over batch and chunk"**:

$$\text{loss} = \frac{1}{B}\sum_{i=1}^{B} L_i \quad\Rightarrow\quad
\frac{\partial\,\text{loss}}{\partial\,x_i} = \frac{1}{B}\frac{\partial L_i}{\partial x_i}$$

(各行 cost 只依赖各自的动作,雅可比块对角,本不会有交叉项。)于是每条轨迹实际受到的推力:

$$\text{有效 guidance} = \frac{\text{guidance\_scale}}{B}\cdot\frac{\partial L_i}{\partial x_i}$$

**B = 本次 replan 调用拼在一起的并发 env 数**——不是设计参数,是调度巧合:
vec 引擎(`scout/eval/rollout_vec.py::_replan`)把"恰好同时需要新 chunk"的
slot 批量喂给 `predict_action_dyn_guided`。一波开始时 B≈n_envs,随着成功回合
(~130 步)提前退场 B 逐渐缩小,跑满 300 步的失败轨迹被越来越强的力度推
(极限 B=1 时回到名义 scale)——**一个没人设计过的 wave 内非平稳性**。

## 2. 实证(同一对 ckpt、同 seed42 同 20 场景,只改 n_envs)

DP-SCOUT-exp3 + dyn-SCOUT-exp3(|dNLL/da|=10):

| B | succ/20 | avg_jerk |
|---|---|---|
| 4 | 9 | **0.989**(乱飞) |
| 20 | 17 | **0.272**(正常) |

匹配 B=20 后的梯度→jerk 阶梯(e2-SCOUT 链,消除 1/B 混淆):

| dyn | \|dNLL/da\| | jerk(B=20) |
|---|---|---|
| exp3 | 10 | 0.272 |
| exp4 | 39 | 0.385 |
| exp5 | 109 | 0.439 |

→ 梯度膨胀效应真实但温和(~1.6×);此前混批比较(B=4 vs B=20)给出的 2.5× 是伪影。

## 3. 影响面

| 场景 | B | 有效 scale(配置 0.5) |
|---|---|---|
| LPB 参考实现 | 1(逐 env 调) | 0.5(名副其实,标定基准) |
| **e2–e5 全部真实 round**(n_envs=50) | ≈50→1 | **开局 ≈0.01** |
| vis_final 首批 | 20 | 0.025 |
| vis_final 重跑 | 4 | 0.125(引发本次排查) |

- **e2–e5 的所有实验都实际跑在名义 1/50 的力度上**。实验内部/之间对比仍有效
  (B 分布一致),但"guidance_scale=0.5"的字面值从未真正生效;与 SOE/LPB 的
  scale 不可直接对标。
- SCOUT01(scale 0.1)有效 ≈0.002——非常弱,这也许是 0.1 与 0.5 两 arm
  行为接近的原因之一。
- **不受影响**:guide=off 路径(不加载 planner);VIB 训练(loss 自优化,
  mean 归约无害);梯度膨胀分析(固定 batch、归一化一致);
  `scale=0` 完全 no-op 的不变量。

## 4. 为什么一直没被抓到

- mock/合成验证(`_verify.py`、guidance_checks)里 guided 调用都是 B=1,
  与 LPB 语义一致,1/B=1 恒成立;
- 每步验证过的不变量("scale=0 与 unguided 逐位一致")对任意 B 都成立,
  检测不到力度缩放;
- 真实 round 恒定 n_envs=50,实验内自洽;只有**跨 n_envs 的 run 互相比较**
  (vis_final 重跑)才把 5× 的差暴露出来。

## 5. 修复方案(待批,未实施)

guided 注入路径的 cost 改 **per-row 归约**:取 sum 的梯度(块对角 ⇒ 每行拿到
自己完整的无缩放梯度),B 无关,`guidance_scale` 语义回到 LPB 的 B=1 标定。
实现上一行级(`compute_loss` 加 `reduction` 参数,或 policy 侧乘回 B)。

**修后注意**:直接沿用 scale=0.5 会立刻比历史所有实验(~0.01)激进 50 倍,
行为不可外推——需要重校/重扫 scale 再续链;历史 e2–e5 结论在其(一致的)
低力度状态下依然自洽。
