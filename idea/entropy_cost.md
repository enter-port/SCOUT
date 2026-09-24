# Entropy cost：封顶后验 KL 的动机与实现

更新：2026-09-24。当前 ATY 使用 `KLCostPlanner`；本文描述实际计算与设计解释的边界。
网络、标准链和标定见 [模型笔记](idea_notes.md)。早期完整推导保留在 [历史归档](archive/entropy_cost_2026-08.md)。

## 1. 为什么使用后验差异

VIB 将状态和动作 chunk 映射成 skill 后验，再结合状态由 decoder 预测下一状态编码。
探索希望在 DP 的动作分布基础上产生行为差异，因此用编码器后验的变化作为代理信号。

对于固定状态和固定确定性 decoder，如果两个动作得到同一个 z 后验，它们经 decoder 推出的
预测状态编码分布也相同。这仅说明“后验不变则模型预测分布不变”：
后验变化大不保证 decoder 输出变化大，更不保证真实环境中的状态熵或成功率增加。
当前代码没有直接估计未来状态熵，也没有优化一个可计算的真实环境 mutual information。

## 2. 实际 cost

设候选动作后验为 `q_a = N(mu, diag(exp(l)))`，anchor 为
`q_0 = N(mu_0, diag(exp(l_0)))`。Anchor 由每个 chunk 的第一个 guided step 的
干净动作估计建立并 detach；后续去噪步骤复用，不重新采样目标 z。

$$f(a)=KL(q_a\|q_0)
=\frac12\sum_i\left[\frac{(\mu_i-\mu_{0,i})^2+\exp(l_i)}{\exp(l_{0,i})}-1-(l_i-l_{0,i})\right].$$

这里 KL 的方向是候选到 anchor，不能交换；均值差由 anchor 方差加权。
代价定义为：

$$C(a)=-\min(f(a),\kappa).$$

最小化 C 等价于封顶前增大后验差异。κ 是配置/标定结果；2.5 只是常见起点。
从对数后验比出发，`E_{z~q_a}[log q_a(z)-log q_0(z)] = KL(q_a || q_0)` 是恒等式；
但选择用它代理探索价值是方法定义，不能仅凭这个恒等式推出实际状态熵增加。

实现：[entropy_costs.py](../scout/guidance/entropy_costs.py) 的
`_kl_rows`、`KLCostPlanner.select_z`、`_kl_backward` 和 `_climb_gradient`。

## 3. 进入 DP 去噪的方式

当前路径先取得 scheduler 的 `pred_original_sample`，选其前 fs 步、反归一化并展平，
经 VIB encoder 计算 KL。对批内 KL 求和并向 noisy trajectory 反传：

$$g=\nabla_{x_t}\sum_b f_b,\qquad
x_t\leftarrow x_t+\eta\sqrt{1-\bar\alpha_t}\,\mathbf1_{f\leq\kappa}g.$$

随后用当前 model_output 和调整后的 trajectory 做 DDPM 反步。
KL 超过 cap 的行爬坡为零；代码在恰等于 cap 时仍保留梯度。
Sum 归约避免每个样本被 batch size 除掉；cost 日志的 per-row mean 不等于梯度归约方式。
η、guidance_start_timestep 和 scheduler 步数都不是跨任务固定常数。

状态编码在 chunk 内缓存；VIB decoder 不参与推理。完整梯度路径保留 UNet 到 x0_hat 的依赖，
但不对模型参数做训练更新。实际动作切片与执行窗口的区别见 [模型笔记 §2](idea_notes.md#2-训练数据与动作空间)。

实现：[policy.py](../scout/guidance/policy.py)、[normalizer.py](../scout/normalizer.py)。
ORBIT 继承这一 KL 计算，但 phase 2 改为反馈和噪声，不能只用上式描述全部 ORBIT 更新。

## 4. “倾斜分布”解释的限制

可以定义一个理想化目标分布：

$$\tilde p(a)=p_{DP}(a)\exp(\min(f(a),\kappa))/Z.$$

对这个定义本身，权重在 `[1, exp(kappa)]` 内，归一化密度比受到相应界限约束。
这不是当前离散去噪器的精确输出分布定理：代码在 x0_hat 上算 cost，采用指定 η、有限步 DDPM，
且 anchor 随 chunk 的初始样本建立。不能把理想化密度比上界当作实际动作安全、动力学可行性
或成功率的保证；也不能宣称 κ 封顶确保采样轨迹的 KL 永不越界。

因此，后验差异是探索代理，κ 是 cost 的封顶参数。实际行为、轨迹质量与救回率仍需要 rollout 检验。

## 5. 与训练和标定的连接

训练 KL 是 `KL(q_a || N(0,I))`；探索 KL 是 `KL(q_a || q_0)`，参照分布不同。
训练使用 latent MSE 加 β×逐维 free-bits KL，E_s 的视觉分支冻结、proprio 分支可训练。
β、free_bits 和输入 LayerNorm 都会影响后验几何及动作梯度，不能只根据历史 η/κ 推断当前剂量。

当前标准链可选 R、R→C、P/R 或 DP KL 中位数标定，并将实际 η/κ 用于 rollout。
标定统计量、未收敛行为和经验适用范围见 [calib README](../scout/calib/README.md) 与
[P/R 研究结论](kappa_pr_base_calibration.md)。
