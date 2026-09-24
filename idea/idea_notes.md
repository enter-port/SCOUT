# SCOUT 模型与训练流程笔记

更新：2026-09-25。本地输入契约修复分支；尚未进行修复版运行验证。本文按当前 `scout/`、`scripts/atom/` 和标准链代码整理。
具体运行以 campaign 配置、生成的训练配置及 checkpoint 为准；历史计划见 [archive](archive/README.md)。
默认预算示例来自 [threading 标准模板](../configs/threading/campaign.json)，不代表所有历史实验或任务的最优参数。

## 1. 当前方法与边界

SCOUT 用 VIB 动力学模型学习状态和动作到 skill 后验的映射，并在 DP 去噪时用该后验构造引导。
标准链支持 `ATY`、`DP`、`ORBIT` 三种 arm，默认运行 ATY 和 DP。

- **DP**：无 guidance 的扩散策略，也会在 round 间重训。
- **ATY**：CLI `--guide atypical`，实现类为 `KLCostPlanner`；用候选后验与本 chunk 的 DP 意图后验之间的封顶 KL 引导动作。
- **ORBIT**：继承同一 KL 与爬坡计算，在壳附近改用反馈和切向噪声。

“DP 冻结”指 rollout/guidance 时不更新参数；DP train 阶段会重新训练 DP。
旧目标 skill 的 NLL 模式 `dyn/expert`、novelty/shell/combo 等仍有代码，但不是标准链的 arm。
Drifting policy 的 1-NFE 方案仅为 [研究提案](research/drift_guide.md)，没有替换当前 DDPM。

## 2. 训练数据与动作空间

VIB 输入为转移 `(S_t, A[t:t+fs], S_{t+fs})`，`fs=dataset.frameskip`，常用值为 8。
`train_vib._slice_transition` 取窗口第 0、1 个观测和前 fs 步动作，将动作展平。
单臂常用每步 10 维（位置 3 + rotation-6d 6 + gripper 1），因此动作向量为 80 维；
双臂任务按配置使用 20 维/步。训练目标是下一时刻的状态编码，不重建像素。

输入 core 必须已完成任务对应的数据拆分和绝对动作转换。HDF5 中绝对动作通常为 axis-angle；
数据集加载器转换到模型的 rotation-6d 表示。Guidance 输入先由 DP action normalizer
反归一化，再送入 VIB。实际工厂使用 `make_action_bridge` / `UnnormalizeOnlyBridge`，
不是假设两个模型的数值空间已经一致。

Policy 在 guided denoise 前将执行窗口传给 planner：起点为 `n_obs_steps - 1`，
长度为 `n_action_steps`。所有 cost 与 anchor 共用该窗口，常见配置为 `[1:9]`；
若 VIB 训练的 chunk 长度不匹配或窗口超出 horizon，则报错。
旧实现的 `[0:8]` 实验属于修复前版本，不能与新实现混为同一条趋势。

Rollout 和 core 标定均以 `[0,1]` CHW 图像进入 VIB adapter；不再对 env
已归一化的图像重复除以 255，也不再在标定端乘 255 补偿。Expert bank 直接
读取 uint8 HDF5，显式除以 255 后与它们共享中心 crop。修复版需重新标定 η、κ。

实现：[train_vib.py](../scout/train_vib.py)、[robomimic_dset.py](../dyn_model/datasets/robomimic_dset.py)、
[normalizer.py](../scout/normalizer.py)、[rollout.py](../scout/eval/rollout.py)、
[entropy_costs.py](../scout/guidance/entropy_costs.py)。

## 3. 网络与冻结范围

以下形状以常见单臂双视角配置为例；实际视角名、图像尺寸与维度由任务配置决定。

| 模块 | 当前实现 | 训练时是否更新 |
|---|---|---|
| E_s 视觉分支 | 从指定 DP checkpoint 提取每视角 ResNet backbone，avgpool 后每视角 512 维 | 冻结，保持 eval |
| E_s proprio 分支 | Conv1d(proprio_dim, 64, kernel=1) | 更新 |
| VIBEncoder | concat 状态与动作 → **LayerNorm** → 两层 hidden=128 的 ReLU MLP → μ/logvar | 更新 |
| DynamicsDecoder | concat z 与状态 → 同类 MLP → 下一状态编码 | 更新 |
| 当前 DP | DiffusionUnetHybridImagePolicy；rollout 使用 ScoutPolicy 的引导路径 | DP train 更新，rollout 不更新 |

状态编码维数为 `512 × n_views + proprio_emb_dim`。双视角、proprio embedding=64 时：

```text
图像特征 1024 + proprio embedding 64 → s_bar: 1088
concat(s_bar:1088, action:80) → LayerNorm(1168)
    → Linear(1168,128) → ReLU → Linear(128,128) → ReLU
    → Linear(128,32) → mu:16, logvar:16
z = mu + exp(0.5 * logvar) * epsilon
concat(z:16, s_bar:1088) → Linear(1104,128) → ReLU
    → Linear(128,128) → ReLU → Linear(128,1088)
```

`EncoderMLP` 本体没有 normalization；VIBEncoder 在它之前另加输入 LayerNorm。
旧笔记把整个 encoder 写成“无 norm”，并据此给出的参数量已不适用。
VIB 的视觉读出是 512 维 avgpool；DP 自身的 obs encoder 读出不能与之混为一谈。
冻结 backbone 由 `ModuleDict` 注册，随 VIB 的 `state_dict` 保存。

实现：[encoder.py](../scout/model/encoder.py)、[vib.py](../scout/model/vib.py)、
[mlp.py](../scout/model/mlp.py)、[resnet_encoder.py](../dyn_model/models/resnet_encoder.py)。

## 4. VIB 训练目标

令 `s_bar = E_s(S_t)`、`target = E_s(S_{t+fs}).detach()`，编码后验为
`q(z | s_bar, a) = N(mu, diag(exp(logvar)))`，重参数采样 z，解码器预测 target。
对样本 b、latent 维 i：

$$\ell_{b,i}^{KL}=\tfrac12(\mu_{b,i}^2+\exp(l_{b,i})-1-l_{b,i}),\qquad
m_b=\operatorname{mean}_j(\hat s_{b,j}-s'_{b,j})^2.$$

实际优化的是：

$$\mathcal L=\frac{\sum_b w_b m_b}{\sum_b w_b}
+\beta\operatorname{mean}_b\sum_i\max(\ell_{b,i}^{KL},\mathrm{free\_bits}).$$

未启用 failure weighting 时第一项就是样本均值；failure weighting 只作用于重建项。
训练日志的 `kl` 是未加 free-bits 地板的真实 KL，优化使用 `kl_fb`，两者不能混读。
一次 backward 更新 proprio embedding、VIB encoder、decoder；目标端停止梯度，视觉 backbone 不更新。

`feature_cache=true` 时预计算冻结视觉特征，仍实时训练 proprio 分支；
数据集/DP encoder 来源改变时必须使用匹配的缓存。train/val 按 episode 划分，当前实现将末尾一部分 demo 留作验证。
β、free_bits、failure_weight、steps_per_epoch 都以实际生成的 dyn 配置为准。

实现：[scout_vib.py](../scout/model/scout_vib.py)、[train_vib.py](../scout/train_vib.py)、
[feat_cache.py](../scout/feat_cache.py)。

## 5. ATY：封顶 KL 与实际去噪更新

每个 action chunk 的第一个 guided denoise step，`select_z` 捕获该步干净动作估计的后验
`q_0=N(mu_0,diag(exp(logvar_0)))` 并 detach，作为该 chunk 固定的 anchor。
它不是另跑一条完整无 guidance 轨迹后得到的终点。状态编码在 chunk 内缓存。
虽然钩子仍叫 `select_z`，ATY 并不从先验采样目标 z，也不使用动力学 decoder。

$$f(a)=KL(q(z\mid\bar s,a)\|q_0)
=\tfrac12\sum_i\left[\frac{(\mu_i-\mu_{0,i})^2+\exp(l_i)}{\exp(l_{0,i})}-1-(l_i-l_{0,i})\right],
\qquad C(a)=-\min(f(a),\kappa).$$

`KLCostPlanner.guided_step` 通过一次候选 encoder 前向和一次 `kl.sum()` backward 得到逐行梯度。
先对 uncapped KL 求导，再按 cap 屏蔽；当前边界语义是 `f <= kappa` 保留梯度，`f > kappa` 置零。
批内使用 sum，不用 mean，避免每条样本的推力被并发数 B 除掉。
标准链使用原始 η 单位：

$$x_t\leftarrow x_t+\eta\sqrt{1-\bar\alpha_t}\,\mathbf1_{f\leq\kappa}\nabla_{x_t}f.$$

梯度经过 `x_t → UNet → x0_hat → action bridge → VIB encoder`；模型参数不做 optimizer 更新。
随后 scheduler 用本步原来的 model_output 和调整后的 trajectory 计算反步。
guidance 由 `t < guidance_start_timestep` 控制，不应把所有任务都写成全程 100 步引导。
常见 DP 配置为 horizon=16、n_obs_steps=2、n_action_steps=8、100 步 DDPM；以 checkpoint 和 eval 配置为准。

底层保留 `eta_dimless` 归一化选项；标准 atom/calib 不开启它，不能与历史无量纲 η 数值互换。
“倾斜分布 p_DP exp(-C)”是理想化解释；当前有限步、按 x0_hat 构造的更新没有证明精确采样该分布，
也不保证真实动力学或成功率上的信任域。

实现：[policy.py](../scout/guidance/policy.py)、[entropy_costs.py](../scout/guidance/entropy_costs.py)；
设计讨论见 [entropy_cost.md](entropy_cost.md)，缩放事故见 [历史归档](archive/guidance_batch_scaling_bug.md)。

## 6. ORBIT：标准链启用 soft feedback

令 `g = grad_x f`，每一行按 `f >= kappa - delta` 判断 phase 2。
phase 1 复用 ATY 爬坡；phase 2 替换爬坡，不叠加第二份 η 剂量，加入反馈和切向噪声。
标准链固定传入 `--orbit-fb-clamp soft`，反馈残差为 `r = delta * tanh((f-kappa)/delta)`：

$$\Delta x=-\lambda r\frac{g}{\max(\|g\|^2,10^{-16})}
+\sigma_n(\sqrt{1-\bar\alpha_t})^p\xi_\perp,
\quad \sigma_n=\sigma_0 d^{n-1},\quad \xi_\perp=\xi-(\xi\cdot\hat g)\hat g.$$

soft 模式只在 `[kappa-delta, kappa+delta]` 带内加入噪声；平坦梯度行的反馈为零，噪声不投影。
phase 2 位移不再乘 η。标准 atom 默认 λ=.5、δ=.25、σ₀=.05、d=.5、p=2，可通过配置覆盖。
原 `fb_clamp=none` 仍在底层 CLI 中，但不是标准链默认。Newton 投影是局部线性近似，
不能据此宣称非线性 KL 永不越壳或不存在过剂量问题。

实现：[orbit_costs.py](../scout/guidance/orbit_costs.py)、[eval_explore.py](../scripts/atom/eval_explore.py)。

## 7. Grid search 与 calibration

Grid 在环境中比较成功指标，calib 在固定 core 上测量引导剂量。

| 阶段/模式 | 当前标准链行为 |
|---|---|
| grid | 显式 η×κ 网格，ORBIT 可再遍历 λ、σ；所有 cell 共用一次 base 无引导 eval 的失败集 |
| grid 选择 | 每个 arm 单独最大化 pass@K；并列取较小 η、κ，ORBIT 同剂量并列保持配置顺序 |
| r | 固定 κ，以 R 比例调整 η |
| rc（默认） | 先 R 调 η，再 C 调 κ；base pair 自己先做一次 R 标定，固定其 η/κ 作为 C 参考点 |
| pr | 约束 ηκ=P，每个 κ 上设 η=P/κ，再求 R 落入目标带；不再追加 C 标定 |
| dp_kl | 无引导 DP 多次采样得到 KL 中位数 κ，再标定 η |
| none | 原样沿用剂量 |

R 是逐引导步 `eta × noise_scale × mean(abs(capped_gradient))` 的均值，
除以整个 core 的 `mean(abs(raw abs_actions))`；C 是 guided 轨迹的 uncapped KL 均值。
DP KL 中位数则使用同一观测上不同 draw 的
`KL(q_final(draw_i) || q_anchor(draw_j)), i != j`，汇总后取 pooled median。

R/RC 沿用预算用尽后采用最后实测值的语义；PR/DP-KL 未收敛会停止链。
R→C 的第二步会改变 κ，最终 R 不保证仍在目标带。P≈6 的经验也不构成跨任务/跨轮保证。
完整定义、预算与限制见 [calib README](../scout/calib/README.md) 和 [P/R 研究结论](kappa_pr_base_calibration.md)。

## 8. 六轮训练与数据回灌

```text
准备 core → base DP → base dyn → grid →（RC 的固定 base 参考）
各 arm 的 round 1…6：
    ATY 标定当前 DP/dyn；DP 不标定；ORBIT 沿用 grid 参数
    无引导 eval → 冻结失败场景 → rescue 重试 → 合并指标/选定轨迹
    round 1…5：累计成功数据重训 DP；ATY/ORBIT 再训练 dyn
    round 6：只测 SR 和 pass@K
```

标准模板：DP base/retrain 均为 600 epochs、batch 64；dyn 为 300 epochs、batch 256、β=1e-5。
未指定 dyn β/batch 时，atom 沿用任务 YAML；模板值不是底层模型的全局默认。
DP `training.resume=False`，每次从头重训；dyn 视觉前端来自本轮新 DP checkpoint，在该次 dyn 训练中冻结。
DP arm 不重训 dyn；`dyn_freeze_after` 可控制 guided arm 后期冻结。

数据口径由 `RolloutPipeline._run_rescue` 和 `TrajSpool` 的选择规则决定：

| 输出 | 当前 rescue 选择规则与用途 |
|---|---|
| success.hdf5 | 被救回场景的成功轨迹；标准链启用 stop-on-first-success，每个救回场景至多一条 |
| all.hdf5 | 被救回场景的成功轨迹 + 始终失败场景的第一次探索失败轨迹；不是每一次尝试的总集合 |
| DP accumulated | core + 本 arm 第 1…N 轮保存的 successes；core 只保留一次 |
| dyn accumulated | core + 本 arm 第 1…N 轮 all.hdf5 中的选定轨迹 |

上述 HDF5 在合并格式中还携带 core；累计 merger 只追加各轮新增 demo，避免把 core 重复计入。
初始 eval 已成功的场景不增加训练数据。零新增成功不跳过 DP 训练；没有新的选定轨迹时仍可用原累计数据训练。
累计文件由标准链显式传递，不扫描未来轮次，不混入其他 arm 或 grid cell。

本项目 rescue 指标：`SR = baseline_solved / N`，`pass@K = (baseline_solved + rescued) / N`。
这里 K 是初始 eval 失败后的最多 K 次重试，不是总共只尝试 K 次。
底层 JSON 键 `pass_at_5` 是历史名称，实际 K 看 `explore_try_times`；标准链 summary 保存为 `pass_at_k`。

实现：[train.py](../train.py)、[rollout_pipeline.py](../scout/eval/rollout_pipeline.py)、
[traj_spool.py](../scout/eval/traj_spool.py)、[merge_sharded.py](../scout/eval/merge_sharded.py)、
[hdf5_writer.py](../scout/eval/hdf5_writer.py)。

## 9. 运行与验证入口

- 运行配置、独立原子操作与阶段恢复：[scripts/README.md](../scripts/README.md)。
- 实验身份和记录：[experiments/README.md](../experiments/README.md)，索引由 experiment_registry.py 生成。
- 验证范围：[atom/VALIDATION.md](../scripts/atom/VALIDATION.md)：本地编排测试及 Coffee 真实 GPU 短链，未跑正式预算六轮，未覆盖所有任务/模式的 GPU 组合。
- 实验结果以具体记录与 rollout JSON 为准；本文不重复维护成功率排名、在跑进程或“全局最优参数”。
