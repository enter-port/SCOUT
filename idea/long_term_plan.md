# Classifier-Guided Exploration 落地计划

> 目标：把导师给的 classifier-guided exploration idea 落地到学长 SOE 工作同等成熟度（仿真 benchmark + multi-round self-improvement + 真机 + 论文）。
> 制定日期：2026-08-06。

## 核心判断（贯穿全程的最大风险）

**next-state prediction 是本 idea 的头号工程风险。** SOE 之所以工程上很顺，是因为它的信息瓶颈走的是 **action reconstruction**（动作只有 10 维，好预测），巧妙绕开了 next-state prediction 这个 world-model 级难题。而本 idea 的解码器要预测 $S_{t+1}$——若观测是图像（SOE 真机是 $216\times288\times3$ ×2 路相机），预测下一帧图像非常难。

**结论：先在低维状态上验证可行性，作为第一个硬里程碑。**

## 五阶段路线

### 阶段 0｜文献与方法夯实（纯查阅 + 推导）
- [x] 精读 SOE：论文 + 代码（核心 `SOE/src/policy/dp_ext.py`，双路径与梯度隔离；笔记见 `SOE/SOE_LIB_training_notes.md`、`SOE/SOE_Feishu_notes.md`）
- [x] Classifier / Classifier-Free Guidance：Dhariwal & Nichol (2021)；把 score 分解公式吃透
- [x] **DIAYN**（Eysenbach et al.）：目标函数与本 idea 一脉相承（`papers/DIAYN.pdf` 在手，必读）
- [x] DeepVIB（`papers/DeepVIB.pdf`）：变分上界与 KL 解析形式
- [x] World model 系列（Dreamer / PlaNet）：搞清图像观测下 next-state prediction 为何难、别人怎么绕
- [ ] baseline 生态：SIME、RISE、VQ-BeT、Diffusion Policy

### 阶段 1｜方案细化与可行性验证（推导 + 小代码）
> 详细实验计划见 [`stage1_plan.md`](stage1_plan.md)（数据 / 解码器形式 / E0–E4 实验 + metric + 过线）。
- [ ] 定状态空间：先用 robomimic **low_dim**
- [ ] 定解码器形式：确定性回归 vs 扩散式 next-state 去噪
- [ ] 推 guidance 可实现性：$Cost=\|z-z_\theta(s,a)\|$（$z_\theta$=reparam）对**带噪** $a$ 求梯度是否稳定、每步反传开销预估
- [ ] 🚩 **硬里程碑**：单任务（如 `lift`）低维 toy demo——证明 classifier guidance 能把动作推向不同 skill、产生有意义的探索（而非噪声）
> 此里程碑是 **go/no-go 节点**。跑通才值得往下走。

### 阶段 2｜核心代码实现（对标 `SOE/src/`）
- [ ] 新建 policy 类（编码器 + dynamics decoder），**复用 SOE 冻结的 base DP**
- [ ] 实现 VIB 训练 loss（next-state 重建 + KL），自定义 `backward`
- [ ] 在 `diffusion.py` 的 denoising loop 里实现 classifier guidance（加 Cost 梯度项）
- [ ] 对比实验：guidance vs SOE latent perturbation，看探索质量（多样性 + 成功率）
- [ ] 里程碑：单任务完整 train → explore → improve 跑通

### 阶段 3｜实验铺开（对标 `SOE/simulation/`）
- [ ] robomimic 四任务（can / lift / square / transport）
- [ ] 多 seed（沿用 SOE：233 / 2333 / 23333 / 233333）
- [ ] multi-round self-improvement 闭环（复用 `run_full_multi_round.py` 编排逻辑）
- [ ] baseline：SOE、SIME、纯 DP + 动作噪声
- [ ] ablation：β、guidance scale、解码器类型
- [ ] 里程碑：产出 SOE 那种多轮 success-rate 曲线 + 对比表

### 阶段 4｜真机 + 论文（对标 `SOE/realworld/` + arXiv）
- [ ] 仿真结论稳了再迁移真机（Flexiv + 双 RealSense，学长应有现成环境）
- [ ] 写作

## 节奏建议
阶段 0–1 是「想清楚」，不要急着写大工程代码；阶段 1 的 toy demo 是真正的 go/no-go；阶段 2–3 可大量复用 SOE 已有的代码框架（数据加载、训练、评估、multi-round 编排），不必从零造。
