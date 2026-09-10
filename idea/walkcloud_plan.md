# WalkCloud(去噪步 x̂₀ 累积云)实现计划

> 2026-09-10 用户令:cloudrep(j 轴重试锚云)实现已全部清除;按新方向重写——
> **不用"采 M 个无引导动作块",而是把去噪 walk 每一步算出的 x̂₀(denoised action
> 估计)累积下来,每步 guidance 用此前所有累积的 x̂₀ 作为云做 drift 排斥**。
> 本文件为实施前计划(未动代码)。

## 0. 一句话机制

每个 chunk 的 100 步去噪 walk 内,把每步的 x̂₀ 一步估计累积成云;第 t 步的
guidance cost = 候选动作到云 {x̂₀_1..x̂₀_{t−1}} 的 drifting 核排斥(soft-min,
softmax 归一化权重)。**零引擎改动、零云账本、零门——纯 cost 替换。**

## 1. 数学形式

设本 chunk 去噪步 t = 1..100,每步策略算出 x̂₀_t(注入前)。步历史云:

```
H_t = { q_φ(z|s̄, x̂₀_1), ..., q_φ(z|s̄, x̂₀_{t−1}) }   ← 全部 detach 常量;仅本 chunk;select_z 重置
```

第 t 步 cost(候选 = 当前 x̂₀_t):

```
S_t  = −τ̃·log Σ_{j<t} exp(−KL(q_φ(z|s̄,x̂₀_t) ‖ H_t[j]) / τ̃)    (soft-min)
cost 行 = −min(S_t, κ)        τ̃ = clamp(0.3·池内极差, 0.02, 0.5)   (adapt 规则)
注入:x_t += η·√(1−ᾱ_t)·∇S_t   (继承 phase-1 注入线,逐行 sum 归约)
```

**边界退化(自带,无需特判)**:
- t=1:云空 → graph-connected 零注入(本步不引导);
- t=2:云 = {x̂₀₁} = 锚 → S ≡ KL(q_t‖q_anchor) = **逐位 atypical**;
- t≥3:云增长,开始偏离 atypical——差异从第三步起。

**与 GAElike(同 i 轴)的精确区别**:GAElike = γ^k 加权**求和**的 capped KL
(每个参照都施压、锚点主导、合力无归一);本方案 = drifting 核 **soft-min**
(最近参照主导、softmax 权重和恒 1、位移有界)。i 轴在"求和聚合"下被证伪
(GAElike 三方向全灭);核聚合形式从未在 i 轴上测过——本 A/B 回答的问题。

## 2. 类设计(唯一新文件 `scout/guidance/walk_cloud_costs.py`)

```
class WalkCloudCostPlanner(KLCostPlanner):        # 继承全部 phase-1(κ 遮罩/注入/遥测)
    __init__(cap, eta_dimless, cloud_tau=0.5, cloud_tau_mode='adapt',
             cloud_tau_frac=0.3, cloud_tau_min=0.02, cloud_hist_max=0)  # 0=不封顶
    select_z(x0_hat, obs): 历史清零(每 chunk 重置;不捕锚不提交——锚=history[0])
    _kl_backward(trajectory, x0_hat, obs):
        mu, logvar = vib_enc(s̄, bridge(x̂₀_t))
        历史空/B失配 → 返回 graph-zero(=t=1 不引导)
        kl = KL 矩阵 (B, H_t) → S = softmin(adapt τ̃)
        g = autograd.grad(S.sum(), trajectory)    # 单次反传
        push (mu, logvar).detach() 进历史          # backward 之后 → 当前步永不自参照
        return (S, g)
    compute_loss: 只读视图(不 push)
    reset(): 清历史
```

**明确不定义** set_row_jobs / try_started_gate / on_try_done → 引擎 gate 分派
落 None 分支,**rollout_vec 零改动**。`_kl_matrix`/`_cloud_softmin` 做成模块级
函数供共用。

## 3. 逐文件改动

| 文件 | 改动 |
|---|---|
| scout/guidance/walk_cloud_costs.py | 新建,上述类(~120 行) |
| scout/guidance/entropy_costs.py 或新模块 | 模块级 `_kl_matrix`/`_cloud_softmin` |
| scout/eval/rollout_pipeline.py | `_attach_planner` 加 `"walkcloud"` 分支(~10 行) |
| scout/eval/run_rollout.py | `--guide` choices+walkcloud;`--cloud-hist-max`;kwargs+summary |
| scripts/rand/ideas/NAMES.md | 重建(从 GAElike-dev 带)+ **walkcloud** 条目 |
| scout/eval/_smoke_walkcloud.py | 新建,§4 冒烟 |
| scripts/probe/ | 新探针(可基于 can824_gae_probe.sh 模板,arm=aty vs walkcloud) |

## 4. 冒烟清单(hermetic,本地+服务器双跑)

1. t=1 空云:graph-connected 零(值+梯度+连图);
2. t=2 逐位 = atypical(J=1 快路径,任意 τ);
3. 历史语义:每步 push 恰一次、长度随 t 递增、select_z 重置、compute_loss 只读;
4. soft-min 公式对独立 python 循环;核梯度 = softmax(−KL/τ̃);
5. 轨迹梯度 vs 中心差分;
6. 零 RNG;
7. cloud_hist_max>0 时保留最新。

## 5. 验证协议(45min/轮纪律)

- 速度 = softmin 档(单反传):can 全集一轮 ~40min ✓ 无需窗口;
- can 先行(aty 基线 14/43 已有),判据 >14 继续;square 随后(判据 >23);
- 预注册证伪:can ≤14 且遥测 mean_S 正常 → i 轴证伪扩展到核聚合,停;
- 盯 jerk(自排斥可能在 walk 尾部抖动)。

## 6. 风险(如实)

1. i 轴证伪可能同样适用于本方案:GAElike 复盘根因(i 轴非决策序列)对本方案同样
   成立,核聚合能否翻案未知——这是实验要回答的,不是已知优势;
2. 自排斥 vs 收敛对抗:walk 后期 x̂₀ 收敛,排斥"最近的自己"可能产生步内抖动
   或推离自然收敛点(κ 遮罩与软最小自限性是现有护栏);
3. 慢 steps 的 S 自行变小(云收敛)→ 引导自然衰减,可能特性也可能白烧。

## 7. 明确不做

- 不采 M 个无引导样本(用户否决);
- 不做跨 chunk 历史(每 chunk 重置;若要跨 chunk 是后续变体);
- 不动引擎、不动 atypical 臂。
