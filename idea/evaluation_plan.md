# 评估方案:SOE 怎么评,SCOUT 怎么评

> 配套 `long_term_plan.md` 阶段 1 的 go/no-go。

---

## 一、SOE 怎么评估

### 1. test-time 探索的三种机制

| 机制 | 做法 | 开关 |
|---|---|---|
| **SOE**(本方法) | 在 16 维潜空间 `z = μ + σε·α` 抖动,再过共用的动作解码器 | `--enable_exploration`(DPExt) |
| **SIME** | CADS 噪声直接加在 `global_cond` 上,随去噪步线性衰减 | `--sime` |
| action noise | 动作空间直接加高斯噪声(最弱基线) | `enable_action_noise` |

SOE 的抖动发生在"装满任务信息的潜空间",动作仍走扩散解码器 → 落在策略支集上(on-manifold)。训练用 `α=1`,只推理放大。

### 2. 成功判定

- 仿真:`env.is_success()["task"]` 二值,episode 一旦成功即结束(`done == success`)。
- 真机:人按 j(success)/ k(fail)/ d(discard) 标注。

### 3. multi-round self-improvement 闭环(6 轮)

每轮做四步:
1. 用上一轮的「核心 demo + 成功探索 rollout」训练;
2. eval:100 个初始态先各跑 1 次(基线),失败的再探索 `try_times=5` 次;
3. 只把 `np.any(dones)` 为真的探索 rollout 筛出,与核心 demo 合并;
4. 进下一轮。

### 4. 四个质量指标(跨 round 画曲线)

| 指标 | 含义 | 论证什么 |
|---|---|---|
| **success rate / round** | 每轮 100 个初始态的成功率 | 自我改进有效(曲线升) |
| **Pass@5** | 100 个初始态里、5 次探索内能解出的比例 | 探索覆盖广 |
| **exploration yield** | 每轮成功探索 rollout 数 | 探索高产 |
| **jerk** | 动作三阶差分的范数 | 动作平滑 / on-manifold |

外加 SNR 诊断:`SNR_i = Var(μ_i)/E[σ_i²]`,只对 `SNR > 0.05` 的"有效维"扰动 → 可控探索。

### 5. 基线

Ours(SOE)vs SIME vs 纯 DP;四任务(Can/Lift/Square/Transport)× 4 seed。

---

## 二、SCOUT 怎么评估

分两层:① 可行性(action 级,快);② 优秀性(task 级,慢,终极)。

### 1. 可行性(action 级)—— guidance 机制本身成立吗

四判据:

| 判据 | 测法 | 过线 |
|---|---|---|
| z 可控 | 固定 s 变 z,动作 std 随 `guidance_scale` 升;`scale=0` 时 ≈ 0 | 多样性单调升 |
| 命中 | 引导动作编码回去 `z_θ(s,a) ≈ z` | 一致性误差降 |
| 方向对 | Cost 随去噪步降(若升 → `guidance_scale` 取负) | cost 递减 |
| on-manifold | 动作可执行、jerk 低、偏离 DP 默认适中 | — |

**生死诊断(先于一切)**:量 `‖∂z_θ/∂a‖`(z_θ = reparam 采样的 skill = p_θ(s̄_t,a);**非均值 μ**)。

- guidance 梯度 `∇_a‖z − z_θ(s,a)‖² = −2(z−z_θ)·(∂μ/∂a + ε·∂σ/∂a)` —— **双通道**(μ-敏感度 + σ-敏感度×ε)。cost 用 reparam 采样 z_θ 而非均值 μ,正是为了在 KL 压制 ∂μ/∂a 时,∂σ/∂a 通道仍可传动(idea.md 原始定义即 p_θ,非 μ;旧版文档误写成 μ,已纠正)。
- β 太大 → μ,σ 都不随 a 变 → guidance 是 no-op → 探索死。但 σ 通道通常比 μ 更抗 KL(KL 是边缘约束 E[σ²]→1,逐点 ∂σ/∂a 可存活)→ 用 z_θ 诊断比 μ 更不容易误判"guidance 已死"。
- 做法:对单个 ckpt,固定一组 ε(或多组取均值降方差),算平均 `‖∂z_θ/∂a‖`,看「a 抖一个 std 时 z_θ 的相对移动」(敏感比);≪ 1 ⇒ 这版 β 太大,降 β 重训。
- 要画「敏感度 vs β」曲线 ⇒ 训多个 β 的 ckpt,各算一次。

### 2. 优秀性(task 级)—— 比基线更有用吗

直接复用 SOE 的脚手架(同四任务、同 100 初始态、同 multi-round),对照 **SOE / SIME / action-noise / 纯 DP**。在 SOE 的四指标上:

| 指标 | SCOUT 该赢吗 | 为什么 |
|---|---|---|
| success rate / round | 该(更快 / 更高) | 有向探索,每轮发现更多有用 demo |
| **Pass@5** | **最该** | 不同 z 奔向不同区域 → 覆盖更广 |
| exploration yield | 该 | guidance 比随机抖动高产 |
| jerk | 该、且好讲 | guidance 叠在 DP 去噪上 → 天然 on-manifold |

### 3. 差异化(SOE 没有显式指标的空白)

SOE 没有显式的 diversity / coverage / novelty 指标,靠 Pass@5 + yield 间接论证。SCOUT 因为 z 是语义 skill,可以加 SOE 做不到的显式指标:

- **可解释**:把 z 各方向与产生的行为变化做相关(如 z 某维 ↔ 抓取高度 / 水平位置)。
- **可定向**(最强卖点):用 SNR / 覆盖分析找出"欠探索 skill",故意把 z 偏过去,看 SCOUT 能否定向补齐——SOE 的无向抖动做不到。

---

## 三、落地对照

| 层 | 代码 | 阶段 | 过了说明 |
|---|---|---|---|
| 可行性 | action 级四判据 + `‖∂z_θ/∂a‖` 生死诊断(reparam 双通道) | 阶段 1 | guidance 机制成立 |
| 优秀性 | 接进 SOE `run.py` / `run_full_multi_round.py` | 阶段 2 / 3 | 探索优秀(终极证明) |

> SCOUT 与 SOE 的本质区别决定评估重心:SOE 靠"抖得多样且安全",SCOUT 要证"指哪打哪 + 覆盖更广"。所以 SCOUT 的头条差异化指标是 **Pass@5(覆盖)** 与 **可定向探索**。
