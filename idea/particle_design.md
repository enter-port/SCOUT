# particle(并行粒子互斥)实现设计 — 待用户审核,未落代码

- 日期:2026-08-30 | 分支目标:`entropy-random-dev`(待建/续用 worktree)
- 上游:`idea/escape_coverage_research.md` M-`particle`;用户 2026-08-30 拍板:砍 ladder,第一波 = particle 三组时序消融。
- 本文档 = 红线 [[code-requires-user-approval]] 要求的实现设计稿。**「待拍板点」小节每一条都需要用户确认后才写代码。**

---

## 1. 机制推导链(理论目标 → 注入公式,逐步标注)

**目标**(用户原话):重试往所有能远离当前分布的方向走,而不是只走最远方向。

**Step 1(目标,恒等改写)**:把"10 条重试各自逃逸"重写为"10 个粒子从同一倾斜分布联合采样":
  q({aᵢ}ᵢ₌₁..K) ∝ ∏ᵢ p_DP(aᵢ|s̄) · e^{β·cost(aᵢ)} · ψ(a₁..a_K)
ψ = 1 时联合分布退化为 K 个独立的 p\* ——即现状(i.i.d.,配合寻模塌缩 → 窄锥)。要覆盖所有方向,需给 ψ 一个**惩罚近重复**的因式。

**Step 2(ψ 的选择,代理)**:取 Gibbs 斥力势 ψ = exp(−λ·Σ_{i<j} k(f(aᵢ), f(aⱼ))),k = RBF 核 exp(−‖·‖²/2h²),f = VIB 编码器均值 μ(16 维/chunk,与 PR/d_act 宽度指标同空间)。这一步是**代理**:Particle Guidance(Corso et al., ICLR 2024)的联合势在扩散采样中的标准取法;核形式选 RBF 是为了(ⅰ)光滑可导、(ⅱ)带宽 h 显式控制斥力作用半径。

**Step 3(对粒子 i 的力,恒等)**:联合目标对 aᵢ 的梯度只经过 cost_i 和含 i 的核项:
  ∇_{aᵢ} log q = ∇log p_DP + β·∇cost_i − λ·Σ_{j≠i} ∇_{aᵢ} k(f(aᵢ), f(aⱼ))
即**每条粒子的注入力 = 现行 entropy cost 力 + 斥力项**。关键性质:cost 项原样保留 → 每条粒子的 DP 一致度预算不变(护栏内建);斥力只改"去哪个峰",不减"逃逸预算"。

**Step 4(其他粒子冻结,代理)**:Particle Guidance 的做法——对粒子 i 求梯度时 aⱼ(j≠i)detach。实现上天然成立:斥力在 `compute_loss` 内按 batch 行算,`autograd.grad(loss, trajectory)` 只对当前去噪变量求导,batch 内其他行是常量。等价于对联合分布做坐标上升式(逐粒子)的一步,而非全联合梯度。

**Step 5(注入时机时序化,新增自由度)**:斥力项乘门控 1{n ≥ pg_start},n = 去噪循环步索引(0-based,DDPM 100 步循环 `for t in scheduler.timesteps`,gst=100 全程)。三组:
- **G1 pg_start=0**:从第一个去噪步互斥——粒子在 x_T 就分叉,方向多样性最大,但早期 x̂₀ 还很模糊,斥力作用在"意图未成型"的候选上;
- **G2 pg_start=50**:前 50 步各爬各的熵锥(意图先成型),后 50 步互斥掰开;
- **G3 pg_start=90**:只有最后 10 步互斥——作用在近乎成型的动作上,是"硬掰"极端。
这是**分叉时机 vs 个体成型度**的一维消融,用户 2026-08-30 指定。

**Step 6(κ 封顶与斥力的交互,标注)**:cost 的 min(KL, κ) 封顶只作用于 cost 项;斥力项不封顶(它是"彼此不同"的约束,不是逃逸量)。风险:λ 过大时斥力可越过 κ 的信任域——λ 定标见 §4。

## 2. 与现行代码的对接(逐点,含"为什么能对接")

| 现状 | 粒子模式的改动 |
|---|---|
| `policy.py:247` 去噪循环 batch=B(并发 env 数),`compute_loss(x0_hat, reduction="sum")` 拿到全部行 | **零改动**。斥力在 planner 内按行分组算,autograd 块对角性质由 sum 归约保证(每行梯度=自身 cost 梯度+自身斥力梯度) |
| `set_row_context(init_ids)` 每次 replan 已传行→场景映射(为 novelty 建) | **复用**。斥力只对同 init_idx 的行对算 |
| `job_queue` 混合并行:同场景 10 条重试散在 env 池里,replan batch 内同场景行数随机(0..10) | **组锁定调度**(particle 模式分支):job 以组为单位入队,一组 = 同 init_state 的 K=10 个粒子,env slot 整组分配(n_envs=50 → 5 组并行;空闲 slot <10 时等待不拆组)。理由:互斥力的物理前提是粒子**同时**在去噪,行分组靠调度保证而非碰运气 |
| 掉队:robomimic 成功即 done → 粒子提前退出 | 组内剩余粒子继续(斥力集合自动收缩 = 活着的粒子互斥,语义正确);SOE 口径"ALL retries run,无早停"只指失败场景不提前收兵,成功的粒子 env done 是环境语义,不变 |
| novelty 的 `done_tries` 串行门(重试等前一条完成) | particle **不走此门**(在线互斥 vs novelty 的事后互斥——用户指定的核心差异) |
| `--guide atypical` 经 factories 建 AtypicalCostPlanner | 新增 `--guide particle` → `ParticleCostPlanner` |

## 3. 接口设计

```
run_rollout 新增 CLI:
  --guide particle
  --pg-lambda λ        # 斥力权重(定标见 §4)
  --pg-h-scale c       # 带宽 = c × 组内 μ 两两距离中位数(在线更新,每 replan 一次)
  --pg-start N         # 斥力介入的去噪步(0-based);G1/G2/G3 = 0/50/90
  --failed-set-json F  # explore-only 模式:加载基线失败集,跳过 eval 段(失败集协议,§5)
```

`ParticleCostPlanner`(新文件 `scout/guidance/particle_costs.py`,继承/组合 AtypicalCostPlanner 的 cost 部分):

```
compute_loss(x0_hat, current_obs, reduction="sum"):
    L_cost = AtypicalCostPlanner.compute_loss(x0_hat, ...)          # 逐行,原样
    if self._repulsion_active(去噪步 n, 由 policy 每步回调 set_denoise_step):
        μ = _enc_forward(self, x0_hat)                               # (B,16),复用 entropy_costs._enc_forward
        按行分组(同 init_idx 为一组):
            对组内每对 (i,j): k_ij = exp(−‖μᵢ−μⱼ‖²/2h²)
            L_rep_i = Σ_{j≠i} k_ij                                   # 代理目标:行 i 的斥力 = 斥力势对 μᵢ 求导后的下降方向
        h = c × 组内两两距离中位数(本 replan 批统计)
    return L_cost + λ · Σ_i L_rep_i                                  # sum 归约,块对角
```

注:L_rep 取 Σk 而非 −Σk:梯度注入是 `cond_grad = −∇loss`,loss 里的 +Σk 使注入力方向 = −∇k = 推离(靠近时 k→1、梯度最大,远离时 k→0、力消失)——RBF 斥力的标准形态。

`policy.py` 注入循环改动(约 3 行,鸭子类型,同 select_z 模式):循环顶部 `if hasattr(planner, 'set_denoise_step'): planner.set_denoise_step(n)`。

遥测:`mean_inject` 拆两路——`[guidance-telemetry] cost=… repulsion=…`(斥力幅度单独可见,λ 定标与红线监控都靠它)。

## 4. λ 与 h 的定标方案(待拍板)

- **h**:在线中位数启发式(DvD 同款思想):h = c × 本组当前 μ 两两距离中位数,c 默认 1.0。斥力半径自动适配场景尺度;不引入全局超参。
- **λ**:冒烟定标(2-3 个失败场景,~10 分钟):λ 从小起(0.25),看分路遥测的 repulsion 注入幅度,目标带 = cost 注入幅度的 0.5-1×(同量级但不压倒);λ 上限红线 = repulsion 超过 cost 幅度 2× 时截断。定标值固定后三组共用(时序消融不混 λ 变量)。
- **不做的**:λ 网格扫描(第一波只测时序;λ 扫描视三组结果再说)。

## 5. 实验协议(用户 2026-08-30 指定,此处落成可执行规格)

**资产**:CAN seed233 SCOUT round0 三件套(服务器 `scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can/train/`:DP/DP-base/580.ckpt + dyn/dyn-base/scout_vib.ckpt),η=0.01、κ=2.5、gst=100,rescue×10。

1. **基线定集**:该 DP(unguided,1 try)在 seed42-141 eval → 失败集存 json(预计 ~40 场景,SR≈0.59-0.62)。此失败集 = 后续所有实验的固定测试集。
2. **G0 对照臂(建议增加,待拍板)**:纯 entropy cost(现行,λ=0)同失败集 rescue×10 —— 严格对照必须同三件套同失败集;历史 0.76-0.78 是别处口径,不能直接引用。
3. **G1/G2/G3**:pg_start = 0/50/90,同失败集 explore-only(`--failed-set-json`),各 rescue×10,共用 λ/h 定标值。
4. **判读**:主指标 = 失败集 pass@10(三组 vs G0);宽度指标(PR/d_act/终态散布,组内量)同步记录;护栏 = 救回不差于 G0;jerk/mean_inject 分路遥测监控剂量。
5. **COLLAPSE 场景照旧不作为评判对象**(定理 C)。

## 6. 冒烟与回归(_verify 新 check,写代码前随设计一起批)

- **check 7(逐位还原)**:λ=0(或 pg_start=∞)时,particle 模式与 atypical 模式输出逐位一致(斥力是纯增量项)。
- **check 8(斥力方向)**:构造两行同组、μ 靠近的 batch,断言注入后两行 μ 距离单调增大(互推);不同组行对断言无影响。
- **check 9(时序门)**:pg_start=50 时前 50 步的 loss 与 atypical 逐位一致、第 51 步起出现斥力项。
- **check 10(组锁定)**:调度器分组分配冒烟(hermetic,假 env):10 slot 同组、5 组并行、掉队收缩后斥力集合正确。
- **check 11(掉队记账)**:粒子提前成功退出后,该场景 results 记账与现行 rescue 语义逐字段一致(try_idx、successful_trajs、first_traj)。

## 7. 待拍板点清单(全部确认后才动代码)

1. **pg_start 语义**:0-based 去噪步计数(100 步循环),G2=第 50 步起斥力(即步索引 50-99 共 50 步),G3=步索引 90-99 共 10 步。若你要的是"前 50 步有斥力"请说,我反向。
2. **λ 定标流程**(§4:冒烟 2-3 场景、目标带 = cost 幅度 0.5-1×、三组共用)是否认可;λ 定标值出来后我会先报数再开正式三组。
3. **G0 对照臂**(纯 entropy 同失败集 rescue×10)加不加 —— 我强烈建议加(否则三组之间可比较、但没有同口径基线)。
4. **explore-only 协议**(`--failed-set-json` 跳过 eval 段)是否认可;替代方案是每组完整跑 eval(确定性 → 失败集逐位一致,可自动校验,但每组多 ~1/3 墙钟)。
5. **h 在线中位数带宽**(c=1.0)是否认可。
6. **改动文件清单**:新 `scout/guidance/particle_costs.py`;改 `scout/guidance/policy.py`(+3 行回调)、`scout/eval/rollout_vec.py`(组锁定分支)、`scout/eval/run_rollout.py`(CLI)、`scout/eval/rollout_pipeline.py`(failed-set-json 入口)、`_verify.py`(check 7-11)。是否认可此改动面。
7. 分支/worktree:`entropy-random-dev` 续用(新 worktree `scout-particle`?)还是新分支。
