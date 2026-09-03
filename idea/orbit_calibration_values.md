# orbit 有量纲标定数值总表(2026-09-02,供无量纲化参考)

> 各任务**现行手标**的 orbit/entropy 引导参数与实测读数。无量纲化(方案 1,`idea/orbit_calibration_protocol.md`)的目标:从这张表的有量纲值推出跨任务通用的无量纲参数,新任务免标。
> 通用固定:λ=0.5(无量纲反馈增益,跨任务不动)、guide=orbit、DDPM 100 步、env50、rescue×10。

## SQUARE(robomimic square,seed233,DP-base 599 + dyn-base 2026_8_26)

| 参数 | 值 | 来源 |
|---|---|---|
| η (guidance_scale) | 3.0 | can/square entropy campaign 共用,eval config |
| κ (atypical_cap) | 2.5 | 同上 |
| δ (orbit-delta) | 0.25 | 08-31 定标 |
| σ_orb | **0.25(甜点)** | 62 失败定集扫描:0.25 优于 0.50 |

实测读数(σ=0.25):62 定集救回 36、p@10 0.74(100 场景含 baseline 首试)、jerk 0.471、PR 1.38、μ̄ 两两距 0.188;探针 phase-2 占比 ≈48%;链 r1(2026_9_1_orbchain)mean_inject ≈0.6–0.7(A/B 实测 0.60–0.72)、jerk 0.168(explore 口径)、eval 0.38。
σ 扫描(6 场景探针):救回 4/3/5 @ σ=0.1/0.25/0.5;noise 注入/行 0.79/1.98/3.90;jerk 0.452/0.458/0.649。定集(62):σ0.50 = 32 救回 / p@10 0.70 / jerk 0.647。
⚠️ 小样本探针(6 场景)给出 σ0.5 最优,定集反转——pass@10 小样本不是可靠标尺。

## CAN(robomimic can,seed233,DP-base 599 + dyn-base 20260824-232156;2026-09-02 标定)

| 参数 | 值 | 来源 |
|---|---|---|
| η | 3.0 | entropy campaign can 标定(沿用) |
| κ | 2.5 | 同上 |
| δ | 0.25 | 沿用 square |
| σ_orb | **0.10(甜点)** | 本次 4 组扫描(seed42-61 前 20 场景 ×10) |

实测读数(20 场景口径,分母含 baseline-solved 场景):

| σ | pass@10 | jerk | collected(成功重试/200) |
|---|---|---|---|
| 0.05 | 0.95 | 0.710 | 44 |
| **0.10** | **0.97** | 0.717 | **49** |
| 0.25 | 0.96 | 0.757 | 47 |
| 0.50 | 0.91 | 0.950 | 25 |

- CAN 链 r1(σ0.25 原样迁移)遥测:mean_inject=1.70、phase-2 占比 71%(square 同配置 0.6–0.7/48%)=偏热实证;20 场景 pass@10 0.96 仍可用但 jerk/collected 劣于 σ0.10。
- **跨任务观察:σ 甜点 can 0.10 ≈ square 0.25 的一半**,与 can VIB 梯度大(exploit campaign 实测 7–43×)、壳更易够到一致。

## TOOL_HANG(seed233,orbit-dev,2026-09-01/02)

| 参数 | 值 | 来源 |
|---|---|---|
| η | **12.0** | 双模型外推+确认探针(VIB 梯度 ≈ square 的 1/2.5) |
| κ / δ / λ | 2.5 / 0.25 / 0.5 | 沿用 |
| σ_orb | 0.25 | 沿用(未重扫) |

实测:scale=12 → mean_inject 1.27(唯一实测落带点)。教训:η=250 从 square 原样迁移 = 剂量过量全灭;η 跨任务必须做 g 换算(η_new = η_old × g_old/g_new)。

## baseline 参照(wandb 实读)

- CAN-8-24-entropy-s233:DP 臂 r1 eval 0.61 / p@10 0.67(终值 0.68/0.73);SCOUT-atypical r1 0.62 / **0.76**(终值 0.78/0.84)。
- 口径提醒:历史 p@10 分母 = eval 失败集;本次 can 标定分母 = 前 20 场景全量(含 ~40% baseline-solved,饱和抬底)。

## 无量纲化换算目标(方案 1)

- ~~σ = β·√(1−ᾱ_t)·s_a~~ **数据否决**:normalizer 归一动作空间下 s_a≈1(每维 std≈1),σ 的任务差异(2.5×)不是动作尺度差;实测 fb 比 1.4×/p2 比 1.7×/noise-fb 比 3.5× 均对不齐甜点比 → σ 无有效在线归一基准,本轮取几何平均 **σ=0.16**(两任务各自平台内:square 定集 σ0.1-0.5 救回 33-36、can 0.05-0.25 pass 0.95-0.97)。
- **η̃ = η/g̃(已实现,orbit-hparam-dev)**:g̃ = live-climb mean 行梯度范数(kl<κ−δ, norm>1e-4,NaN→0 剔除),归一后逐行 clamp 3× 名义;η̃=0.33(对齐 square 旧 mean_inject 1.05 的换算)。η 本身实测两任务近似 robust(行梯度比仅 1.17×,"7-43×"是 exploit NLL 旧数);η̃ 的真正卖点是 toolhang 类梯度差 4× 任务免手标。
- κ = p90 暂缓(core 分布与在线工作点分布语义不一致;κ=2.5 三任务实测可用);δ=0.25、λ=0.5 共用。
- **实测落地(commits f54d2a2→6ec56e6→cf8920f,三轮 review/遥测迭代)**:中段遥测 square mean_inject 0.71/can 0.48(旧带 1.05/1.24 偏低但方向健康)、mean_g_med 0.18/0.20、noise 1.28 双任务对齐、max 3.7(row cap)。

## 20 场景旧基线(对比用,2026-09-02 重算)

- square σ0.25 legacy(2026_9_1_orbchain s233 r1 json 前 20 场景):**16/20 = 0.80**(baseline-solved 8 + orbit 救回 8/12)。
- can σ0.10 legacy(本次标定):19/20 = 0.97;σ0.25:18/20 = 0.96。

## 🏁 fb soft-clamp 实验(2026-09-02 深夜,ORBIT-9-2-fbclamp-test,commit 311c695)

option C 落地:Newton 残差 (kl−κ)→δ·tanh((kl−κ)/δ)(远壳拉力饱和到 λδ/‖g‖)+ 切向噪声限带 [κ−δ,κ+δ](randn 先抽后置零,RNG 流跨模式一致)。叠加调度组合(σ_eff=0.16×0.5^(r−1)/η̃0.33/p=2):

| 位置 | aty | 旧 orbit(链) | 调度版(上轮) | **调度+fb clamp** | 判定 |
|---|---|---|---|---|---|
| sq r5(r4 trio) | .99/救19/.51 | 救0(inject 4.2) | .84/救4/.89 | **.97/救17/.51** | ✓(≥17, jerk≤0.8) |
| can r1(base) | .97/救17/.35 | 救14(1.70 偏热) | .97/救17/.58 | **.97/救17/.20** | ✓ 回归守卫,jerk 全场最低 |
| can r2(exp1) | .96/救16/.32 | 救0 | .80/救0/1.84 | **.96/救16/.39** | ✓(0→16,与 aty 打平) |

- fb 遥测:sqR5 0.55→**0.143**、canR2 0.61→**0.073**(λδ/√g_shell 公式吻合:g_shell 2.77/7.49);sat_rows 读出壳饱和度(sq 78%/can 95%)。
- **完整修复链定型**:旧 orbit(0/14/0)→ +σ 轮衰减+η̃+p=2(can r1 14→17,其余仍差)→ +fb clamp(**sqR5 4→17、canR2 0→16,三格全部与 aty 打平 ±2**)。jerk 全部落回 aty 档。
- 最终跨任务固定参数组:**η̃=0.33(eta-dimless)/ σ=0.16×0.5^(round−1) / fb_clamp=soft / noise_anneal=2 / κ=2.5 / δ=0.25 / λ=0.5**。
- 待办:①接链驱动(round_orbit*.sh 传 --orbit-round NUM --orbit-sigma-decay 0.5 --orbit-fb-clamp soft --orbit-noise-anneal 2 --orbit-eta-dimless --guidance-scale 0.33)重启正式链做全链验证;②κ 轮次重标(KL 分布右移的根治项)留远期。

## 🏁 轮次调度实验(2026-09-02 晚,ORBIT-9-2-hparam-test,20 场景×10)

修复组合 = σ_eff=0.16×0.5^(round−1) + η̃=0.33(eta-dimless)+ noise-anneal p=2(commit cafb9f5),对照同位置 aty(η3.0/κ2.5):

| 位置 | aty | 调度版 orbit | 旧 orbit(链) |
|---|---|---|---|
| sq r5(r4 trio) | **0.99 / 救19 / jerk .51** | 0.84 / 救4 / jerk .89 | 0.67 / **救0**(33场景口径 aty 22) |
| can r1(base) | 0.97 / 救17 / .35 | **0.97 / 救17 / .58** | 0.73 / 救14(剂量 1.70 偏热) |
| can r2(exp1) | (历史 .83) | 0.80 / **救0** / jerk **1.84** | 0.67 / 救0 |

- **η̃ 治剂量实证**:sq r5 位置 mean_inject 0.47(旧 orbit 同位置 4.2,爆 8.5×;VIB 梯度膨胀 13.8× 被归一吸收)。
- **can r1 翻盘打平 aty**(17 vs 17)✓;sq r5/can r2 仍差 → **残余破坏者 = Newton 反馈 fb**:壳饱和行(KL≫κ,p2 89-98%)的 fb 步长 ∝ |f−κ| 无上限,每步猛拽回 κ(sq r5 fb=0.55、can r2 fb=0.61 vs 常态 0.24-0.33;jerk 0.89/1.84 高于 aty)。aty 免疫因为 cap 把壳外行梯度清零=壳外零注入。
- **下一步修复候选**(按干净程度):①fb 的 |f−κ| clamp 到 δ 带内(壳外行不加力,对齐 aty cap 语义);②λ 轮次衰减;③后期直接退火到 --guide atypical(σ→0 且关 fb)。
- ⚠️ sq r6 位置未跑:DP-SCOUT-exp5 ckpt 不存在(square 链 r5 retrain 未完成即停,链归并行会话管)。

## 🏁 无量纲参数验证结果(2026-09-02,orbit-hparam-dev @cf8920f)

固定一组跨任务参数:**η̃=0.33(eta-dimless) / σ=0.16 / κ=2.5 / δ=0.25 / λ=0.5**,seed233,seed42-61 前 20 场景 ×10:

| 任务 | 旧(逐任务手标) | 新(同一组无量纲参数) | jerk 新 | 遥测(mean_inject / g_med / noise) |
|---|---|---|---|---|
| square | σ0.25:**0.80** | **0.96** | 0.321(旧 0.45-0.65) | 0.62 / 0.17 / 1.28 |
| can | σ0.10:**0.97** | **0.98** | 0.605(旧 0.71-0.95) | 0.49 / 0.22 / 1.28 |

结论:**两任务"基本吻合"达标**(can 持平,square +3 场景——20 场景二项噪声 ±2 内,方向为正),jerk 双双更平滑,collected 更多(sq78/can68)。同一组参数零任务内标定直用。残差:dimless 剂量(mean_inject 0.49-0.62)系统性低于旧带(1.05-1.24)但 pass@10 不劣——旧剂量偏保守;η̃ 若要严格对齐旧带可升到 ~0.5,平台内不敏感。注意单次运行、square 提升不宣称显著。

实现三轮迭代(全过 subagent review):(1)f54d2a2 初版 batch-median → (2)6ec56e6 live-climb mean(can 遥测 0.35 揭示 shell 行污染 median 3× 饥饿)+ 独立重算 check17 → (3)cf8920f 逐行 3× cap(handover 大梯度行遥测爆 33-70)。最终口径:η̃·√(1−ᾱ)·min(‖g‖/g̃, 3)。
