# aty（KLCostPlanner）超参数细表 —— 按参数逐行 / 按任务分列

> 2026-09-04 整理（用户令）。"—" = 该 campaign 不适用此参数。
> 剂量单位：raw = guidance_scale 直接乘梯度（`scale·√(1−ᾱ_t)·g`，默认）；
> η̃ = dimless（动作空间步长，`--aty-eta-dimless` / orbit 系 `--orbit-eta-dimless`）。
> **用户定案（09-04）：后续 aty 一律使用 raw guidance_scale。**
> 服务器路径根 = `/root/workspace/baojiachun/`。

## 一、参数矩阵（6 个 aty campaign）

| 参数 | can·8-24 entropy | can·aty-try1 | sq·8-26 entropy | sq·beatSOE att | th·9-1 探针（废） | th·p10 现行 |
|---|---|---|---|---|---|---|
| **引导核心** | | | | | | |
| guide 模式 | atypical | orbit（σλδ=0 退化 = aty phase-1） | atypical | atypical | atypical | atypical |
| guidance_scale | 3.0 | 0.33 | 3.0 | 3.0 | 12.0（标定点 s3/s12/s30） | **1.0** |
| 剂量单位 | raw | η̃（dimless） | raw | raw | raw | raw |
| eta-dimless flag | — | ✓（--orbit-eta-dimless） | — | — | — | — |
| κ（--atypical-cap） | 2.5 | 2.5 | 2.5 | 2.5 | 2.5 | 2.5 |
| gst（guidance_start_timestep） | 100 | 100 | 100 | 100 | 100 | **50** |
| **orbit 相位参数** | | | | | | |
| λ（--orbit-lam） | — | 0（退化） | — | — | — | — |
| σ（--orbit-sigma） | — | 0（退化） | — | — | — | — |
| δ（--orbit-delta） | — | 0（退化） | — | — | — | — |
| fb_clamp | — | none（未传） | — | — | — | — |
| noise_anneal | — | —（默认 1） | — | — | — | — |
| **探索协议** | | | | | | |
| explore 模式 | fresh（新场景自蒸馏） | rescue（SOE 口径） | fresh | rescue | rescue | rescue |
| 每场景重试次数 | 1（fresh 协议） | 1（ETRIES=1，pass@1） | 1 | 10 | 10 | 10 |
| 探索范围 | 100 新场景/轮 | 全部失败场景 | 100 新场景/轮 | gate 12 / stage2 62 失败定集 | 10 失败场景 | 9-init 轮转窗口（89 失败集） |
| n_envs | 12 | 25 | 12 | 12（config 默认） | 10 | 25 |
| workers（SHARD_P） | 1（单进程，分片前世代） | 1 | 1 | 1 | 1 | 3/臂 |
| seed（场景/TSEED） | eval 42 / TSEED 233·2333·23333 | eval 42 / TSEED 233 | eval 42 / TSEED 233·2333·23333 | gate 42、stage2 rescue-seed 43 / TSEED 233·2333·23333 | 42 / TSEED 233 | 42 / eval-seed 42（三件套 TSEED 233） |
| **eval 协议** | | | | | | |
| n_init_states | 100 | 100 | 100 | —（探针无 eval 相） | — | — |
| eval try_times | 5 | —（ETRIES=1 即口径） | 5 | — | — | — |
| horizon | 300 | 300 | 500 | 500 | 700 | 700 |
| **训练与轮次** | | | | | | |
| 轮数 | 6 | 6（r6 eval-only） | 6 | —（单轮探针） | —（一次性标定） | —（纯 rollout 探针） |
| DP retrain | 1500ep/轮（config self_improvement） | 300ep/轮 | 1500ep/轮 | — | — | — |
| dyn retrain | 逐轮 | 100ep/轮 | 逐轮 | — | — | — |
| DYN_FREEZE_AFTER | 3（v3 默认） | 6 | 3 | — | — | — |

## 二、orbit v3 对照（can/square 9-2 固定参数组，非 aty）

| 参数 | 值 |
|---|---|
| guide / guidance_scale / 单位 | orbit / **η̃ 0.33**（dimless） |
| κ / gst | 2.5 / 100 |
| λ / δ | 0.5 / 0.25 |
| σ | 0.16 × 0.5^(r−1)（轮衰减 decay=0.5） |
| fb_clamp / noise_anneal | soft / 2 |
| 探索 | rescue ×10、4 worker × 25 env |
| 轮数 / 训练 | 6（r6 eval-only）/ DP 300ep、dyn 100ep、FREEZE=6 |
| seed | TSEED 233（sq 另有 v3 链）、eval 42 |
| wandb / 路径 | `CAN-9-2-orbit-s233` / `SQUARE-9-2-orbit-s233`；`scout-orbit/data/2026_9_2_orbchain/ORBIT-s233/<task>/` |

## 三、项目名与路径（aty 主表 6 列）

| campaign | wandb project | 数据路径（服务器） | 状态 / 主读数 |
|---|---|---|---|
| can·8-24 entropy | `CAN-8-24-entropy`（config 权威） | `scout-entropy/data/2026_8_21_entropy/CAN-entropy-<seed>/` | 完赛；scale 3.0 = entropy_e2e 剂量响应最优（15/39 @ try5） |
| can·aty-try1 | `CAN-aty-try1-s233` | `scout-orbit/data/2026_9_3_atytry1/ATY-s233/can/` | ✅09-04 完赛 5h18m；r6 终值 SR **.72**（对照 9-2-orbit .84）；raw 等效 ≈ 0.33/g_med ≈ 3.3 ≈ 历史 3.0 |
| sq·8-26 entropy | `SQUARE-8-26-entropy`（config 权威） | `scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-<seed>/` | 完赛；s233 六轮 .61→.78 |
| sq·beatSOE att | 无独立项目（探针，json 留档） | `scout-rand/data/particle/sq2_att_<seed>/`、`sq2_conf_s233_seed4{2,3}_gate{1,0}/` | gate 6/12；stage2 **30/62 → pass@10 .68**（8 臂中 orbit σ0.25 36 最强） |
| th·9-1 探针（废） | run 名 `aty-probe-s12-10env` | `scout-orbit/data/2026_9_3_atyprobe/`（probe1_s12 / calib_s3_t2 / calib_s30_t2） | 短标定 s3=3/s12=4/s30=1 非单调；gst100 锚点 clip 坏时代，读数作废 |
| th·p10 现行 | `TOOLHANG-9-4-p10probe` | `scout-orbit/data/2026_9_4_p10probe/`（r1 三件套只读：`scout-orbit/data/2026_9_1_toolhang/TOOLHANG-s233/tool_hang/`） | 进行中（R11-R18）；**s1 池化 32/62 vs 同窗 DP 28/62（+4）**；s0.15/s1.2/s12/cap50/cap100 均更差 |

## 四、剂量换算锚（th / κ2.5 / gst50，本日实测）

| 量 | 值 |
|---|---|
| M̄（aty live-climb 均值 g_med） | 1.95–2.05（[kl-telemetry] 实测） |
| 换算律 | η̃ = s · g_med（每步精确，误差 4.8e-7） |
| 固定 η̃ 活体带 | η̃1→0.50；η̃2→0.75；raw s1→0.35（带匹配 η̃≈0.4–0.7） |
| can/sq 的 g_med | 未留盘（量级锚 O(0.1)） |

## 备注

1. `configs/eval_tool_hang_entropy.yaml` 仓库现值仍为 **12.0 / gst100**（旧
   PROVISIONAL）；th·p10 现行的 1.0/gst50 由探针 per-arm 副本承载，config 正式
   落地待拍板。
2. gst 差异（can/sq 100 vs th 50）：th 的 50 = 锚点 a⁰ 修复（t49 收敛 x̂₀），
   can/sq 标定在 100 下工作正常、未做该修复。
3. raw 数值跨任务不可互套（VIB 梯度尺度差数量级；迁移须按
   [[config-transfer-dose-recalibration]] 现测换算）。
4. can/sq entropy 世代的 retrain 预算 1500ep 取自 config `self_improvement.
   num_epochs`（与 orbit 世代 300ep 单位不同源，仅记录不做横向比较）。

---

# 逐 campaign 参数档案(2026-09-09 整理,用户令)

覆盖:TOOLHANG-9-5-orbit(s233/2333/23333)、CAN-9-2-orbit(s233/2333/23333)、SQUARE-9-1-orbit(s233)、SQUARE-8-30-SOE(s233/2333/23333)、CAN-8-24-SOE(s233/2333/23333)、SQUARE-8-26-entropy(s233/2333/23333)、CAN-8-24-entropy(s233/2333/23333)。权威值以服务器各链 `round.log` / `rollout/*/log/*.json` / wandb config 为准。

## A. SCOUT 系链的公共协议(全部 SCOUT campaign 共用)

| 项 | 值 |
|---|---|
| seed 语义 | TSEED 一籽控全部随机性:core 划分 = `sorted(default_rng(TSEED).choice(200,20))` + DP `training.seed` + dyn `cfg.seed`;CUDA 确定性(cudnn_deterministic + `CUBLAS_WORKSPACE_CONFIG=:4096:8`) |
| eval | 固定 100 场景,场景 i = `np.random.seed(42+i)` 后 reset(seed 42..141);DDPM 100 步采样器 |
| explore(XMODE=soe) | eval 失败场景 → 同初态 rescue 重试 ×ETRIES(=pass@K 的 K);数据回灌累积 |
| 数据规则 | DP 重训用 `success_accum.hdf5`(core+全部探索成功);dyn 重训用 `all_accum.hdf5`(core+全部轨迹);每轮重建 |
| retrain | DP 300ep scratch(ckpt 149/299);dyn 100ep(`scout_vib.ckpt`);DP 重训无条件执行 |
| DYN_FREEZE_AFTER | 6(dyn 每个 full 轮都重训) |
| round0 | seeded 20/200 core 划分 + base DP 600ep(ckpt 99..599)+ dyn-base |
| 一轮结构 | [1/3] rollout(eval+explore)→ [2/3] DP retrain → [3/3] dyn retrain(仅 SCOUT 臂) |
| wandb | 7 键:`eval/success_rate`、`explore/pass@10`(值=pass@ETRIES)、`DP/epoch`、`DP/loss`、`dyn/KL-loss`、`dyn/mse-loss`、`dyn/epoch` |
| VIB 训练模板 | β=3e-5、free_bits=0.005、failure_weight=1;VIBEncoder 输入 LayerNorm;lr warmup 5ep+cosine;steps_per_epoch=100 |
| pass@K 口径坑 | json 键名历史写死 `pass_at_5`,值 = pass@ETRIES |

⚠️ 所有 guidance_scale 数值均为 2026-08-21「1/B 缩放 bug 修复后」的真力度(修复前历史 0.5 ≈ 修复后 0.01)。

## B. TOOLHANG-9-5-orbit(s233 / s2333 / s23333)——tool_hang 三臂 DP / ATY / ORBIT

| 项 | 值 |
|---|---|
| 任务 | tool_hang(单臂 7 维 abs,sideview+手腕双相机 84×84,max_steps=700,horizon 700) |
| core | 40 条(非其他任务的 20) |
| 臂 | ①DP `--guide off` ②ATY(SCOUT)`--guide atypical --atypical-cap 2.5 --guidance-scale 0.5`(raw),gst=50 走 config ③ORBIT 同 ATY 剂量 + `--guide orbit --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05 --orbit-sigma-decay 0.5 --orbit-round $NUM --orbit-fb-clamp soft --orbit-noise-anneal 2` |
| ETRIES | 5(pass@5) |
| 并发 | SHARD_P=8/臂 × env25(phase A eval 单体) |
| 轮数 | 6 轮:r1–r5 full + r6:s233=eval-only(SR only,pass@5 事后补测);s2333/s23333=**eval-pk**(同 full 两相 rescue rollout 后跳过 retrain,原产 SR+pass@5) |
| base 三件套 | s233:拷自 TOOLHANG-9-4(DP 599 + dyn-base 20260904-170403);s2333/s23333:fresh round0 真训(DP 600ep ~3.1–3.2h) |
| DP 重训 batch | 09-09 起 `DP_BATCH=256`(仅 [2/3],LR 1e-4 不动);生效:s23333 链 r2 起、s2333 链 r4 起;s233 链全程 b64 |
| 代码/数据 | scout-th95 worktree(orbit-dev@cfca66e→89d31b3);DATA_ROOT=`scout-th95/data/2026_9_5_toolhang/TOOLHANG-s{seed}`;wandb `TOOLHANG-9-5-orbit-s{seed}` |
| 结果(s233 终值) | SR:SCOUT(ATY) .67 > ORBIT .55 > DP .42;p@5:ORBIT .93 > SCOUT .87 > DP .84 |

## C. CAN-9-2-orbit(s233 / s2333 / s23333)——can 单 SCOUT 臂,orbit v3 剂量

| 项 | 值 |
|---|---|
| 任务 | can(image,max_steps=300) |
| 臂/剂量 | 单 SCOUT 臂 `--guide orbit`,v3 固定参数组:`--guidance-scale 0.33 --orbit-eta-dimless`(η̃ 无量纲)+ κ=2.5 + `--orbit-sigma 0.16 --orbit-sigma-decay 0.5 --orbit-round N`(σ 轮程 r1..r5=0.16/0.08/0.04/0.02/0.01)+ fb soft + noise-anneal 2 + λ0.5 + δ0.25 |
| ETRIES | 10(pass@10) |
| 并发 | phase A EVALNENV=25 单体 → phase B SHARD_P=2 × env25 |
| 轮数 | 6 轮:r1–r5 full + r6:s233=eval-only;s2333/s23333=eval-pk |
| base 三件套 | s233:2026_8_21_entropy 世代(DP-base 599 + dyn-base 20260824-232156)+ core 重建([2,14,…,190]/2279 步);s2333/s23333:fresh round0(can DP 600ep 仅 ~25–27min) |
| 代码/数据 | scout-orbit worktree(orbit-dev@e110ffc+3 未提交 M);DATA_ROOT=`data/2026_9_2_orbchain/ORBIT-s{seed}/can`;wandb `CAN-9-2-orbit-s{seed}` |
| 结果 | s233:SR .57/.65/.82/.82/.83/**.84**,p@10 .72/.83/.84/.87/**.90**;s23333:r1 .73/.79→r2 .91→**r3 p@10 .93**(超母链峰值);s2333:r1 .73/.78→r2 .88/.84→r3 p@10 .89 |

## D. SQUARE-9-1-orbit(s233)——square 单 SCOUT 臂,orbit v3(v3 重跑版)

| 项 | 值 |
|---|---|
| 任务 | square(image,max_steps=500) |
| 臂/剂量 | 与 CAN-9-2-orbit 完全相同的 orbit v3 参数组 |
| ETRIES / 并发 | 10;phase A 25 env;r1–r2 SHARD_P=2、r3 起 SHARD_P=4 × env25(square 失败集大) |
| 轮数 | 6 轮:r1–r5 full + r6 eval-only |
| base 三件套 | 2026_8_26_entropy 世代(DP-base 599 + dyn-base 20260826-112119) |
| 历史 | 同名 v1 链(08-31~09-02,旧剂量 σ0.25 固定、无 η̃)r5 壳饱和 0 救回作废;本条为 09-02 20:52 v3 分片重跑(wandb 旧 run 已清) |
| 代码/数据 | scout-orbit worktree;DATA_ROOT=`scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square`;wandb `SQUARE-9-1-orbit-s233` |
| 结果 | SR .37/.53/.66/.64/.71/**.72**;p@10 .70/.72/.81/.79/**.89**(r5);救回 33/19/15/15/18 |

## E. SQUARE-8-30-SOE(s233 / s2333 / s23333)——SOE square 基线,DDIM eta=1 重跑版

| 项 | 值 |
|---|---|
| 方法 | 学长 SOE(DPExt:action reconstruction + 潜空间加噪),SCOUT 对齐协议 |
| 背景 | 8-29 世代 DDIM eta=0(给定 x_T 完全确定)与我方 DDPM 不可比 → 08-30 改 `square_soe.json` `"eta": 1.0` 原协议重跑(commit bb3c59c) |
| 协议对齐 | 同 TSEED core 划分、同 100 场景(seed42+i)、rescue 同初态 ×10、累积回灌;SOE 内部 round 0..5 ↔ SCOUT r1..6 |
| SOE 架构/训练 | num_obs=1、chunk 20、DDIM 20 步(eta=1)、batch 64、lr 3e-4、readout 64→style 16、kl_w 1e-3、noise 2.0;训练 1000ep×100it/轮(与 SCOUT 300ep 全数据单位不同,已知差异) |
| square 特有 | HORIZON=500;VISGATE=0(渲染门关);SAVE0=200 SAVE=200;PRUNE_ACCUM=1;数据 `soe_data/datasets/square/{image_v141_abs_6drot.hdf5, square_core_soe_s{seed}.hdf5}`(与 SCOUT core 逐位 PASS) |
| 代码/数据 | `SOE/`(fork soe-scout-align)+ `SOE_scripts_2/` + `.venv_soe`;DATA_ROOT=`soe_data/2026_8_30_soe/SOE-s{seed}`;wandb `SQUARE-8-30-SOE-s{seed}` |
| 结果 | r6 终值 SR .65/.57/.45,p@10 .93/.82/.73(s233 p@10 为 square 全场最高);r1 靶子 .77/.74/.67。(8-29 eta=0 世代存档:SR .58/.55/.54,p@10 .83/.86/.83) |

## F. CAN-8-24-SOE(s233 / s2333 / s23333)——SOE can 基线

| 项 | 值 |
|---|---|
| 方法/协议 | SOE,SCOUT 对齐(同 E 的划分/场景/rescue×10/回灌;SOE round k ↔ SCOUT r k+1) |
| SOE 架构/训练 | 同 E,但用 `can_soe.json`(SIME can_image.json 重建);HORIZON=300;**eta 保持默认 0**(can 世代未改——与 square eta=1 重跑不同,跨任务 SOE 绝对值比较需注意采样器差异) |
| 瘦身 | s23333(08-29 补跑)带 SAVE0=200 SAVE=200 PRUNE_ACCUM=1;s233/s2333(08-26)原版 |
| 代码/数据 | 同 E 基建;DATA_ROOT=`soe_data/2026_8_26_soe/SOE-s{seed}/can`;wandb `CAN-8-24-SOE-s{seed}` |
| 结果 | s233:SR .48→.70→.67→.74→.72→**.75**,p@10 .74/.76/.79/.80/.79/.81;s2333:SR 终 .65,p@10 终 .84;s23333:.56→.67→.75→.66→.67→(终轮读数以 wandb 为准,p@10 带 .85–.88)。终值 ≈SCOUT-DP 臂、低于 SCOUT 最低值 |

## G. SQUARE-8-26-entropy(s233 / s2333 / s23333)——SCOUT entropy cost,square 两臂

| 项 | 值 |
|---|---|
| 任务 | square(image,max_steps=500) |
| 臂 | ①SCOUT `--guide atypical --atypical-cap 2.5`(entropy cost,内部代号 atypical)②DP `--guide off` |
| 剂量 | η=guidance_scale 3.0(square 探针定稿:旧 VIB 不咬合、η8 两崩)+ κ=2.5 + gst=100(config `eval_square_entropy.yaml`);dyn config `vib_square_exp1.yaml`(β3e-5/fb0.005/fw1) |
| ETRIES / env | rescue×10;env50(r1 曾因渲染门误判短暂 env6/12,08-26 16:39 拆门后全臂 env50;vis_validate 门 square 世代已禁用) |
| 轮数 | 6 轮全 full |
| base 三件套 | fresh round0;s233/s2333 08-26 启动,s23333 组 08-29 启动(双臂) |
| 代码/数据 | scout-entropy worktree(round_entropy.sh);DATA_ROOT=`scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s{seed}`;wandb `SQUARE-8-26-entropy-s{seed}` |
| 结果 | s233:SR .38→…→.70,p@10 终 .89 vs DP .60/.78;s2333:→.62/.87 vs DP .49/.64;s23333:r3 checkpoint SR .52/p@10 .87(终值见 wandb,SCOUT vs DP 配对) |

## H. CAN-8-24-entropy(s233 / s2333 / s23333)——SCOUT entropy cost 正式实验(can 两臂)

| 项 | 值 |
|---|---|
| 任务 | can(image,max_steps=300) |
| 臂 | SCOUT(atypical/entropy cost)vs DP(off),同 base 三件套配对 |
| 剂量 | η=3.0 + κ=2.5 + gst=100(config `eval_can_entropy.yaml`);can 首发定标世代,后续任务的 3.0/2.5/100 由此迁出 |
| ETRIES / env | rescue×10;env 混合口径:r1–r3 段 12/6(渲染事故降档),08-25 15:07 起 env50 续跑(s233: r3→r6 / s2333: r2→r6 / s23333: r4→r6);同 n_envs 逐位复现、换 n_envs 翻转 ±1–2 场景(已知) |
| 轮数 | 6 轮全 full |
| 轮结构 | round0 = split 20/200 + DP 600ep(~23min)+ dyn-base(TOTAL ~26m);DP 臂跳过 [3/3] |
| 代码/数据 | scout-entropy worktree(commit 27c753b);DATA_ROOT=`scout-entropy/data/2026_8_21_entropy/CAN-entropy-s{seed}`;wandb `CAN-8-24-entropy-s{seed}` |
| 结果 | SCOUT SR 终值 .78/.83/.86 vs DP .68/.74/.77(三 seed 全胜 +9~+10);p@10 .84/.94/.91 vs .73/.81/.83;六轮救回总数 201 vs 132(1.52×);r1 Δp@10 = +9/+7/+9 |

## I. 跨 campaign 易混点速查

- **pass@K**:entropy/orbit 的 can·square 系 = ×10;TOOLHANG 系 = ×5。
- **末轮口径**:SQUARE-9-1-orbit 与 CAN-9-2-orbit 的 s233 = r6 eval-only(只有 SR);TOOLHANG 全部 + CAN-9-2 的 s2333/s23333 = eval-pk(SR+pass@{ETRIES} 原产,09-07 口径定案);entropy 两 campaign 6 轮全 full。
- **guidance 力度单位**:entropy 系 η=3.0 raw 有量纲(内嵌 VIB 梯度尺度,跨任务须换算);orbit v3 η̃=0.33 无量纲(免换算);TOOLHANG-9-5 ATY/ORBIT 用 raw 0.5(toolhang 剂量带)。
- **orbit σ 轮程**:9-1/9-2 = 0.16×0.5^(r−1);TOOLHANG-9-5 = 0.05×0.5^(r−1)(toolhang 定标更小)。
- **SOE 采样器**:can(8-24)eta=0;square(8-30)eta=1;SOE 训练单位 1000ep×100it ≠ SCOUT 300ep 全数据。
- **r6 ckpt 语义**:r6 eval-only/eval-pk 用 exp5 ckpt,与「r6 full 会产出的 exp6」不同代;跨链比较统一按 r5 代 p@10 + r6 SR。
