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
