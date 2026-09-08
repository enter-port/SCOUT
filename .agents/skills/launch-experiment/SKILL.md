---
name: launch-experiment
description: 启动一条新的 SCOUT multi-round 实验链(campaign)。只要用户说「开一个任务 / 开个实验 / 启动 campaign / 跑一条链 / 再来一个变体」,或给出形如 CAN-aty-try1-s233、SQUARE-9-2-orbit-s233 的新实验需求(哪怕只说名字和一两点差异),就用本 skill:交互式问清实验名、任务、seed、轮数、base 来源、guidance 机制与剂量、pass@K、worker 数、GPU 等全部参数,复述确认后走「写脚本 → 拷 base → DRY_RUN → subagent review → 启动 → 挂监控」的完整部署流程。禁止跳过参数询问直接开跑。
---

# 启动 SCOUT 实验链(launch-experiment)

目的:把一次 campaign 启动固化为可重复流程。参考模板(已验证,优先照抄结构):
- `soe_scripts/round_orbit.sh` — 现役 round 驱动(orbit v3 两相,can/square)
- `soe_scripts/round_aty.sh` — phase-1-only 变体(= round_orbit.sh + GEXTRA 单行替换)
- `soe_scripts/th_chain.sh` / `aty_chain.sh` — 链 wrapper(NROUNDS 循环 + done_round 幂等)
- `soe_scripts/th6_restart_launch.sh` / `can_aty_launch.sh` — 启动编排(预拷 base + round0 + tmux 起臂)

服务器事实:`ssh -o BatchMode=yes -p 1022 root@106.14.2.243`;REPO=`/root/workspace/baojiachun/scout-orbit`(脚本在 `soe_scripts/`,旧布局);数据根 `$REPO/data/<campaign>/<NAME-s<SEED>>/<task>/`;venv=`/root/workspace/baojiachun/.venv/bin/python`。

## Step 0 — 现场核实(动手前)

1. 本地:`git branch --show-current && git status --short`(主 checkout 可能被并行会话切走,动文件前必核)。
2. 服务器:`tmux ls`、`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`、`free -g`。**GPU7 禁用(ECC 硬件缺陷)**;GPU/RAM 用量写入后面的参数询问选项。

## Step 1 — 参数收集(必须问,不猜)

用 AskUserQuestion 分组问(每次 ≤4 问,2-3 轮问完),每问给推荐默认。问题字段与默认:

**组 A|身份与规模**
- 实验名(自由输入;惯例 `<TASKUP>-<variant>-s<SEED>`,如 `CAN-aty-try1-s233`)。若用户只说了名字+差异点,先据此填默认再确认。
- task:`can` / `square` / `tool_hang` / `transport`
- seed(TSEED):`233`(若该 seed 的三件套已存在则可复用 base)
- NROUNDS:`6`(rounds 1..N-1 full + 第 N 轮 eval-only)/ `8` / 其他。**末轮 eval-only 的测量口径(用户令 2026-09-07)= SR + pass@{ETRIES} 两样都要**:即末轮也要跑 rescue ×ETRIES 探索测量(用当臂最新 retrain ckpt,冻结失败集→救回→merge 出 pass@K),只是不再 retrain、不回灌;round 脚本须支持此模式(th95 曾因 r6 缺 pass@5 事后补跑 th95_p5meas.sh,勿重蹈)

**组 B|base 来源**(「使用的 base」)
- 从已有 campaign 拷三件套(同 seed 确定性等价,round0 秒过)→ 追问拷贝源(如 `CAN-9-2-orbit-s233` = `data/2026_9_2_orbchain/ORBIT-s233/can`),并现场验证源文件在(core / DP-base 最大 epoch ckpt / dyn-base scout_vib.ckpt)
- 全新 round0(seeded split + DP 600ep + dyn-base,多花数小时)
- 三件套布局:`<src>/<task>/rollout/<task>_core.hdf5` + `<src>/<task>/train/DP/DP-base/checkpoints/<max>.ckpt` + `<src>/<task>/train/dyn/dyn-base/<ts>/scout_vib.ckpt`(+ config.yaml)。注意 ckpt 可能是软链,**cp 用 `-L` 解引用**。

**组 C|guidance 与剂量**
- 臂结构:单 SCOUT 臂(9-2/aty 式)/ 双臂 SCOUT+DP(th6 式)/ 仅 DP 臂
- guide 机制:`orbit v3 两相` / `orbit phase1-only(=atypical)` / `裸 atypical` / `off` / `shell`
- 剂量 η̃(dimless):`0.33`(跨任务免换算);κ(atypical-cap):`2.5`
  ⚠️ 选 `裸 atypical` 时没有 dimless 开关,--guidance-scale 是绝对剂量,跨 ckpt/任务必须先剂量换算([[config-transfer-dose-recalibration]]),要向用户点破这一点。
- pass@K(ETRIES):`1`(pass@1,每失败场景一try)/ `10`(pass@10,SOE 口径)
- SHARD_P(explore worker 数):`1` / `2`(can 默认)/ `4`(square 默认);SHARD_ENVS/EVALNENV 固定 25

**组 D|杂项**
- GPU:从 Step 0 实测的空闲卡里选(报告每卡显存/利用率)
- WPROJ:默认 = 实验名(必须与历史 project 不同名,wandb 隔离)

收集完,**一句话复述关键数值规格**(任务/seed/轮数/guide/剂量/ETRIES/worker/GPU/base 源),有歧义先问再动手(「测 10 个 env」=100→10 条 rollouts 的教训)。

## Step 2 — 写脚本三件套(全部新文件)

红线:**绝不原地覆写任何在跑链会再读的文件**(round_orbit.sh 等现役脚本只读,变体一律新文件名);ssh 传文件走 `tr -d '\r' < 本地 | ssh 'cat > 远端'`,传输与启动分开。

1. `round_<variant>.sh`:服务器端 `cp soe_scripts/round_orbit.sh` + **python 行级替换** GEXTRA 行(勿用 sed 拼变量)+ 文件头注释(diff 与原版对照验证唯一差异)。已验证的三种 GEXTRA:
   - orbit v3 两相:`--atypical-cap "$ATT_CAP" --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma "$ORB_SIGMA" --orbit-sigma-decay "$ORB_SIGMA_DECAY" --orbit-round "$NUM" --orbit-fb-clamp soft --orbit-noise-anneal "$ORB_ANNEAL" --orbit-eta-dimless --guidance-scale "$ETA_TILDE"`
   - phase1-only(=atypical;σ=λ=δ=0 时 run_rollout help 背书 bit-identical to --guide atypical,且保留 dimless 剂量):`--atypical-cap "$ATT_CAP" --orbit-lam 0 --orbit-delta 0 --orbit-sigma 0 --orbit-eta-dimless --guidance-scale "$ETA_TILDE"`
   - off(DP 臂):GUIDE=off,GEXTRA=()
   全部:部署后 `bash -n` + `grep` 逐字段验证(heredoc 部分执行的教训)。
2. `<name>_chain.sh`:照抄 `aty_chain.sh` 结构改五处(task 名、round 脚本名、DATA_ROOT 默认、WPROJ 默认、每轮 env 行的 ETRIES/SHARD_P)。保持 done_round 幂等 + console log + ABORT 语义。
3. `<name>_launch.sh`:照抄 `can_aty_launch.sh`:`cd "$ROOT" || exit 1` + `unset DATA_ROOT` + export 全部 env(NROUNDS/SHARD_P/SHARD_ENVS/EVALNENV/DYN_FREEZE_AFTER/ETRIES)+ 预拷三件套(幂等 `[ -f ] || cp`,cp -L)+ round0 BASE 臂(rc!=0 则**不起臂**)+ `tmux new-session -d -s <会话名>` 起臂。
   本地镜像存 `<repo>/scripts/rounds/`(Write 工具绝对路径,避开 shell 中文路径)。

## Step 3 — 预拷 base + DRY_RUN

1. 预拷三件套到新 DATA_ROOT(见组 B 布局),`ls -la` 核对字节数。
2. `DRY_RUN=1 ... bash soe_scripts/round_<variant>.sh <task> SCOUT 1 full`,从 argv 输出逐项核对:`--guide`、剂量 flags(η̃/κ/σ/λ/δ)、`--explore-try-times $ETRIES`、`shard_rollout.sh $SHARD_P`、`--n-envs 25`、DP/dyn ckpt 路径全在新根、`--wandb-project` 新名、`logging.project=\'$WPROJ\'` 转义正确。DRY_RUN 残留(success_accum 等会被真跑重建)无害,但**不得**对准已有数据的根。

## Step 4 — subagent review(启动前红线)

派 general-purpose subagent 只读 review,任务书附:文件清单与 ssh 读法、scrutiny 清单(变体 diff 纯净性 / P=1 边界 / ETRIES=1 rescue 语义 / base 拷贝完整性 / 链幂等 / 与在跑 campaign 隔离含 heartbeat pkill 模式 / DRY_RUN 残留 / 引号与 set -u)+ **术语表**(orbit/atypical/η̃ dimless/ETRIES/SHARD_P/round0/eval-only/TOTAL 行)。P0/P1 修复并复验后才可启动;P2 记录进汇报。

## Step 5 — 启动与确认

1. 终查 GPU 空闲 + RAM(重 worker 容量法则 ≤6-8;再核 GPU7 仍无人误用)。
2. `bash soe_scripts/<name>_launch.sh` → 确认 round0 rc=0(base 拷贝源则 TOTAL 0m0s)+ tmux 会话出现 + round1 START 行。
3. 抓引导 banner(`grep guidance_scale rollout.stdout`)确认剂量覆盖生效,报告 GPU 开始吃显存。

## Step 6 — 收尾

1. 挂监控 cron:CronCreate 用 `recurring=false + maxRuns=<N>` 有限次(每小时),prompt 内嵌「只读;未经新令不得 kill/启动/修改;快照可能过时以服务器实况为准」+ 完赛/异常判定与诊断收集指令。
2. 持久 memory:写 campaign 文件(任务定义/实现决定/脚本位置/监控 id/ETA)+ MEMORY.md 索引行。
3. 汇报用户:三点差异如何落地(机制理由)、base 复用说明、review 结论与 P2 清单(如 wandb backfill 键名沿用 pass@10 的语义说明)、ETA。git commit 是否做听用户(改动先过审红线)。

## 红线速查(全文见 AGENTS.md)

代码参数未经用户确认不落实(本 skill 的 Step 1 就是履行它)|禁原地覆写在跑文件|禁私杀训练/rollout|GPU7 禁用|服务器只动 baojiachun|ssh pkill 模式加字符类防自匹配|同一只读调用 >2 次=病态循环|自造词首现解释并入 NAMES.md(NAMES.md 不在当前分支则汇报说明)。
