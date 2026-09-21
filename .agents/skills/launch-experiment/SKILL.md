---
name: launch-experiment
description: 启动并跑完一条 SCOUT multi-round 实验链(campaign 全生命周期)。只要用户说「开一个任务 / 开个实验 / 启动 campaign / 跑一条链 / 跑一个 task / 再来一个变体」,或给出形如 CAN-aty-try1-s233、SQUARE-9-2-orbit-s233 的新实验需求(哪怕只说名字和一两点差异),就用本 skill:交互式问清实验名、任务、seed、轮数、base 来源、guidance 机制与剂量、pass@K、worker 数、GPU 等全部参数,复述确认后走「写脚本 → 拷 base → DRY_RUN → subagent review → 启动 → 挂监控 → 完赛收账」的完整流程。启动后本 skill 的监控/事故/收账各节继续适用,直到出终表。禁止跳过参数询问直接开跑。
---

# 启动并跑完 SCOUT 实验链(launch-experiment)

目的:把一次 campaign 的完整生命周期(接令→启动→在飞→收账)固化为可重复流程。

**参考模板**(本地 repo `scripts/` = 服务器同源;优先照抄结构,变体一律派生新文件):
- `scripts/rounds/round_orbit.sh` — 现役 round 驱动基模板(orbit v3 两相,can/square)
- `scripts/rounds/round_aty.sh` — phase-1-only 变体(= round_orbit.sh + GEXTRA 单行替换)
- `scripts/rounds/round_orbit_canpk.sh` / `round_orbit_sqb256.sh` — **末轮 eval-pk** 版(canpk)/ +DP b256 版(sqb256)
- `scripts/toolhang/round_th95.sh` — 三臂世代 round(BASE / full / eval-only / **eval-pk** 四模式)
- chain wrapper:`scripts/rounds/aty_chain.sh`、`orb233x_chain.sh`、`scripts/toolhang/th95_chain.sh`(NROUNDS 循环 + done_round 幂等 + 末轮调 eval-pk)
- launch 编排:`scripts/rounds/can_aty_launch.sh`、`scripts/launches/orb2333_launch.sh`、`scripts/toolhang/th95_s2333_launch.sh`(真 round0 + rc=0 后自动开臂)

**服务器事实(2026-09-12 收编后)**:`ssh -o BatchMode=yes -p 1022 root@106.14.2.243`;REPO=`/root/workspace/baojiachun/scout`(主 repo,checkout main,与云端同步);**旧 12 个分身 worktree 已全删(09-13)**,脚本唯一真本=repo `scripts/`,baojiachun 级遗留 `soe_scripts/` 只是历史副本勿再写入;数据根 `$REPO/data/<campaign>/<NAME-s<SEED>>/<task>/`,**找历史数据先读该目录 `PROVENANCE.md`**;venv=`/root/workspace/baojiachun/.venv/bin/python`。

## Step 0 — 现场核实(动手前)

1. 本地:`git branch --show-current && git status --short`(主 checkout 可能被并行会话切走,动文件前必核)。
2. 服务器:`tmux ls`、`nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader`、`free -g`。**GPU7 禁用(ECC 硬件缺陷)**;GPU/RAM 实测写入后面的参数询问选项。
3. 若任务/seed 的 base 三件套可能已存在:先查 `scout/data/` 内同 seed campaign(+ PROVENANCE.md),穷举已有资产再给「拷贝 vs 新训」选项。

## Step 1 — 参数收集(必须问,不猜)

用 AskUserQuestion 分组问(每次 ≤4 问,2-3 轮问完),每问给推荐默认。问题字段与默认:

**组 A|身份与规模**
- 实验名(自由输入;惯例 `<TASKUP>-<variant>-s<SEED>`,如 `CAN-aty-try1-s233`)。若用户只说了名字+差异点,先据此填默认再确认。
- task:`can` / `square` / `tool_hang` / `transport`。**全新任务**(数据都没有)先走数据管线:下载 low_dim_v141 → `dataset_states_to_obs` 出图 → robomimic_dataset_conversion 出 abs_actions → `split_core.py` 抽 core(见 AGENTS.md [[scout-data-pipeline]])。
- seed(TSEED):`233`(同 seed 三件套已存在则可拷 base)
- NROUNDS:`6`(rounds 1..N-1 full + 第 N 轮 **eval-pk**)/ `8` / 其他。**末轮口径(定谳)= SR + pass@{ETRIES} 两样都要**:eval-only 分支写死 n_tries=1 出不了 pass@K,必须走 eval-pk 模式——与 full 完全相同的两段 rescue rollout(phase A 首试 SR+冻结失败集 → phase B 分片 rescue ×ETRIES → merge 出 pass@K),仅跳过 [2/3]+[3/3] retrain、不回灌(th95 曾因 r6 缺 pass@5 事后补测,勿重蹈)。

**组 B|base 来源**(「使用的 base」)
- 从已有 campaign 拷三件套(同 seed 确定性等价,round0 秒过)→ 追问拷贝源(如 `data/2026_9_2_orbchain/ORBIT-s233/can`),现场验证源文件在
- 全新 round0(seeded split + DP 600ep + dyn-base;can ~27min / toolhang ~3.3h / square 更慢)
- 三件套布局:`<src>/<task>/rollout/<task>_core.hdf5` + `<src>/<task>/train/DP/DP-base/checkpoints/<max>.ckpt` + `<src>/<task>/train/dyn/dyn-base/<ts>/scout_vib.ckpt`(+ config.yaml)。ckpt 可能是软链,**cp 用 `-L` 解引用**。

**组 C|guidance 与剂量**
- 臂结构:单 SCOUT 臂(9-2/aty 式)/ 双臂 SCOUT+DP(th6 式)/ 三臂 DP+ATY+ORBIT(th95 式)
- guide 机制:`orbit v3 两相` / `orbit phase1-only(=atypical,走退化 orbit 实现)` / `裸 atypical` / `off(DP 臂)`
- 剂量:orbit v3 固定组 **η̃0.33(dimless)/ κ2.5 / σ0.16×0.5^(r−1) / fb soft / anneal2 / δ.25 / λ.5**(一组参数双任务免标定)。
  ⚠️ 选 `裸 atypical` 时没有 dimless 开关,--guidance-scale 是绝对剂量,跨 ckpt/任务必须先剂量换算([[config-transfer-dose-recalibration]]),要向用户点破。
- pass@K(ETRIES):`1` / `5` / `10`(SOE 口径)
- SHARD_P(explore worker 数)×25env:can 默认 2 / square 4 / toolhang 8。**容量口径(09-13 用户确认)**:flush_every 缓存世代 worker RSS 仅 17-22G,**8 worker/臂一直没问题**(09-08 实测 24 worker×25env 零 OOM);只有旧无 flush 代码才需要 2-worker 包线。性能上每卡甜点 3-4 分片,P=8 单卡转 GPU-bound(th95 接受,换 rescue 墙钟)。
- DP retrain batch:新链默认 **b256**(09-09 拍板,LR 1e-4 不动,单轮省 ~45%);旧 b64 链续跑勿中途混换(链内可比性)。

**组 D|杂项**
- GPU:从 Step 0 实测的空闲卡里选(报告每卡显存/利用率)
- WPROJ:默认 = 实验名(必须与历史 project 不同名,wandb 隔离)

收集完,**一句话复述关键数值规格**(任务/seed/轮数/臂/guide/剂量/ETRIES/worker/batch/GPU/base 源),有歧义先问再动手(「测 10 个 env」=100→10 条 rollouts 的教训)。

## Step 2 — 写脚本三件套(全部新文件)

红线:**绝不原地覆写任何在跑链会再读的文件**(现役脚本只读,变体一律新文件名;要改共享脚本走「写新 + `mv` 原子替换」,且换前清点 pre-swap 读者——CPFS 上旧 inode 读者可能报 Stale file handle 整链 ABORT);ssh 传文件走 `tr -d '\r' < 本地 | ssh 'cat > 远端'`,**传输与启动分开**两条 ssh。

1. `round_<variant>.sh`:`cp` 基模板 + **python 行级替换** GEXTRA 行(勿用 sed 拼变量)+ 文件头注释(diff 与原版对照验证唯一差异)。已验证的三种 GEXTRA:
   - orbit v3 两相:`--atypical-cap "$ATT_CAP" --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma "$ORB_SIGMA" --orbit-sigma-decay "$ORB_SIGMA_DECAY" --orbit-round "$NUM" --orbit-fb-clamp soft --orbit-noise-anneal "$ORB_ANNEAL" --orbit-eta-dimless --guidance-scale "$ETA_TILDE"`
   - phase1-only(=atypical;σ=λ=δ=0 时 bit-identical to --guide atypical,且保留 dimless 剂量):`--atypical-cap "$ATT_CAP" --orbit-lam 0 --orbit-delta 0 --orbit-sigma 0 --orbit-eta-dimless --guidance-scale "$ETA_TILDE"`
   - off(DP 臂):GUIDE=off,GEXTRA=()
   b256 版参照 `round_orbit_sqb256.sh`(DP_BS=256 注入 train.py 四处);eval-pk 模式参照 `round_th95.sh`。部署后 `bash -n` + `grep` 逐字段验证(heredoc 部分执行的教训)。
2. `<name>_chain.sh`:照抄 `aty_chain.sh` / `orb233x_chain.sh` 结构改五处(task 名、round 脚本名、DATA_ROOT 默认、WPROJ 默认、每轮 env 行的 ETRIES/SHARD_P)。保持 done_round 幂等(靠 round.log TOTAL 行跳过已完成轮)+ console log + **末轮调 eval-pk** + ABORT 语义。
3. `<name>_launch.sh`:照抄 `can_aty_launch.sh` / `th95_s2333_launch.sh`:`cd "$ROOT" || exit 1` + export 全部 env(NROUNDS/SHARD_P/SHARD_ENVS/EVALNENV/DYN_FREEZE_AFTER/ETRIES)+ 预拷三件套(幂等 `[ -f ] || cp`,cp -L)+ round0 BASE 臂(rc!=0 则**不起臂**)+ `tmux new-session -d -s <会话名>` 起臂。
   本地镜像按类别存 `scripts/rounds|launches|<task>/`(Write 工具绝对路径,避开 shell 中文路径)。

## Step 3 — 预拷 base + DRY_RUN

1. 预拷三件套到新 DATA_ROOT(见组 B 布局),`ls -la` 核对字节数。
2. `DRY_RUN=1 ... bash <round脚本> <task> <ARM> 1 full`(另补一发末轮 eval-pk 对 smoke 根),从 argv 输出逐项核对:`--guide`、剂量 flags(η̃/κ/σ/λ/δ)、`--explore-try-times $ETRIES`、`shard_rollout.sh $SHARD_P`、`--n-envs 25`、DP/dyn ckpt 路径全在新根、`--wandb-project` 新名、`logging.project=\'$WPROJ\'` 转义正确、b256 注入。DRY_RUN 残留(success_accum 等真跑会重建)无害,但**不得**对准已有数据的根。

## Step 4 — subagent review(启动前红线)

派 general-purpose subagent 只读 review,任务书附:文件清单与 ssh 读法、scrutiny 清单(变体 diff 纯净性 / ETRIES=1 等 rescue 语义边界 / base 拷贝完整性 / 链幂等 / 与在跑 campaign 隔离含 heartbeat pkill 模式 / DRY_RUN 残留 / 引号与 set -u / launch 里 `unset DATA_ROOT` 类自杀行)+ **术语表**(orbit/atypical/η̃ dimless/ETRIES/SHARD_P/round0/eval-pk/TOTAL 行)。P0/P1 修复并复验后才可启动;P2 记录进汇报。

## Step 5 — 启动与确认

1. 终查 GPU 空闲 + RAM(8-worker 容量口径见组 C;再核 GPU7 仍无人误用)。
2. `bash <name>_launch.sh` → 确认 round0 rc=0(拷贝源则 TOTAL 0m0s)+ tmux 会话出现 + round1 START 行。
3. 抓引导 banner(`grep guidance_scale rollout.stdout`)确认剂量覆盖生效,报告 GPU 开始吃显存。

## Step 6 — 挂监控(在飞期)

**cron 规则**:`CronCreate` 用 recurring=true + maxRuns=<N> 有限次(如每小时 × 预计轮数),**间隔≤实验单轮时长且轮完即续**;prompt 内嵌「只读;未经新令不得 kill/启动/修改;快照可能过时,以服务器实况为准」+ 完赛/异常判定(完赛分支写幂等 no-op);🔴一会话只能绑一个 scheduled task(再建被拒→CronUpdate 现有 cron 扩职);退役配方=CronUpdate(recurring=false + maxRuns=1 + no-op prompt),CronDelete 不可用时靠 prompt 内嵌自禁用。

**每轮内部结构(判读基准)**:[1/3] rollout = phase A 单体 eval 100 固定场景(seed42..141)出 SR+冻结失败集 → phase B SHARD_P×25env 分片对失败场景同初态重试 ×ETRIES(心跳进 wandb)→ merge 合单三件套 + backfill pass@K;[2/3] DP retrain = success_accum 300ep(**0 救回也重训**防死锁);[3/3] dyn retrain(仅 SCOUT 臂)= all_accum 100ep,超过 DYN_FREEZE_AFTER 跳过、rollout 回溯最近 dyn;链式输入 exp{n}←exp{n−1} 缺则回退 base。

**判活速查**:round.log 只在段首尾写行,重训静默窗正常;train 真进度=`grep -o "Training epoch [0-9]*" | tail -1`+mtime+cputime(tqdm \r 渲染会让 tail 误读早期 epoch);heartbeat 的 `x/y` **不是**失败集大小;tmux grep 返回空可能是瞬态假阴性,二次核实再定性;链越跑越慢(it/epoch 逐轮涨)= success_accum 变大的协议固有性质,非卡死;dyn 段窗口 DP train.log 不滚动属正常。

**事故红线**:🔴绝不私杀任何训练/rollout 进程(须用户明令;进程已死如实报告不算);🔴explore 已合并而死在 retrain 的轮,**用现存 success_accum 手工从 [2/3] resume,禁原生重启整轮重跑**(round 守卫 else 分支会 rm 单轮 all/success+log=数据永久丢失);卷满 `errno=28 No space left`(此 CPFS 上 df 不可信,用 dd 写探针;处置权在用户);`Stale file handle` rc=2 = mv 换脚本的 ESTALE 读者(当轮产物通常完整,恢复=补真实 TOTAL 行+原串重启 tmux);慢≠卡,先看 it/epoch×it/s 数值签名;torch-shm rc=1 可发生在 300ep 完整跑完后的收尾——先验 ckpt 完整性(zip CRC/torch.load)再决定是否吃 workers=0 兜底重跑。

## Step 7 — 完赛收账

1. **终表从 json 权威汇总**:`rollout/<ARM>-expN/log/*.json`(SR=rollout json 的 success_rate;p@K=explore json 的 `pass_at_5` 键——键名历史沿用,实为 pass@{ETRIES},以 config 的 explore_try_times 辨);r6=eval-pk 原产双读数免补测;读数规则=6 轮终值+救回配对总量,单轮 SR 转化 SNR<1。
2. 汇总图:听用户令;风格惯例=mute 学术配色、曲线端点数值标注、SR 与 pass@K 分图(参照 `experiments/2026_9_4_figs_summary`、`2026_9_11_th95_3seed_figs_summary`)。
3. 持久 memory:写 campaign 文件(任务定义/实现决定/脚本位置/监控 id/终表)+ MEMORY.md 索引行;监控 cron 退役;wandb 操作**按 id 不按 display_name**(删除永久不可逆)。
4. 汇报用户:参数差异落地、review 结论与 P2 清单、终表、ETA/完赛摘要。git commit 是否做听用户(改动先过审红线)。

## 红线速查(全文见 AGENTS.md)

代码参数未经用户确认不落实(本 skill 的 Step 1 就是履行它)|禁原地覆写在跑文件(mv 换脚本防 ESTALE)|禁私杀训练/rollout|retrain 死→从 [2/3] resume 禁整轮重跑|GPU7 禁用|服务器只动 baojiachun|ssh pkill 模式加字符类防自匹配|同一只读调用 >2 次=病态循环|自造词首现解释并入 NAMES.md(NAMES.md 不在当前分支则汇报说明)|数值规格歧义先复述再启动。
