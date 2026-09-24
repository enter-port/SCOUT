---
name: launch-experiment
description: 启动、记录、监控并收账一条 SCOUT 实验链。用户提出开实验、启动 campaign、跑 task、做变体，或给出新的实验需求时使用；统一接入服务器实验登记系统。
---

# SCOUT 实验生命周期

这个 skill 适用于任何能访问 SCOUT 服务器的模型或代理。它负责从参数确认、实验登记、脚本 dry-run、启动、监控到结果收账的完整流程。先读 [references/registry.md](references/registry.md)，模型不得自造实验目录、编号或另一份账本。

## 记录规则

- 服务器 repo：`/root/workspace/baojiachun/scout`；数据基目录：`/root/workspace/baojiachun`。
- 实验记录目录：`experiments/<TASK>/full/<ID>/` 或 `experiments/<TASK>/auxiliary/<ID>/`；数据目录：`data/<TASK>/full/<ID>/` 或 `data/<TASK>/auxiliary/<ID>/`。
- `full` 是正式实验链，包括未完成链和只有历史证据的正式链；`auxiliary` 是探针、剂量/参数标定、冒烟、分片验证、可视化、基础训练等。
- 名称由系统分配：`TASK-YYYY-MM-DD`。只有同 TASK、同日期存在多条记录才使用 `-01`、`-02`；full 和 auxiliary 共享当日编号序列。不要手写编号，也不要按 seed、arm、round 增加编号。
- `uid` 不变；同日新增时原无后缀名称可能变为 `-01`，旧名保存到 `aliases`。
- `experiment.json` 是身份、配置、标签、状态和笔记的唯一可编辑来源；README、results 和总 registry 由 sync 生成，禁止手改。
- 旧数据路径保留兼容软链接；W&B 名、原始源路径、旧实验名不得删除。数据不存在但归档存在时只能登记归档证据，不能用旧表数字补结果。

## 1. 参数确认和现场核实

用户说“开一个实验”时，先判断 full/auxiliary。已有参数不重复询问，缺失参数不能猜，必须问清：

- TASK、实验说明、分类和日期；历史补登记要说明日期依据。
- seed 列表、实验臂、轮数、末轮评估口径。
- base 来源（已有记录/已有 checkpoint/重新训练）和具体路径。
- guidance 机制及 eta/kappa/sigma/lambda/delta、clamp、anneal 等剂量。
- pass@K、探索重试次数、worker/shard、每 shard 环境数、训练 batch、GPU、W&B project。
- 失败处理、预计状态和是否需要画图。

用一句话复述 `TASK / 分类 / seed / 轮数 / 臂 / guidance / 剂量 / pass@K / workers / batch / GPU / base`；有歧义就停下询问。`pass_at_5` 是历史键名，K 以 `explore_try_times` 为准。

先只读核实：

```bash
ssh -o BatchMode=yes -p 1022 root@106.14.2.243 \
  'cd /root/workspace/baojiachun/scout && tmux ls; nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; free -g'
```

核实 base、数据目录、GPU 和运行会话；GPU7 禁用，不抢占已有任务。现场读数写入笔记。

## 2. 先登记，再启动

创建训练目录、复制 checkpoint、启动 tmux 前，必须先登记：

```bash
cd /root/workspace/baojiachun/scout
python3 scripts/experiment_registry.py new \
  --task CAN --category full --date 2026-09-25 \
  --title 'Orbit v4 / ATY vs DP' --expected-rounds 6 \
  --config-json '{"seeds":[233,2333],"arms":["ATY","DP"],"guidance":"orbit-v3","eta_tilde":0.33,"kappa":2.5,"pass_k":5,"workers":4,"batch":256,"gpu":"0"}' \
  --tags 'orbit-v4,pass@5' --notes 'base: CAN-2026-09-02'
```

CLI 参数不足时用 `http://127.0.0.1:8765` 新建，再在详情页保存配置；不要直接 mkdir。同一任务同日的 ID 由系统统一分配。创建后执行 `show <ID-or-uid>` 和 `check`，保存返回的 ID、uid、数据路径和 aliases。

## 3. 写脚本、复制 base、dry-run

脚本一律新建，不能原地覆盖正在使用的文件。服务器唯一真本是 repo 内 `scripts/`；`soe_scripts/` 只读历史副本。

参考模板：`scripts/rounds/round_orbit.sh`、`round_aty.sh`、`round_orbit_sqb256.sh`、`scripts/toolhang/round_th95.sh` 及其 chain/launch wrapper。变体文件头写父模板、实验 ID、差异、base 来源和时间。

复制 base 三件套使用 `cp -L`，只能写本实验数据目录。确认 rollout、DP-base、dyn-base checkpoint/config 均在新路径。先做 dry-run，不要指向旧数据：

```bash
DRY_RUN=1 DATA_ROOT=/root/workspace/baojiachun/scout/data/<TASK>/<category>/<ID> \
  bash scripts/<new-round-script>.sh <task> <arm> 1 full
bash -n scripts/<new-script>.sh
grep -nE 'DATA_ROOT|WPROJ|ETRIES|SHARD_P|guidance|explore-try|batch|eval-pk' scripts/<new-script>.sh
```

逐项核对新 ID/W&B project、guidance、剂量、ETRIES、workers、`--n-envs`、checkpoint、batch 和末轮 eval-pk。ETRIES=1 只能称 pass@1。启动前做一次只读 review：变体 diff、base 完整性、链幂等、round0 门禁、资源隔离、引号和 set -u；P0/P1 先修，P2 写笔记。

## 4. 启动和同步

用户确认参数后才启动 launch wrapper。确认 round0 成功、tmux 出现、round1 写入 START；然后将状态设为 `running`，笔记写启动时间、session、脚本、GPU、base 和实际参数。启动失败保留源文件，状态设为 `failed`，不要删除记录重来。

每轮 TOTAL 后同步：

```bash
python3 scripts/experiment_registry.py sync --id <ID-or-uid>
python3 scripts/experiment_registry.py check
```

详情页能查看逐条指标、配置源文件、SHA-256 和修改历史。读取错误必须显式报告。

## 5. 监控、事故和收账

监控只读。除非用户明确下令，不得 kill、重启或修改训练进程。round.log 的静默重训不等于卡死，先看 epoch、mtime、CPU 时间。看到 TOTAL 前不标记轮完成；一条臂完成不代表整链完成。

retrain 中断从已有 `[2/3]` 或 `[3/3]` 产物恢复，不删除 success_accum/all_accum 后整轮重跑。Stale file handle、磁盘满、torch-shm、GPU OOM 等先保存原始错误和路径，再按用户批准的方式恢复。不得把 heartbeat x/y 当失败集大小，不得用旧表补 JSON。

完成后从服务器原始 rollout JSON 和 round.log 读权威结果，执行 `sync --id`、`check`，核对 SR、pass@K、seed、arm、round、失败臂和实际配置。将状态设为 `completed`、`failed` 或 `paused`，笔记写结论、未完成项和下一步。画图前先刷新服务器结果，再使用 `plot-campaign-figs` skill。

向用户汇报正式 ID、参数、脚本、结果源、状态、异常和未完成项。

## 红线

- 不登记就启动；不猜缺失参数。
- 不手工复用 ID、W&B project、tmux session 或数据目录。
- 不原地改在跑脚本，不私杀进程，不用旧表填缺失结果。
- 不把 auxiliary 冒充 full，不把历史文档标为已完成。
- 不修改 `/root/workspace/baojiachun` 之外的路径来伪造结果。
