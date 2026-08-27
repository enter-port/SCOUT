# rand RUNBOOK — entropy-random-dev 长期 campaign(每个测试 subagent 必读必守)

**目标**:把随机性加进 guidance cost。**停止条件:纯 rescue(base DP+dyn、
seed42 100 场景、无数据回灌)pass@10 > 0.85**。现行最好 = 方案三 0.76-0.78
(rescued 14/19)。baseline 20 场景筛查分在 data/rand/base3_screen/。

## 红线(违反 = 事故)
1. 服务器改动仅限 `/root/workspace/baojiachun/`;绝不删别的东西;不碰别人进程。
2. **绝不 kill 任何进程**(自己的 tmux 也不行——跑错了就让它跑完,报告即可)。
3. 用 GPU 前查占用:`ssh -o BatchMode=yes -p <port> root@106.14.2.243 'nvidia-smi --query-gpu=index,memory.used --format=csv,noheader'`。
   - **优先级(用户令 2026-08-27):1022 优先,不够再用 1024**。
   - port **1022**:GPU 0/1/3/4 = square 链(收尾后释放,释放后可用),GPU 2/5 可用,**GPU7 禁用**(ECC 硬件缺陷)。
   - port **1024**(仅当 1022 无空卡):8 张可用(独立容器,同 CPFS;**有他人 tmux server/stu2v2a,绝不碰**)。
4. 一切写操作走新文件;不改共享代码文件(见下);部署自己的 idea 文件用
   `tr -d '\r' < local | ssh ... 'cat > /root/workspace/baojiachun/scout-rand/scout/guidance/rand_costs/<name>.py'`
   然后 `python -m py_compile` 验证。
5. wandb 一律不用(脚本已 --no-wandb);指标读 `data/rand/<tag>/log/*.json` 的
   `baseline_solved / exploration_rescued / pass_at_5`(字段名硬编码,实际是 pass@10)。

## 冲突管理契约
- 你的 idea = `scout/guidance/rand_costs/<name>.py` 一个文件(定义 NAME +
  make_planner,见 dose.py 模板与 __init__.py 头注释);**绝不编辑**共享文件
  (entropy_costs.py / rollout_vec.py / rollout_pipeline.py / run_rollout.py /
  rand_screen.sh / rand_full.sh)。
- 钩子:select_z(每 chunk 捕获意图基线)、set_row_jobs(每 replan 给
  (state, init_idx, try_idx))、compute_loss(x̂₀, obs, reduction="sum" 必须支持)。
- 随机性必须由 (rand_seed, init_idx, try_idx) 确定性播种。
- 私有超参走 `--rand-kwargs "k1=v1,k2=v2"`(自动进 entropy_kwargs,免改共享代码)。

## 测试协议(用户令 2026-08-27)
1. **20 场景筛参**(scenes 42-61):`bash soe_scripts/rand_screen.sh <tag> rand_<name> <gpu> --rand-kwargs "..."`,
   SCALE 环境变量换 η:`SCALE=1.0 bash ...`。
2. 筛到最优 → **100 场景 pass@10**:`bash soe_scripts/rand_full.sh <tag>-full rand_<name> <gpu> ...`。
3. 判读:rescued 场景数为主指标(基线 19);pass@10 与 0.76-0.78 带 比;
   20 场景的失败集 ≈ 8 个,rescued 差 ±2 内算平。
4. tmux 启动:`ssh -p <port> ... 'tmux new-session -d -s rand_<tag> "bash /root/workspace/baojiachun/scout-rand/soe_scripts/rand_screen.sh <tag> rand_<name> <gpu>"'`;
   stdout 块缓冲——进度看 GPU util / telemetry 行,结果看 json。
5. 完成后:把结果(tag、json 数字、结论)写进 `data/rand/RESULTS.md`(追加,
   每条 3 行以内),并在报告里给出 rescued 对基线是子集还是新增场景
   (方法:指纹比对,见 dispersion_shell.py 的 demo_fps 思路)。

## 已判死刑的方向(勿重试)
- 均匀随机 16 维方向目标(方案A):严格子集、PR 只 1.56、剂量↑PR↓。
  诊断:均匀方向经低秩 Jacobian 投影被收回窄锥。
- κ 扫过 1.25-5:2.5 最优,5 更差。η(全局)扫过 26× 范围。
