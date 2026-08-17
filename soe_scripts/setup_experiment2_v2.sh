#!/bin/bash
# experiment2 v2 setup (user 2026-08-17): SPLIT eval/explore protocol,
# free_bits 0.005, wandb one-run-per-round. Mirrors experiment1's layout
# under data/experiment2 with hardlinked cores + frozen DP-base (zero-copy).
#   configs/vib_{can,square}_image_e2.yaml  -> dyn-base (SCOUT-round0)
#   configs/eval_{can,square}_e2.yaml       -> split protocol rollout
# Driver: soe_scripts/round_e2.sh
set -eu
cd /root/workspace/baojiachun/scout

E1=data/experiment1
E2=data/experiment2

mkdir -p "$E2/logs" \
         "$E2/can/rollout" "$E2/can/train/DP" "$E2/can/train/dyn" \
         "$E2/square/rollout" "$E2/square/train/DP" "$E2/square/train/dyn"

# cores + frozen DP-base: hardlinks (same inode; experiment1 keeps its link so
# the still-running square chains' DP-base prereq check stays satisfied)
ln -f "$E1/can/rollout/can_core.hdf5"       "$E2/can/rollout/can_core.hdf5"
ln -f "$E1/square/rollout/square_core.hdf5" "$E2/square/rollout/square_core.hdf5"
[ -d "$E2/can/train/DP/DP-base" ]    || cp -al "$E1/can/train/DP/DP-base"    "$E2/can/train/DP/DP-base"
[ -d "$E2/square/train/DP/DP-base" ] || cp -al "$E1/square/train/DP/DP-base" "$E2/square/train/DP/DP-base"

# configs (yaml round-trip; comments from the originals are not carried over)
python - <<'PYEOF'
import yaml

def fix_paths(o):
    if isinstance(o, dict):
        return {k: fix_paths(v) for k, v in o.items()}
    if isinstance(o, list):
        return [fix_paths(x) for x in o]
    if isinstance(o, str):
        return o.replace("data/experiment1", "data/experiment2")
    return o

for t, up in [("can", "CAN"), ("square", "SQUARE")]:
    with open(f"configs/vib_{t}_image.yaml") as f:
        cfg = yaml.safe_load(f)
    cfg = fix_paths(cfg)
    cfg["beta"] = 1.0e-4          # explicit: same as experiment1
    cfg["free_bits"] = 0.005      # experiment2 (user 2026-08-17): floor 16*0.005 nats
    cfg["save_dir"] = f"/root/workspace/baojiachun/scout/data/experiment2/{t}/train/dyn/dyn-base"
    w = cfg.setdefault("wandb", {})
    w["project"] = f"{up}-experiment2"
    w["name"] = "SCOUT-round0"
    w["metric_prefix"] = "dyn/"
    with open(f"configs/vib_{t}_image_e2.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    with open(f"configs/eval_{t}.yaml") as f:
        ecfg = yaml.safe_load(f)
    ecfg = fix_paths(ecfg)
    # split protocol (user 2026-08-17): eval scenes FIXED every round (seed 42
    # -> 42..141, 100); explore scenes FRESH every round (round i uses base
    # seed i*1000+42 -> +0..+499, 500, try_times 1). round_e2.sh passes the
    # per-round --explore-seed explicitly; these are the round-1 defaults.
    ecfg["explore"] = {"base_seed_round1": 1042, "n_scenes": 500, "try_times": 1}
    with open(f"configs/eval_{t}_e2.yaml", "w") as f:
        yaml.safe_dump(ecfg, f, sort_keys=False)
print("configs written")
PYEOF

cat > "$E2/experiment.md" <<'MDEOF'
# experiment2 v2 — SPLIT eval/explore 协议(fb=0.005,β=1e-4)

启动:2026-08-17 晚(用户指令;推翻并替换同日早些的 β=1e-3 试验版,该版输出与 wandb 2-* 记录已全部清除)。

## 协议(与 experiment1 的差异)
- **eval 与 explore 分离**:
  - eval:每轮**固定**场景 seed 42 → 42..141(100 个),单次,纯测量,不产数据;
  - explore:每轮**新**场景,第 i 轮用 base seed i*1000+42 → +0..+499(500 个,try_times=1);
  - 数据:explore 成功 → success.hdf5(训 DP);explore 全部轨迹 → all.hdf5(训 dyn);均跨轮累积(success_accum / all_accum)。
- **dyn**:free_bits 0.05→**0.005**,β=1e-4(不变);dyn-base 记为 wandb `SCOUT-round0`。
- **wandb**:每 task 一个 project(**CAN-experiment2** / **SQUARE-experiment2**),一个 round 一个 run:`{SCOUT,DP}-round{i}`(rollout 建 run,DP/dyn retrain 通过 WANDB_RUN_ID 续写同一 run)。分节指标:`eval/success_rate`(终值)、`eval/env_done`(进度)、`explore/env_done`(进度)、`explore/success_count` / `explore/total`(终值)、`explore/avg_jerk`、`DP/train_loss|lr|epoch`、`dyn/latent_mse|kl|lr|epoch`(实时,无 μ)。
- **DP retrain 自适应 epoch**:clamp(12000/n_demos, 100, 600)——500 场景的 explore 产量约为旧 core 的 15-30x,固定 600ep 到 round6 会到 ~90x 数据量;该公式保持梯度步数预算接近 "600ep×20demos" 的既有约定。

## 设置(其余同 experiment1)
- core / DP-base:硬链自 experiment1(can/square 580.ckpt;experiment1 保留链接以维持 square 链的 prereq 检查)。
- 驱动 `soe_scripts/round_e2.sh <task> <DP|SCOUT> <round>`;配置 `configs/{vib,eval}_{can,square}_*_e2.yaml`。
- GPU:can-SCOUT=4 / can-DP=5 / square-SCOUT=6 / square-DP=7;会话 `round2_<task>_<A>_<num>`;console log `data/experiment2/logs/`。
- round1 → round6 自动链式迭代。

## 观察点
- KL 是否守住 0.005×16=0.08 nats 的地板(对照 fb=0.05 时的 ~1.0);
- explore 500 场景的 success_count 逐轮变化(SCOUT vs DP);
- eval/success_rate 六轮曲线 DP vs SCOUT。
MDEOF

echo "=== verify ==="
find "$E2" -maxdepth 2 | sort
grep -H 'beta:\|free_bits:' configs/vib_can_image_e2.yaml configs/vib_square_image_e2.yaml
grep -HA2 'wandb:' configs/vib_can_image_e2.yaml | head -6
grep -HA4 'explore:' configs/eval_can_e2.yaml
