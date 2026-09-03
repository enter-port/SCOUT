#!/bin/bash
# experiment3 (user 2026-08-18): can only, SAME params as experiment2's final
# recipe (beta 3e-5, fb 0.005, lambda 5, guidance_scale 0.5) with ONE change:
# explore runs on the SAME fixed scene set as eval (seed 42 -> 42..141) and
# 100 scenes instead of 500 fresh ones per round.
# No python changes: round_e3.sh just passes --explore-seed 42 --n-explore 100.
set -eu
cd /root/workspace/baojiachun/scout

E2=data/experiment2/can
E3=data/experiment3/can

mkdir -p "$E3/rollout" "$E3/train/DP" "$E3/train/dyn" data/experiment3/logs
ln -f "$E2/rollout/can_core.hdf5" "$E3/rollout/can_core.hdf5"
[ -d "$E3/train/DP/DP-base" ] || cp -al "$E2/train/DP/DP-base" "$E3/train/DP/DP-base"

python - <<'PYEOF'
import yaml

def fix(o):
    if isinstance(o, dict): return {k: fix(v) for k, v in o.items()}
    if isinstance(o, list): return [fix(x) for x in o]
    if isinstance(o, str): return o.replace("experiment2", "experiment3")
    return o

# eval: experiment3 paths; explore = SAME fixed scenes as eval (42..141, 100)
with open("configs/eval_can_e2.yaml") as f:
    ecfg = yaml.safe_load(f)
ecfg = fix(ecfg)
ecfg["explore"] = {"base_seed_round1": 42, "n_scenes": 100, "try_times": 1}
with open("configs/eval_can_e3.yaml", "w") as f:
    yaml.safe_dump(ecfg, f, sort_keys=False)

# vib: same beta/fb/lambda; experiment3 paths + CAN-experiment3 round0
with open("configs/vib_can_image_e2.yaml") as f:
    cfg = yaml.safe_load(f)
cfg = fix(cfg)
cfg["save_dir"] = "/root/workspace/baojiachun/scout/data/experiment3/can/train/dyn/dyn-base"
w = cfg.setdefault("wandb", {})
w["project"] = "CAN-experiment3"
w["name"] = "SCOUT-round0"
with open("configs/vib_can_image_e3.yaml", "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print("configs written")
PYEOF

# driver: experiment3 root, fixed explore seed = eval seed, 100 scenes
sed -e 's/experiment2/experiment3/g' \
    -e 's/^ESEED=$((NUM \* 1000 + 42)).*/ESEED=$SEED               # e3: explore uses the SAME fixed scene set as eval/' \
    -e 's/^NEXPLORE=500$/NEXPLORE=100/' \
    soe_scripts/round_e2.sh > soe_scripts/round_e3.sh
chmod +x soe_scripts/round_e3.sh

cat > data/experiment3/experiment.md <<'MDEOF'
# experiment3 — can 专用:explore 固定用 eval 同款场景

启动:2026-08-18(用户指令)。与 experiment2 最终配方**完全一致**(β=3e-5、fb=0.005、λ=5、guidance_scale=0.5、split 协议、自适应 epoch、DP-base 同一硬链),唯一差异:

- **explore 场景 = eval 场景**:固定 seed 42 → 42..141(100 个),每轮不变(不再用 i*1000+42 的 500 新场景)。

目的:对照"每轮换新场景"(experiment2)与"固定场景重试"(experiment3,接近 SOE 的 retry 语义但每场景 1 次、不带 pass@k 重试)。

- 驱动 `soe_scripts/round_e3.sh`(round_e2.sh 变体:ESEED=$SEED、NEXPLORE=100、DATA=experiment3、wandb CAN-experiment3)。
- GPU:can-SCOUT=GPU0 / can-DP=GPU1;会话 `round3_can_<A>_<num>`;dyn-base=SCOUT-round0。
- 其余规则(累积、自适应 epoch、自动链到 round6)同 experiment2。
MDEOF

echo "=== verify ==="
grep -E '^ESEED=|^NEXPLORE=' soe_scripts/round_e3.sh
grep -A3 'explore:' configs/eval_can_e3.yaml | head -4
grep -E '^beta:|^free_bits:' configs/vib_can_image_e3.yaml
grep -E 'guidance_scale' configs/eval_can_e3.yaml
