#!/bin/bash
# aty_calib_probe.sh -- SHORT atypical dose-calibration point on tool_hang
# (user order 2026-09-03: calibrate now, keep probes short). One point =
# first 10 failed inits of the r1 frozen set x 2 tries = 20 rollouts,
# n_envs=10, ~25 min. Reads out mean_inject (THE calibration signal) +
# rescued count (secondary, small-n). Separate output dir; read-only reuse
# of the r1 trio + failed.json; never touches the running campaign.
# usage: SCALE=3 GPU=6 bash soe_scripts/aty_calib_probe.sh
set -uo pipefail
SCALE=${SCALE:?set SCALE=<guidance_scale>}
GPU=${GPU:?set GPU=<cuda id>}
TRIES=${TRIES:-2}
cd /root/workspace/baojiachun/scout-orbit || exit 1
PY=/root/workspace/baojiachun/.venv/bin/python
TH=data/2026_9_1_toolhang/TOOLHANG-s233/tool_hang
DP=$(ls -t $TH/train/DP/DP-base/checkpoints/*.ckpt | head -1)
VIB=$(ls -t $TH/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$TH/rollout/tool_hang_core.hdf5
FAILED=$TH/rollout/SCOUT-exp1/failed.json
T=data/2026_9_3_atyprobe/calib_s${SCALE}_t${TRIES}
mkdir -p $T/log
[ -f "$FAILED" ] || { echo "[calib] FATAL: no failed.json"; exit 1; }
[ -f "$CORE" ] && [ -n "$DP" ] && [ -n "$VIB" ] || { echo "[calib] FATAL: trio missing"; exit 1; }

export MUJOCO_GL=egl SCOUT_RENDER_GPU=$GPU TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

# per-scale config copy (guidance_scale has NO CLI flag; dose enters via config)
$PY - "$SCALE" "$T" <<'PYEOF'
import sys, yaml
scale, T = sys.argv[1], sys.argv[2]
with open("configs/eval_tool_hang_entropy.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["exploration"]["guidance_scale"] = float(scale)
out = f"{T}/calib_cfg_s{scale}.yaml"
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[calib-cfg] exploration.guidance_scale={cfg['exploration']['guidance_scale']} -> {out}")
PYEOF

$PY - "$FAILED" "$T/failed10.json" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
idx = spec["failed_init_indices"][:10]
spec["failed_init_indices"] = idx
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print(f"[calib] failed set cut to {len(idx)} inits")
PYEOF
echo "[calib] tries=$TRIES (x$TRIES per init)"

echo "[calib] scale=$SCALE tries=$TRIES gpu=$GPU (20 rollouts)"
T0=$(date +%s)
env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config "$T/calib_cfg_s${SCALE}.yaml" --task tool_hang --exp-num 1 \
  --base-dp-ckpt "$DP" --vib-ckpt "$VIB" --core-hdf5 "$CORE" \
  --guide atypical --atypical-cap 2.5 \
  --explore-mode rescue --explore-try-times "$TRIES" --failed-set-json "$T/failed10.json" \
  --n-envs 10 --seed 42 --eval-seed 42 \
  --wandb-name "aty-calib-s${SCALE}-t${TRIES}" \
  --output-dir "$T" --output-success "$T/success.hdf5" --output-all "$T/all.hdf5" \
  --output-json "$T/log/aty_calib_explore.json" \
  > $T/rollout.stdout 2>&1
RC=$?
T1=$(date +%s)
echo "[calib] rc=$RC wall=$(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"
echo "[calib] mean_inject (last 2 telemetry lines) = THE readout:"
grep -a "guidance-telemetry" $T/rollout.stdout | tail -2
echo "[calib] rescued + jerk:"
$PY -c "
import json
d = json.load(open('$T/log/aty_calib_explore.json'))
print({k: d.get(k) for k in ['exploration_rescued','n_failed','collected_trajs','avg_jerk','pass_at_5']})"
echo "[calib] DONE $(date '+%F %T')"
