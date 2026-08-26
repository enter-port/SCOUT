#!/bin/bash
# sq_calib_probe.sh -- square entropy-cost dose-response probe (user 2026-08-26:
# <=5min/run, finish within 1h). Mini rescue run (10 scenes, try 2) on the OLD
# square base DP (e1 580) + newest old square VIB (e3 dyn-SCOUT-exp4).
# usage: sq_calib_probe.sh <tag> <gpu> <config> <off|atypical>
set -u
TAG=${1:?}; GPU=${2:?}; CFG=${3:?}; MODE=${4:?}
cd /root/workspace/baojiachun/scout-entropy
PY=/root/workspace/baojiachun/.venv/bin/python
DP=/root/workspace/baojiachun/scout/data/2026_8_14/experiment1/square/train/DP/DP-base/checkpoints/580.ckpt
VIB=/root/workspace/baojiachun/scout/data/2026_8_14/experiment3/square/train/dyn/dyn-SCOUT-exp4/20260819-201418/scout_vib.ckpt
CORE=/root/workspace/baojiachun/scout/data/robomimic/square/ph/image_v141_abs_core20.hdf5
OUT=data/square_calib/$TAG
mkdir -p "$OUT"
GARGS=(--guide off)
[ "$MODE" = atypical ] && GARGS=(--guide atypical --atypical-cap 2.5 --vib-ckpt "$VIB")
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout --config "$CFG" --task square --exp-num 1 \
  --base-dp-ckpt "$DP" --core-hdf5 "$CORE" "${GARGS[@]}" \
  --explore-mode rescue --explore-try-times 2 \
  --n-init-states 10 --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
  --output-dir "$OUT" > "$OUT/stdout.log" 2>&1
grep -aE "eval: success|rescued" "$OUT/stdout.log" | tail -2
