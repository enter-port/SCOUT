#!/bin/bash
# verify_envspeed.sh -- 5-min-capped rollout speed probe: identical mini
# workload (5 eval + 10 explore scenes, guided), single variable = n_envs.
# Wall time of the completed run -> throughput ratio between env counts.
# Usage: verify_envspeed.sh <gpu> <n_envs> <tag>
set -u
GPU=${1:?gpu} NE=${2:?n_envs} TAG=${3:?tag}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
D=$REPO/data/2026_8_21/CAN-exp1-2333/can
OUT=$REPO/data/2026_8_21/_rootcause/envspeed_$TAG
mkdir -p "$OUT"
cd "$REPO" || exit 1
T0=$(date +%s)
echo "START $(date '+%F %T') GPU$GPU n_envs=$NE"
timeout 290 env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_can_exp1.yaml --task can --exp-num 0 \
  --base-dp-ckpt "$D/train/DP/DP-SCOUT-exp5/checkpoints/99.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide dyn --vib-ckpt "$D/train/dyn/dyn-SCOUT-exp3/20260822-055010/scout_vib.ckpt" \
  --seed 42 --eval-seed 42 --explore-seed 6042 \
  --n-explore 10 --explore-try-times 1 \
  --n-init-states 5 --n-envs "$NE" \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" \
  --output-all "$OUT/all.hdf5" --no-wandb \
  > "$OUT/rollout.stdout" 2>&1
RC=$?
T1=$(date +%s)
echo "END rc=$RC wall=$(( T1 - T0 ))s (n_envs=$NE)"
if [ $RC -eq 0 ] && [ -f "$OUT/all.hdf5" ]; then
  "$PY" soe_scripts/vis_validate.py "$OUT/all.hdf5" 20 > "$OUT/validate.log" 2>&1
  tail -1 "$OUT/validate.log"
else
  echo "TIMED OUT or failed (rc=$RC) -- treat 290s as lower bound"
fi
