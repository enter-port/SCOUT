#!/bin/bash
# verify_2333.sh -- rerun the 2333-DP round-1 explore setting on a FREE gpu to
# check the 2/100 anomaly (user 2026-08-22). Usage: verify_2333.sh <n_envs> <gpu>
set -u
NE=${1:?n_envs} GPU=${2:?gpu}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
D=$REPO/data/2026_8_21/CAN-exp1-2333/can
OUT=$REPO/data/2026_8_21/_verify_2333/env$NE
mkdir -p "$OUT"
cd "$REPO" || exit 1
echo "START $(date '+%F %T') n_envs=$NE GPU$GPU"
env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m scout.eval.run_rollout \
  --config configs/eval_can_exp1.yaml --task can --exp-num 0 \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide off --seed 42 --eval-seed 42 --explore-seed 1042 \
  --n-explore 100 --explore-try-times 1 \
  --n-init-states 100 --n-envs "$NE" \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" \
  --output-all "$OUT/all.hdf5" --no-wandb \
  > "$OUT/rollout.stdout" 2>&1
RC=$?
echo "END rc=$RC $(date '+%F %T')"
"$PY" soe_scripts/vis_validate.py "$OUT/all.hdf5" 20 > "$OUT/validate.log" 2>&1 \
  && echo "validation OK" || echo "validation CORRUPT"
