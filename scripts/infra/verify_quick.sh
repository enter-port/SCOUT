#!/bin/bash
# verify_quick.sh -- <=5-min guided mini-rollout probe for the render-corruption
# hunt (user rule: every experiment <= 5 minutes). Same ckpts/protocol as the
# corrupt r6-attempt1, just short: 5 eval + 10 explore scenes @ n_envs=25.
# Usage: verify_quick.sh <gpu> <tag>            (one experiment ~2-3 min)
set -u
GPU=${1:?gpu} TAG=${2:?tag}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
D=$REPO/data/2026_8_21/CAN-exp1-2333/can
OUT=$REPO/data/2026_8_21/_rootcause/quick_$TAG
mkdir -p "$OUT"
cd "$REPO" || exit 1
echo "START $(date '+%F %T') GPU$GPU $TAG"
timeout 280 env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_can_exp1.yaml --task can --exp-num 0 \
  --base-dp-ckpt "$D/train/DP/DP-SCOUT-exp5/checkpoints/99.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide dyn --vib-ckpt "$D/train/dyn/dyn-SCOUT-exp3/20260822-055010/scout_vib.ckpt" \
  --seed 42 --eval-seed 42 --explore-seed 6042 \
  --n-explore 10 --explore-try-times 1 \
  --n-init-states 5 --n-envs 25 \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" \
  --output-all "$OUT/all.hdf5" --no-wandb \
  > "$OUT/rollout.stdout" 2>&1
RC=$?
echo "END rc=$RC $(date '+%F %T')"
if [ $RC -eq 0 ] && [ -f "$OUT/all.hdf5" ]; then
  "$PY" soe_scripts/vis_validate.py "$OUT/all.hdf5" 20 > "$OUT/validate.log" 2>&1
  tail -1 "$OUT/validate.log"
else
  echo "rollout did not complete (rc=$RC)"
fi
