#!/bin/bash
# verify_gpu2.sh -- replicate the 2333-SCOUT r6-attempt1 corruption condition
# EXACTLY (same ckpts, same seeds, guided, n_envs=25) on a given GPU, with
# forensic preservation (validate.log + per-demo bad list kept per attempt).
# Usage: verify_gpu2.sh <gpu> <tag>
set -u
GPU=${1:?gpu} TAG=${2:?tag}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
D=$REPO/data/2026_8_21/CAN-exp1-2333/can
OUT=$REPO/data/2026_8_21/_rootcause/$TAG
mkdir -p "$OUT"
cd "$REPO" || exit 1
echo "START $(date '+%F %T') GPU$GPU tag=$TAG"
env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_can_exp1.yaml --task can --exp-num 0 \
  --base-dp-ckpt "$D/train/DP/DP-SCOUT-exp5/checkpoints/99.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide dyn --vib-ckpt "$D/train/dyn/dyn-SCOUT-exp3/20260822-055010/scout_vib.ckpt" \
  --seed 42 --eval-seed 42 --explore-seed 6042 \
  --n-explore 100 --explore-try-times 1 \
  --n-init-states 100 --n-envs 25 \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" \
  --output-all "$OUT/all.hdf5" --no-wandb \
  > "$OUT/rollout.stdout" 2>&1
RC=$?
echo "END rc=$RC $(date '+%F %T')"
"$PY" soe_scripts/vis_validate.py "$OUT/all.hdf5" 100 > "$OUT/validate.log" 2>&1
tail -1 "$OUT/validate.log"
grep -c 'noise\|frozen' "$OUT/validate.log" || true
