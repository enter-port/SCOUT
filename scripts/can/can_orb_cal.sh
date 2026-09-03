#!/bin/bash
# CAN orbit sigma calibration (user order 2026-09-02): pass@10 of orbit
# rescue x10 on the FIRST 20 eval scenes (seed42-61), failed-set json =
# ALL of [0..19] so the eval phase is skipped entirely. No atypical/placebo
# arms; baseline reference = wandb CAN-8-24-entropy-s233 history.
# Trio = the same can s233 base assets the CAN-9-2-orbit-s233 chain uses.
# Usage: GPU=<g> SIGMA=<v> [LAM=0.5] [DELTA=0.25] bash /tmp/can_orb_cal.sh <tag>
#        DRYRUN=1 GPU=2 SIGMA=0.05 bash /tmp/can_orb_cal.sh s005   (argv check)
set -uo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
TAG=${1:?tag}
SIGMA=${SIGMA:?set SIGMA=<orbit sigma>}
LAM=${LAM:-0.5}
DELTA=${DELTA:-0.25}
GPU=${GPU:?set GPU=<cuda id>}
T=/root/workspace/baojiachun/scout-rand/data/2026_9_2_orbchain/ORBIT-s233/can
OUT=data/can_orb_cal/$TAG
mkdir -p "$OUT"
if [ ! -f "$OUT/failed_set.json" ]; then
  printf '{"failed_init_indices": [%s], "n_eval": 100, "base_seed": 42}\n' \
    "$(seq -s, 0 19)" > "$OUT/failed_set.json"
fi

if [ -n "${DRYRUN:-}" ]; then
  echo "DRYRUN argv: env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout --config configs/eval_can_entropy.yaml --task can --exp-num 0 --base-dp-ckpt $T/train/DP/DP-base/checkpoints/599.ckpt --core-hdf5 $T/rollout/can_core.hdf5 --guide orbit --atypical-cap 2.5 --orbit-lam $LAM --orbit-delta $DELTA --orbit-sigma $SIGMA --vib-ckpt $T/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt --seed 42 --eval-seed 42 --explore-mode rescue --failed-set-json $OUT/failed_set.json --save-failed-set $OUT/failed_set.json --try-times 10 --explore-try-times 10 --n-envs 50 --no-wandb --output-dir $OUT"
  echo "failed_set: $(cat "$OUT/failed_set.json")"
  exit 0
fi

env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -m scout.eval.run_rollout \
  --config configs/eval_can_entropy.yaml --task can --exp-num 0 \
  --base-dp-ckpt "$T/train/DP/DP-base/checkpoints/599.ckpt" \
  --core-hdf5 "$T/rollout/can_core.hdf5" \
  --guide orbit --atypical-cap 2.5 \
  --orbit-lam "$LAM" --orbit-delta "$DELTA" --orbit-sigma "$SIGMA" \
  --vib-ckpt "$T/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt" \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue \
  --failed-set-json "$OUT/failed_set.json" --save-failed-set "$OUT/failed_set.json" \
  --try-times 10 --explore-try-times 10 --n-envs 50 --no-wandb \
  --output-dir "$OUT" >> "data/can_orb_cal/$TAG.log" 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] $TAG rc=$? sigma=$SIGMA lam=$LAM delta=$DELTA (log data/can_orb_cal/$TAG.log)"
