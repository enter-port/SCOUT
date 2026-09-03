#!/bin/bash
# orbit-hparam-dev dimless validation (user order 2026-09-02): SAME fixed
# dimensionless hyperparams on square AND can, 20 scenes (seed42-61) x10
# pass@10, compared against the legacy per-task-calibrated runs.
#   eta_tilde = 0.33  (target: square legacy mean_inject ~1.05)
#   sigma     = 0.16  (geometric mean of the two task sweet spots 0.25/0.10)
#   kappa=2.5 delta=0.25 lam=0.5 (shared by both tasks already)
# Usage: GPU=<g> bash /tmp/hparam_dimless_test.sh <square|can> <tag>
set -uo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-hparam
PY=/root/workspace/baojiachun/.venv/bin/python
TASK=${1:?square|can}
TAG=${2:?tag}
GPU=${GPU:?set GPU}
ETA_TILDE=${ETA_TILDE:-0.33}
SIGMA=${SIGMA:-0.16}

case "$TASK" in
  square)
    T=/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square ;;
  can)
    T=/root/workspace/baojiachun/scout-rand/data/2026_9_2_orbchain/ORBIT-s233/can ;;
  *) echo "task must be square|can"; exit 1 ;;
esac

OUT=data/hparam_test/$TAG
mkdir -p "$OUT"
if [ ! -f "$OUT/failed_set.json" ]; then
  printf '{"failed_init_indices": [%s], "n_eval": 100, "base_seed": 42}\n' \
    "$(seq -s, 0 19)" > "$OUT/failed_set.json"
fi

env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -m scout.eval.run_rollout \
  --config configs/eval_${TASK}_entropy.yaml --task $TASK --exp-num 0 \
  --base-dp-ckpt "$T/train/DP/DP-base/checkpoints/599.ckpt" \
  --core-hdf5 "$T/rollout/${TASK}_core.hdf5" \
  --guide orbit --atypical-cap 2.5 \
  --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma "$SIGMA" \
  --orbit-eta-dimless --guidance-scale "$ETA_TILDE" \
  --vib-ckpt "$(ls -d $T/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)" \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue \
  --failed-set-json "$OUT/failed_set.json" --save-failed-set "$OUT/failed_set.json" \
  --try-times 10 --explore-try-times 10 --n-envs 50 --no-wandb \
  --output-dir "$OUT" >> "data/hparam_test/$TAG.log" 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] $TAG rc=$? eta_tilde=$ETA_TILDE sigma=$SIGMA (log data/hparam_test/$TAG.log)"
