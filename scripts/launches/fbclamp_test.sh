#!/bin/bash
# fb soft-clamp validation arms (user order 2026-09-02): 20 scenes
# (seed42-61) x10 pass@10, wandb project ORBIT-9-2-fbclamp-test (test-only).
# Fix under test = option C: Newton residual soft-clamped to delta (tanh)
# + tangential noise banded to [kappa-delta, kappa+delta], ON TOP of the
# round-schedule combo (sigma_eff=0.16*0.5^(round-1), eta-dimless 0.33,
# noise-anneal p=2). Judgement vs atypical at the same trios:
#   soft_sqR5  >= rescue 17 (aty 19-2), jerk <= 0.8
#   soft_canR2 : 0 -> >=5, jerk 1.84 -> <0.8
#   soft_canR1 : regression guard, >= 15 (sched arm was 17)
#   aty_canR2  : fills the missing 20-scene aty baseline at exp1 trio
# Usage: GPU=<g> bash /tmp/fbclamp_test.sh <soft_sqR5|soft_canR2|soft_canR1|aty_canR2>
set -uo pipefail
export TMPDIR=/tmp
ARM=${1:?arm}
cd /root/workspace/baojiachun/scout-hparam
PY=/root/workspace/baojiachun/.venv/bin/python
SQ=/root/workspace/baojiachun/scout-rand/data/2026_9_1_orbchain/ORBIT-s233/square
CN=/root/workspace/baojiachun/scout-rand/data/2026_9_2_orbchain/ORBIT-s233/can
WPROJ=ORBIT-9-2-fbclamp-test

case "$ARM" in
  soft_sqR5) TASK=square; DP=$SQ/train/DP/DP-SCOUT-exp4/checkpoints/299.ckpt;
             VIB=$SQ/train/dyn/dyn-SCOUT-exp4/20260902-075528/scout_vib.ckpt; ROUND=5 ;;
  soft_canR1) TASK=can; DP=$CN/train/DP/DP-base/checkpoints/599.ckpt;
             VIB=$CN/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt; ROUND=1 ;;
  soft_canR2|aty_canR2) TASK=can; DP=$CN/train/DP/DP-SCOUT-exp1/checkpoints/299.ckpt;
             VIB=$CN/train/dyn/dyn-SCOUT-exp1/20260902-125340/scout_vib.ckpt; ROUND=2 ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
case "$TASK" in
  square) CORE=$SQ/rollout/square_core.hdf5 ;;
  can)    CORE=$CN/rollout/can_core.hdf5 ;;
esac

OUT=data/fbclamp_test/$ARM
mkdir -p "$OUT"
if [ ! -f "$OUT/failed_set.json" ]; then
  printf '{"failed_init_indices": [%s], "n_eval": 100, "base_seed": 42}\n' \
    "$(seq -s, 0 19)" > "$OUT/failed_set.json"
fi

GUIDE_ARGS=()
case "$ARM" in
  soft_*) GUIDE_ARGS=(--guide orbit --atypical-cap 2.5 --orbit-lam 0.5
                      --orbit-delta 0.25 --orbit-sigma 0.16
                      --orbit-round "$ROUND" --orbit-sigma-decay 0.5
                      --orbit-noise-anneal 2.0 --orbit-fb-clamp soft
                      --orbit-eta-dimless --guidance-scale 0.33) ;;
  aty_*)  GUIDE_ARGS=(--guide atypical --atypical-cap 2.5) ;;
esac

set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -m scout.eval.run_rollout \
  --config configs/eval_${TASK}_entropy.yaml --task $TASK --exp-num 0 \
  --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
  "${GUIDE_ARGS[@]}" \
  --vib-ckpt "$VIB" \
  --seed 42 --eval-seed 42 --explore-mode rescue \
  --failed-set-json "$OUT/failed_set.json" --save-failed-set "$OUT/failed_set.json" \
  --try-times 10 --explore-try-times 10 --n-envs 50 \
  --wandb-project "$WPROJ" --wandb-name "$ARM" \
  --output-dir "$OUT" >> "data/fbclamp_test/$ARM.log" 2>&1
echo "[$(date '+%m-%d %H:%M:%S')] $ARM rc=$? (log data/fbclamp_test/$ARM.log)"
