#!/bin/bash
# expert_guide_exp.sh -- expert z-bank guided vs unguided rollout (user 2026-08-21).
#
# ckpts: exp1 ROUND-3 trio (DP-SCOUT-exp3 + dyn-SCOUT-exp3, can). Scenes: the
# seed-42 fixed set 42..141 (100 scenes) for BOTH arms in ONE run (split mode):
#   eval phase    = UNGUIDED base DP over seeds 42..141  (--eval-seed 42)
#   explore phase = EXPERT-GUIDED over the same seeds     (--explore-seed 42)
# Guidance scale 0.01 (post-1/B-fix calibration, eval_can_e2fix01.yaml). The
# expert z-bank is built at planner-attach time from the core hdf5.
# Render-corruption guard: vis_validate.py gate; retry SAME n_envs once.
#
# Usage:  GPU=<id> bash soe_scripts/expert_guide_exp.sh
set -u
GPU=${GPU:?set GPU=<cuda id>}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
cd "$REPO" || exit 1

E2=$REPO/data/experiment2/can
DPDIR=$E2/train/DP/DP-SCOUT-exp3
VIBDIR=$E2/train/dyn/dyn-SCOUT-exp3
CORE=$E2/rollout/can_core.hdf5
CFG=configs/eval_can_e2fix01.yaml
[ -f "$CFG" ] || { echo "FATAL: missing $CFG"; exit 1; }
DP=$(ls -t "$DPDIR"/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -n "$DP" ] || { echo "FATAL: no ckpt under $DPDIR"; exit 1; }
VIB=$(ls -t "$VIBDIR"/*/scout_vib.ckpt 2>/dev/null | head -1)
[ -n "$VIB" ] || { echo "FATAL: no scout_vib.ckpt under $VIBDIR"; exit 1; }
[ -f "$CORE" ] || { echo "FATAL: missing core $CORE"; exit 1; }
DIR=$REPO/data/vis_final/expert_guide_e2-SCOUT05-round3
mkdir -p "$DIR"
echo "DP=$DP"
echo "VIB=$VIB"

OK=0
for TRY in 1 2; do
  rm -f "$DIR/all.hdf5" "$DIR/success.hdf5"; rm -rf "$DIR/log"
  echo "[expert] try$TRY n_envs=20 GPU$GPU $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m scout.eval.run_rollout \
    --config "$CFG" --task can \
    --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
    --guide expert --vib-ckpt "$VIB" \
    --eval-seed 42 --explore-seed 42 \
    --n-explore 100 --explore-try-times 1 \
    --n-init-states 100 --n-envs 20 \
    --exp-num 0 \
    --output-dir "$DIR" --output-success "$DIR/success.hdf5" \
    --output-all "$DIR/all.hdf5" --no-wandb \
    > "$DIR/rollout.stdout" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[expert] try$TRY rc=$RC -- tail:"
    tail -n 10 "$DIR/rollout.stdout"
    continue
  fi
  if "$PY" soe_scripts/vis_validate.py "$DIR/all.hdf5" 20 > "$DIR/validate.log" 2>&1; then
    echo "[expert] try$TRY OK (validation passed) $(date '+%F %T')"
    OK=1; break
  fi
  echo "[expert] try$TRY RENDER CORRUPT -- validator says:"
  grep CORRUPT "$DIR/validate.log" | head -3
done
[ $OK -eq 1 ] || { echo "[expert] FAILED"; exit 1; }
echo "[expert_guide_exp] done $(date '+%F %T')"
