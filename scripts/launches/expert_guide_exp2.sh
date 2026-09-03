#!/bin/bash
# expert_guide_exp2.sh -- expert z-bank from the ROUND-3 RETRAIN DATA (user
# 2026-08-21): bank = rollout/SCOUT-exp3/success_accum.hdf5 (core + rounds
# 1-3 exploration successes -- exactly what trained DP-SCOUT-exp3), instead
# of core-20 only (expert_guide_exp.sh). Everything else identical: exp1
# round-3 trio, seeds 42..141 for both arms, scale 0.01, n_envs 20.
# Outputs -> data/vis_final/expert_guide_e2-SCOUT05-round3_bankaccum/.
#
# Usage:  GPU=<id> bash soe_scripts/expert_guide_exp2.sh
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
BANK=$E2/rollout/SCOUT-exp3/success_accum.hdf5
CFG=configs/eval_can_e2fix01.yaml
[ -f "$CFG" ] || { echo "FATAL: missing $CFG"; exit 1; }
DP=$(ls -t "$DPDIR"/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -n "$DP" ] || { echo "FATAL: no ckpt under $DPDIR"; exit 1; }
VIB=$(ls -t "$VIBDIR"/*/scout_vib.ckpt 2>/dev/null | head -1)
[ -n "$VIB" ] || { echo "FATAL: no scout_vib.ckpt under $VIBDIR"; exit 1; }
[ -f "$CORE" ] || { echo "FATAL: missing core $CORE"; exit 1; }
[ -f "$BANK" ] || { echo "FATAL: missing bank $BANK"; exit 1; }
DIR=$REPO/data/vis_final/expert_guide_e2-SCOUT05-round3_bankaccum
mkdir -p "$DIR"
echo "DP=$DP"
echo "VIB=$VIB"
echo "BANK=$BANK"

OK=0
for TRY in 1 2; do
  rm -f "$DIR/all.hdf5" "$DIR/success.hdf5"; rm -rf "$DIR/log"
  echo "[expert2] try$TRY n_envs=20 GPU$GPU $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m scout.eval.run_rollout \
    --config "$CFG" --task can \
    --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
    --guide expert --vib-ckpt "$VIB" --bank-hdf5 "$BANK" \
    --eval-seed 42 --explore-seed 42 \
    --n-explore 100 --explore-try-times 1 \
    --n-init-states 100 --n-envs 20 \
    --exp-num 0 \
    --output-dir "$DIR" --output-success "$DIR/success.hdf5" \
    --output-all "$DIR/all.hdf5" --no-wandb \
    > "$DIR/rollout.stdout" 2>&1
  RC=$?
  if [ $RC -ne 0 ]; then
    echo "[expert2] try$TRY rc=$RC -- tail:"
    tail -n 10 "$DIR/rollout.stdout"
    continue
  fi
  if "$PY" soe_scripts/vis_validate.py "$DIR/all.hdf5" 20 > "$DIR/validate.log" 2>&1; then
    echo "[expert2] try$TRY OK (validation passed) $(date '+%F %T')"
    OK=1; break
  fi
  echo "[expert2] try$TRY RENDER CORRUPT -- validator says:"
  grep CORRUPT "$DIR/validate.log" | head -3
done
[ $OK -eq 1 ] || { echo "[expert2] FAILED"; exit 1; }
echo "[expert_guide_exp2] done $(date '+%F %T')"
