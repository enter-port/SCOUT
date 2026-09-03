#!/bin/bash
# fix01_confirm.sh -- 1/B bug-fix confirmation runs (user 2026-08-21).
#
# Post-fix guidance_scale=0.01 (user: reproduces the historical real-round
# effective strength 0.5/B with B~50). Same protocol as the vis_final guided
# runs: can e2 ROUND-4 trio (DP-SCOUT-exp4 + dyn-SCOUT-exp4), seed-42 scene
# set (env.reset(seed=42+i), i=0..19), 20 guided rollouts, all.hdf5 written.
#
# Two runs that differ ONLY in n_envs (the batching coincidence B):
#   g1_guide_on_B20 : n_envs=20  (vis_final batch size)
#   g1_guide_on_B4  : n_envs=4   (the batch size that exposed the bug)
# PASS = B4 matches B20 on success + avg_jerk (per-row force B-independent),
#        and both are comparable to the pre-fix B=20 scale=0.5 record
#        (16/20, jerk 0.385; effective scale back then 0.5/20=0.025).
# Render-corruption guard: vis_validate.py gate; retry the SAME n_envs once.
#
# Usage:  GPU=<id> bash soe_scripts/fix01_confirm.sh
set -u
GPU=${GPU:?set GPU=<cuda id>}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
cd "$REPO" || exit 1

E2=$REPO/data/experiment2/can
DPDIR=$E2/train/DP/DP-SCOUT-exp4
VIBDIR=$E2/train/dyn/dyn-SCOUT-exp4
CORE=$E2/rollout/can_core.hdf5
CFG=configs/eval_can_e2fix01.yaml
[ -f "$CFG" ] || { echo "FATAL: missing $CFG (from eval_can_e2.yaml, guidance_scale 0.01)"; exit 1; }
DP=$(ls -t "$DPDIR"/checkpoints/*.ckpt 2>/dev/null | head -1)
[ -n "$DP" ] || { echo "FATAL: no ckpt under $DPDIR"; exit 1; }
VIB=$(ls -t "$VIBDIR"/*/scout_vib.ckpt 2>/dev/null | head -1)
[ -n "$VIB" ] || { echo "FATAL: no scout_vib.ckpt under $VIBDIR"; exit 1; }
[ -f "$CORE" ] || { echo "FATAL: missing core $CORE"; exit 1; }
OUTROOT=$REPO/data/vis_final/fix01_e2-SCOUT05-round4
echo "DP=$DP"
echo "VIB=$VIB"

run_one(){ # run_one <tag> <nenvs>
  local TAG=$1 NE=$2 TRY RC OK=0
  local DIR=$OUTROOT/g1_guide_on_$TAG
  mkdir -p "$DIR"
  for TRY in 1 2; do
    rm -f "$DIR/all.hdf5" "$DIR/success.hdf5"; rm -rf "$DIR/log"
    echo "[$TAG] n_envs=$NE try$TRY GPU$GPU $(date '+%F %T')"
    env CUDA_VISIBLE_DEVICES=$GPU "$PY" -m scout.eval.run_rollout \
      --config "$CFG" --task can \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
      --guide dyn --vib-ckpt "$VIB" \
      --eval-seed 42 --explore-seed 42 \
      --n-explore 20 --explore-try-times 1 \
      --n-init-states 20 --n-envs "$NE" \
      --exp-num 0 \
      --output-dir "$DIR" --output-success "$DIR/success.hdf5" \
      --output-all "$DIR/all.hdf5" --no-wandb \
      > "$DIR/rollout.stdout" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "[$TAG] try$TRY rc=$RC -- tail:"
      tail -n 6 "$DIR/rollout.stdout"
      continue
    fi
    if "$PY" soe_scripts/vis_validate.py "$DIR/all.hdf5" 20 > "$DIR/validate.log" 2>&1; then
      echo "[$TAG] try$TRY OK (validation passed) $(date '+%F %T')"
      OK=1; break
    fi
    echo "[$TAG] try$TRY RENDER CORRUPT -- validator says:"
    grep CORRUPT "$DIR/validate.log" | head -2
  done
  [ $OK -eq 1 ] || { echo "[$TAG] FAILED"; return 1; }
  return 0
}

run_one B20 20 && run_one B4 4 && echo "[fix01_confirm] all done $(date '+%F %T')"
