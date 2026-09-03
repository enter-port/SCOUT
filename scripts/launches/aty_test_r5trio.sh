#!/bin/bash
# Atypical-only (phase-1, no orbit phase-2) control run on the r5 trio
# (user order 2026-09-02): DP-SCOUT-exp4/299.ckpt + dyn-SCOUT-exp4.
# Identical protocol/seed to the dead orbit r5 round (eval seed42 x100 ->
# rescue x10 on eval-failed scenes, env50) with --guide atypical at the
# SAME eta/cap as orbit's phase-1 climb (eta=3.0 from eval_square_entropy
# .yaml, cap=2.5). Isolates phase-2's contribution; paired comparison
# target: orbit r5 = 0/33 rescued, pass@10 0.67.
# Output: scout-rand/data/aty_test_s233_r4trio/ (standalone -- the chain
# data tree is not touched).
# Usage: GPU=4 bash /tmp/aty_test_r5trio.sh
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
ROOT=/root/workspace/baojiachun/scout-rand
T=$ROOT/data/2026_9_1_orbchain/ORBIT-s233/square
OUT=$ROOT/data/aty_test_s233_r4trio
mkdir -p "$OUT"
cd "$ROOT" || exit 1
export TMPDIR=/tmp
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
log(){ echo "[aty-test] $(date '+%F %T') $*"; }

log "START atypical r5-trio rollout GPU=$GPU (DP=$T/train/DP/DP-SCOUT-exp4/checkpoints/299.ckpt)"
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU \
  /root/workspace/baojiachun/.venv/bin/python -u -m scout.eval.run_rollout \
  --config configs/eval_square_entropy.yaml \
  --task square --exp-num 5 \
  --base-dp-ckpt "$T/train/DP/DP-SCOUT-exp4/checkpoints/299.ckpt" \
  --vib-ckpt "$T/train/dyn/dyn-SCOUT-exp4/20260902-075528/scout_vib.ckpt" \
  --core-hdf5 "$T/rollout/square_core.hdf5" \
  --guide atypical --atypical-cap 2.5 \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue --explore-try-times 10 \
  --n-envs 50 \
  --wandb-minimal \
  --output-dir "$OUT" \
  --output-success "$OUT/success.hdf5" \
  --output-all "$OUT/all.hdf5" \
  --wandb-name ATYP-s233-r5trio \
  --wandb-project SQUARE-9-1-orbit-s233 \
  > "$OUT/rollout.stdout" 2>&1
rc=$?
log "rollout rc=$rc"
if [ $rc -ne 0 ]; then
  log "FATAL: rollout failed -- tail of $OUT/rollout.stdout:"
  tail -20 "$OUT/rollout.stdout"
  exit $rc
fi
log "DONE -- summary json:"
cat "$OUT"/log/*.json
log "ALL DONE"
