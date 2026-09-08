#!/bin/bash
# th95_p5meas.sh (2026-09-07) -- round6 pass@5 measurement addendum for
# TOOLHANG-9-5-orbit-s233 (user order: r6 was eval-only so it carries only
# first-try SR; this re-measures the SAME r5-retrained policy with rescue x5
# explore to produce pass@5). Isolated output dir <ARM>-pass5-r6 so existing
# r6 outputs and any future <ARM>-exp* accum globs are untouched; no retrain,
# no wandb, no data feedback. Protocol mirrors round_th95.sh [1/3] verbatim
# (eval seed 42 / 100 scenes, 8 shard workers x 25 envs, flush-every 100).
# Usage: bash scripts/toolhang/th95_p5meas.sh <ATY|ORBIT>
set -u
A=${1:?usage: th95_p5meas.sh <ATY|ORBIT>}
case "$A" in
  ATY)   GPU=2; GUIDE=atypical
         GEXTRA=(--atypical-cap 2.5 --guidance-scale 0.5) ;;
  ORBIT) GPU=4; GUIDE=orbit
         GEXTRA=(--atypical-cap 2.5 --guidance-scale 0.5
                 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05
                 --orbit-sigma-decay 0.5 --orbit-round 6 --orbit-fb-clamp soft
                 --orbit-noise-anneal 2) ;;
  *) echo "arm must be ATY or ORBIT (got: $A)"; exit 1 ;;
esac
REPO=/root/workspace/baojiachun/scout-th95
PY=/root/workspace/baojiachun/.venv/bin/python
DATA=$REPO/data/2026_9_5_toolhang/TOOLHANG-s233
TASK=tool_hang
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
RDIR=$DATA/$TASK/rollout/$A-pass5-r6
export MUJOCO_GL=egl TMPDIR=/tmp CUBLAS_WORKSPACE_CONFIG=:4096:8
export SCOUT_RENDER_GPU=$GPU PYTHONUNBUFFERED=1
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
cd "$REPO" || exit 1
mkdir -p "$RDIR/log"

DPCKPT=$(ls -t "$TDP/DP-$A-exp5/checkpoints/"*.ckpt 2>/dev/null | head -1)
VIBCKPT=$(ls -t "$TDYN/dyn-$A-exp5/"*/scout_vib.ckpt 2>/dev/null | head -1)
[ -n "$DPCKPT" ] && [ -n "$VIBCKPT" ] || { echo "FATAL: r5 ckpts missing for $A"; exit 1; }
VIBARGS=(--vib-ckpt "$VIBCKPT")
echo "[$(date '+%F %T')] p5meas $A START gpu=$GPU dp=${DPCKPT##*/} vib=$(dirname "${VIBCKPT#$TDYN/}")"

if [ -f "$RDIR/failed.json" ]; then
  echo "[1a] failed.json exists -- skip eval (resume)"
else
  echo "[1a] eval phase guide=$GUIDE n_envs=25 eval=42(100) -> $RDIR/failed.json"
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout \
    --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num 6 \
    --base-dp-ckpt "$DPCKPT" --core-hdf5 "$CORE" \
    --guide "$GUIDE" --seed 42 --eval-seed 42 \
    --explore-mode rescue --eval-only --save-failed-set "$RDIR/failed.json" \
    --n-envs 25 \
    ${VIBARGS[@]+"${VIBARGS[@]}"} \
    ${GEXTRA[@]+"${GEXTRA[@]}"} \
    --no-wandb \
    --output-dir "$RDIR" \
    --output-success "$RDIR/success.hdf5" \
    --output-all "$RDIR/all.hdf5" \
    > "$RDIR/eval.stdout" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[1a] eval rc=$rc -- see $RDIR/eval.stdout"; exit 1; }
  echo "[1a] eval done $(date '+%F %T')"
fi

echo "[1b] explore phase: 8 workers x n_envs=25 guide=$GUIDE failed-of-eval(x5) -> merged"
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU PYTHON=$PY CLEANUP_SHARDS=1 \
  bash scripts/infra/shard_rollout.sh 8 \
  "$RDIR/log/${TASK}_${A}_explore_exp6.json" \
  "$RDIR/success.hdf5" \
  "$RDIR/all.hdf5" \
  "$CORE" \
  -- \
  --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num 6 \
  --base-dp-ckpt "$DPCKPT" --core-hdf5 "$CORE" \
  --guide "$GUIDE" --seed 42 --eval-seed 42 \
  --explore-mode rescue --explore-try-times 5 \
  --failed-set-json "$RDIR/failed.json" \
  --n-envs 25 --flush-every 100 \
  ${VIBARGS[@]+"${VIBARGS[@]}"} \
  ${GEXTRA[@]+"${GEXTRA[@]}"} \
  --no-wandb \
  --output-dir "$RDIR" \
  --output-success "$RDIR/success.hdf5" \
  --output-all "$RDIR/all.hdf5" \
  > "$RDIR/explore.stdout" 2>&1
rc=$?
[ $rc -ne 0 ] && { echo "[1b] explore rc=$rc -- see $RDIR/explore.stdout"; exit 1; }

$PY - "$RDIR/log/${TASK}_${A}_explore_exp6.json" <<'PYEOF'
import sys, json
d = json.load(open(sys.argv[1]))
print(f"[RESULT] {sys.argv[1]}")
for k in ("success_rate", "exploration_rescued", "pass_at_5", "explore_try_times"):
    print(f"  {k} = {d.get(k)}")
PYEOF
echo "[$(date '+%F %T')] p5meas $A DONE"
