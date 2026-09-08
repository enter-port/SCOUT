#!/bin/bash
# th95_p5meas_dp.sh (2026-09-07) -- DP-arm variant of th95_p5meas.sh for the
# TOOLHANG-9-5-orbit-s233 round6 pass@5 figure (three complete lines need the
# DP r6 point too). Same protocol: r5 DP ckpt (DP-DP-exp5/299.ckpt), guide off,
# eval seed 42 / 100 scenes -> rescue x5, isolated dir DP-pass5-r6, no retrain,
# no wandb. Separate file: th95_p5meas.sh was recently executed and its case
# block only knows ATY|ORBIT (in-place edit of recently-run scripts avoided).
# Usage: bash scripts/toolhang/th95_p5meas_dp.sh
set -u
A=DP; GPU=0; GUIDE=off
REPO=/root/workspace/baojiachun/scout-th95
PY=/root/workspace/baojiachun/.venv/bin/python
DATA=$REPO/data/2026_9_5_toolhang/TOOLHANG-s233
TASK=tool_hang
TDP=$DATA/$TASK/train/DP
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
RDIR=$DATA/$TASK/rollout/$A-pass5-r6
export MUJOCO_GL=egl TMPDIR=/tmp CUBLAS_WORKSPACE_CONFIG=:4096:8
export SCOUT_RENDER_GPU=$GPU PYTHONUNBUFFERED=1
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
cd "$REPO" || exit 1
mkdir -p "$RDIR/log"

DPCKPT=$(ls -t "$TDP/DP-$A-exp5/checkpoints/"*.ckpt 2>/dev/null | head -1)
[ -n "$DPCKPT" ] || { echo "FATAL: r5 DP ckpt missing"; exit 1; }
echo "[$(date '+%F %T')] p5meas DP START gpu=$GPU dp=${DPCKPT##*/}"

if [ -f "$RDIR/failed.json" ]; then
  echo "[1a] failed.json exists -- skip eval (resume)"
else
  echo "[1a] eval phase guide=off n_envs=25 eval=42(100) -> $RDIR/failed.json"
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout \
    --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num 6 \
    --base-dp-ckpt "$DPCKPT" --core-hdf5 "$CORE" \
    --guide off --seed 42 --eval-seed 42 \
    --explore-mode rescue --eval-only --save-failed-set "$RDIR/failed.json" \
    --n-envs 25 --no-wandb \
    --output-dir "$RDIR" \
    --output-success "$RDIR/success.hdf5" \
    --output-all "$RDIR/all.hdf5" \
    > "$RDIR/eval.stdout" 2>&1
  rc=$?
  [ $rc -ne 0 ] && { echo "[1a] eval rc=$rc -- see $RDIR/eval.stdout"; exit 1; }
  echo "[1a] eval done $(date '+%F %T')"
fi

echo "[1b] explore phase: 8 workers x n_envs=25 guide=off failed-of-eval(x5) -> merged"
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU PYTHON=$PY CLEANUP_SHARDS=1 \
  bash scripts/infra/shard_rollout.sh 8 \
  "$RDIR/log/${TASK}_${A}_explore_exp6.json" \
  "$RDIR/success.hdf5" \
  "$RDIR/all.hdf5" \
  "$CORE" \
  -- \
  --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num 6 \
  --base-dp-ckpt "$DPCKPT" --core-hdf5 "$CORE" \
  --guide off --seed 42 --eval-seed 42 \
  --explore-mode rescue --explore-try-times 5 \
  --failed-set-json "$RDIR/failed.json" \
  --n-envs 25 --flush-every 100 \
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
echo "[$(date '+%F %T')] p5meas DP DONE"
