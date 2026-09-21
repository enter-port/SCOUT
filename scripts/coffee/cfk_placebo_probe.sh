#!/bin/bash
# cfk_placebo_probe.sh -- unguided retry baseline on the same r6 failed set
# with the same round5 DP ckpt (the missing anchor all three reviewers asked
# for). usage: bash scripts/coffee/cfk_placebo_probe.sh <GPU>
set -u
GPU=${1:?gpu}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s23333
OUT=$CAMP/PROBE_PARAMDEV_R6/placebo
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$S/coffee/train/DP/DP-ATY-exp5/checkpoints/299.ckpt
FAILED=$S/coffee/rollout/ATY-exp6/failed.json
[ -f "$DPCKPT" ] || { echo FATAL; exit 1; }
[ -f "$FAILED" ] || { echo FATAL; exit 1; }
echo "[placebo] gpu=$GPU dp=$DPCKPT"
timeout -k 30 5400 env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl \
  TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout \
  --config configs/eval_coffee_entropy.yaml --task coffee --exp-num 0 \
  --base-dp-ckpt "$DPCKPT" --guide off --core-hdf5 "$S/coffee/rollout/coffee_core.hdf5" \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue --failed-set-json "$FAILED" \
  --explore-try-times 5 \
  --n-envs 25 --no-wandb \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" --output-all "$OUT/all.hdf5" \
  > "$OUT/probe.stdout" 2>&1
rc=$?
echo "[placebo] rc=$rc $(date "+%F %T")"
J=$(ls "$OUT"/log/*_explore_*.json "$OUT"/log/*_rollout_*.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - "$J" << "PYEOF"
import json, sys
d = json.load(open(sys.argv[1]))
print("[placebo] RESULT n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"),
       d.get("pass_at_5"), d.get("avg_jerk")))
PYEOF
exit $rc
