#!/bin/bash
# cfk_dynbase_r6probe.sh -- causal test for the ATY rescue collapse (2026-09-21).
#
# Question: does swapping the retrained (flattened) dyn back to the ROUND-0
# FROZEN dyn-base restore guided rescue on the round-6 failed scenes?
#   field round-6 (dyn-ATY-exp5 guiding DP-ATY-exp5): rescued 0/20, 0/25, 2/21
# This probe: same DP-ATY-exp5, same scenes (ATY-exp6/failed.json), same
# ATY dose (eta 5.6, kappa 2.5, gst 50 -- the round-0 calibrated dose, which
# is exactly right for dyn-base), same rescue protocol (x5 tries, rescue
# mode, initial states = failed-set init indices). Single process so the
# [guidance-telemetry] mean_inject lines survive in probe.stdout (field dose
# telemetry the campaign runs lost).
#
# Writes ONLY under data/2026_9_20_coffee_mg_p1/PROBE_DYNBASE_R6/s<SEED>/ --
# touches no chain directory.
#
# usage: bash scripts/coffee/cfk_dynbase_r6probe.sh <GPU> <SEED>
set -u
GPU=${1:?gpu}; SEED=${2:?233|2333|23333}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s$SEED
OUT=$CAMP/PROBE_DYNBASE_R6/s$SEED
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$S/coffee/train/DP/DP-ATY-exp5/checkpoints/299.ckpt
VIBC=$(ls -t $S/coffee/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$S/coffee/rollout/coffee_core.hdf5
FAILED=$S/coffee/rollout/ATY-exp6/failed.json
for f in "$DPCKPT" "$VIBC" "$CORE" "$FAILED"; do
  [ -f "$f" ] || { echo "[dynbase-r6] FATAL missing $f"; exit 1; }
done
echo "[dynbase-r6] seed=$SEED gpu=$GPU dp=$DPCKPT vib=$VIBC failed=$FAILED"
timeout -k 30 5400 env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl \
  TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout \
  --config configs/eval_coffee_entropy.yaml --task coffee --exp-num 0 \
  --base-dp-ckpt "$DPCKPT" --vib-ckpt "$VIBC" --core-hdf5 "$CORE" \
  --guide atypical --guidance-scale 5.6 --atypical-cap 2.5 \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue --failed-set-json "$FAILED" \
  --explore-try-times 5 \
  --n-envs 25 --no-wandb \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" --output-all "$OUT/all.hdf5" \
  > "$OUT/probe.stdout" 2>&1
rc=$?
echo "[dynbase-r6] seed=$SEED rc=$rc $(date '+%F %T')"
J=$(ls "$OUT"/log/*_explore_*.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("[dynbase-r6] RESULT seed-explore json:", sys.argv[1])
print("  n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"),
       d.get("pass_at_5"), d.get("avg_jerk")))
EOF
grep -c "guidance-telemetry" "$OUT/probe.stdout" || true
exit $rc
