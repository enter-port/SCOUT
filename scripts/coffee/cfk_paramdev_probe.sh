#!/bin/bash
# cfk_paramdev_probe.sh -- param-dev acceptance test: coffee s23333 round5
# ckpts (DP-ATY-exp5 + dyn-ATY-exp5) on the ATY-arm r6 failed set, rescue x5,
# at the RECALIBRATED (eta, kappa) of one of the three methods (m1/m2/m3).
# Protocol identical to PROBE_REANCHOR_R6 / PROBE_DYNBASE_R6 for direct
# comparison:
#   field r6 (5.6/2.5):  2/21, pass@5 0.81, jerk 1.27
#   probe reanchor (4.45/1.982): 1/21
#   dyn-base swap (5.6/2.5): 7/21, pass@5 0.86
#   GATE: pass@5 > 0.86  <=>  rescued >= 8/21
# usage: bash scripts/coffee/cfk_paramdev_probe.sh <GPU> <METHOD> <ETA> <KAP>
set -u
GPU=${1:?gpu}; METHOD=${2:?method}; ETA=${3:?eta}; KAP=${4:?kap}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s23333
OUT=$CAMP/PROBE_PARAMDEV_R6/${METHOD}_e${ETA}_k${KAP}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$S/coffee/train/DP/DP-ATY-exp5/checkpoints/299.ckpt
VIBC=$(ls -t $S/coffee/train/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)
CORE=$S/coffee/rollout/coffee_core.hdf5
FAILED=$S/coffee/rollout/ATY-exp6/failed.json
for f in "$DPCKPT" "$VIBC" "$CORE" "$FAILED"; do
  [ -f "$f" ] || { echo "[paramdev] FATAL missing $f"; exit 1; }
done
echo "[paramdev] method=$METHOD gpu=$GPU eta=$ETA kap=$KAP dp=$DPCKPT vib=$VIBC"
timeout -k 30 5400 env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl \
  TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout \
  --config configs/eval_coffee_entropy.yaml --task coffee --exp-num 0 \
  --base-dp-ckpt "$DPCKPT" --vib-ckpt "$VIBC" --core-hdf5 "$CORE" \
  --guide atypical --guidance-scale "$ETA" --atypical-cap "$KAP" \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue --failed-set-json "$FAILED" \
  --explore-try-times 5 \
  --n-envs 25 --no-wandb \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" --output-all "$OUT/all.hdf5" \
  > "$OUT/probe.stdout" 2>&1
rc=$?
echo "[paramdev] method=$METHOD rc=$rc $(date '+%F %T')"
J=$(ls "$OUT"/log/*_explore_*.json "$OUT"/log/*_rollout_*.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("[paramdev] RESULT n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"),
       d.get("pass_at_5"), d.get("avg_jerk")))
EOF
grep -o 'mean_inject=[0-9.]*' "$OUT/probe.stdout" | tail -1 || true
exit $rc
