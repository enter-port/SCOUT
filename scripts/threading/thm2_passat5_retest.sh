#!/bin/bash
# thm2_passat5_retest.sh -- THREADING-MG-p2 seed-233 round-1 ckpt: rescue x5
# on the round-2 failure set with the C-recalibrated (eta, kappa), single
# process. Mirrors scripts/coffee/cfk_reanchor_r6probe.sh. Output lands in
# data/2026_9_22_threading_mg_p2/PROBE_CKAP/ (OUTSIDE the per-seed tree so the
# running chains' accum globs never see it).
# usage: bash scripts/threading/thm2_passat5_retest.sh <GPU> <ETA> <KAP>
set -u
GPU=${1:?gpu}; ETA=${2:?eta}; KAP=${3:?kap}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_22_threading_mg_p2
S=$CAMP/THREADING-MG-p2-s233
OUT=$CAMP/PROBE_CKAP/retest_e${ETA}_k${KAP}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$S/threading/train/DP/DP-ATY-exp1/checkpoints/299.ckpt
VIBC=$(ls -t $S/threading/train/dyn/dyn-ATY-exp1/*/scout_vib.ckpt | head -1)
CORE=$S/threading/rollout/threading_core.hdf5
FAILED=$S/threading/rollout/ATY-exp2/failed.json
for f in "$DPCKPT" "$VIBC" "$CORE" "$FAILED"; do
  [ -f "$f" ] || { echo "[ckap] FATAL missing $f"; exit 1; }
done
echo "[ckap] eta=$ETA kap=$KAP dp=$(basename $(dirname $(dirname $DPCKPT))) vib=$VIBC"
timeout -k 30 5400 env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl \
  TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout \
  --config configs/eval_threading_entropy.yaml --task threading --exp-num 0 \
  --base-dp-ckpt "$DPCKPT" --vib-ckpt "$VIBC" --core-hdf5 "$CORE" \
  --guide atypical --guidance-scale "$ETA" --atypical-cap "$KAP" \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue --failed-set-json "$FAILED" \
  --explore-try-times 5 \
  --n-envs 25 --no-wandb \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" --output-all "$OUT/all.hdf5" \
  > "$OUT/probe.stdout" 2>&1
rc=$?
echo "[ckap] rc=$rc $(date '+%F %T')"
J=$(ls "$OUT"/log/*_explore_*.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("[ckap] RESULT n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"),
       d.get("pass_at_5"), d.get("avg_jerk")))
EOF
grep -o 'mean_inject=[0-9.]*' "$OUT/probe.stdout" | tail -1 || true
exit $rc
