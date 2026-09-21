#!/bin/bash
# cfk_reanchor_r6probe.sh -- coffee r6 failed-set rescue with the RETRAINED
# dyn-ATY-exp5 at the RE-ANCHORED dose (eta_r, kappa_r) instead of the
# round-0 calibrated (5.6, 2.5). Same protocol as PROBE_DYNBASE_R6 (same DP
# ckpt, same scenes, x5 tries) so results are directly comparable:
#   field r6 (dyn-ATY-exp5 @ 5.6/2.5): 0/20 0/25 2/21
#   dyn-base swap (@ 5.6/2.5):        13/20 13/25 7/20
# usage: bash scripts/coffee/cfk_reanchor_r6probe.sh <GPU> <SEED> <ETA> <KAP>
set -u
GPU=${1:?gpu}; SEED=${2:?seed}; ETA=${3:?eta}; KAP=${4:?kap}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s$SEED
OUT=$CAMP/PROBE_REANCHOR_R6/s${SEED}_e${ETA}_k${KAP}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$S/coffee/train/DP/DP-ATY-exp5/checkpoints/299.ckpt
VIBC=$(ls -t $S/coffee/train/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)
CORE=$S/coffee/rollout/coffee_core.hdf5
FAILED=$S/coffee/rollout/ATY-exp6/failed.json
for f in "$DPCKPT" "$VIBC" "$CORE" "$FAILED"; do
  [ -f "$f" ] || { echo "[reanchor-r6] FATAL missing $f"; exit 1; }
done
echo "[reanchor-r6] seed=$SEED gpu=$GPU eta=$ETA kap=$KAP dp=$DPCKPT vib=$VIBC"
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
echo "[reanchor-r6] seed=$SEED rc=$rc $(date '+%F %T')"
J=$(ls "$OUT"/log/*_explore_*.json "$OUT"/log/*_rollout_*.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("[reanchor-r6] RESULT n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"),
       d.get("pass_at_5"), d.get("avg_jerk")))
EOF
grep -o 'mean_inject=[0-9.]*' "$OUT/probe.stdout" | tail -1 || true
exit $rc
