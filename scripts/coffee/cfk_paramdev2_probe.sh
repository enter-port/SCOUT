#!/bin/bash
# cfk_paramdev2_probe.sh -- beat-placebo acceptance test (param-dev goal 2):
# coffee s23333, flight round r in {2..5}, guided rescue x5 on the PLACEBO
# (DP) arm's round-r failed set using the previous round's ckpts:
#   policy = DP-DP-exp{r-1}   (the placebo arm's own previous-round DP)
#   dyn    = dyn-ATY-exp{r-1} (the DP arm retrains no dyn -- empty dirs)
# Success per round: rescued_guided > rescued_DP(r), i.e. pass@5 beats the
# placebo json read from the server:
#   r2: 9/15 -> need >=10   r3: 12/15 -> need >=13
#   r4: 7/12 -> need >=8    r5: 8/11 -> need >=9
# usage: bash scripts/coffee/cfk_paramdev2_probe.sh <GPU> <FLIGHT_ROUND> <ETA> <KAP> [aty|base]
set -u
GPU=${1:?gpu}; R=${2:?flight_round}; ETA=${3:?eta}; KAP=${4:?kap}
VIBMODE=${5:-aty}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s23333
PM1=$((R-1))
OUT=$CAMP/PROBE_PARAMDEV_R2/flight_r${R}_e${ETA}_k${KAP}${VIBMODE:+_$VIBMODE}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$S/coffee/train/DP/DP-DP-exp${PM1}/checkpoints/299.ckpt
if [ "$VIBMODE" = "base" ]; then
  VIBC=$(ls -t $S/coffee/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
else
  VIBC=$(ls -t $S/coffee/train/dyn/dyn-ATY-exp${PM1}/*/scout_vib.ckpt | head -1)
fi
CORE=$S/coffee/rollout/coffee_core.hdf5
FAILED=$S/coffee/rollout/DP-exp${R}/failed.json
for f in "$DPCKPT" "$VIBC" "$CORE" "$FAILED"; do
  [ -f "$f" ] || { echo "[paramdev2] FATAL missing $f"; exit 1; }
done
echo "[paramdev2] flight=r$R gpu=$GPU eta=$ETA kap=$KAP dp=$DPCKPT vib=$VIBC"
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
echo "[paramdev2] flight=r$R rc=$rc $(date '+%F %T')"
J=$(ls "$OUT"/log/*_explore_*.json "$OUT"/log/*_rollout_*.json 2>/dev/null | head -1)
[ -n "$J" ] && python3 - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("[paramdev2] RESULT flight n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"),
       d.get("pass_at_5"), d.get("avg_jerk")))
EOF
grep -o 'mean_inject=[0-9.]*' "$OUT/probe.stdout" | tail -1 || true
exit $rc
