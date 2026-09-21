#!/bin/bash
# paramdev_launch.sh -- one-shot orchestrator for the param-dev acceptance
# test (coffee s23333, round5 ckpts, r6 failed set, rescue x5):
#   1) run the three deterministic calibrations (m1 yaml-only, m2 one GPU
#      forward pass, m3 json-only) -> PROBE_PARAMDEV_R6/calib_<m>.json
#   2) spawn three tmux rescue runs on GPU 0/1/2 (re-checking they are free)
# tmux sessions: paramdev_m1 / paramdev_m2 / paramdev_m3
set -u
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s23333
OUT=$CAMP/PROBE_PARAMDEV_R6
ETA0=5.6
KAP0=2.5
JSTAR=0.506522570213964   # GRID_s233 ATY eta5.6 k2.5 flight avg_jerk
mkdir -p "$OUT"
cd "$ROOT" || exit 1

# --- gate: GPUs 0/1/2 must be idle (no other users, shared-server rule) ---
for g in 0 1 2; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -lt 1000 ] || { echo "[paramdev] FATAL GPU$g busy (${used}MiB)"; exit 1; }
done

# --- calibrations ---
echo "[paramdev] calib m1 (information budget)"
$PY scripts/coffee/cfk_param_calib.py m1 \
  --base-dyn-dir "$S/coffee/train/dyn/dyn-base" \
  --round-dyn-dir "$S/coffee/train/dyn/dyn-ATY-exp5" \
  --eta0 $ETA0 --kappa0 $KAP0 --out "$OUT/calib_m1.json" | tee "$OUT/calib_m1.stdout"

echo "[paramdev] calib m2 (posterior precision, one forward pass on GPU0)"
CUDA_VISIBLE_DEVICES=0 $PY scripts/coffee/cfk_param_calib.py m2 \
  --seed-data-root "$S" \
  --base-vib "$(ls -t $S/coffee/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)" \
  --round-vib "$(ls -t $S/coffee/train/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)" \
  --dp-ckpt "$S/coffee/train/DP/DP-ATY-exp5/checkpoints/299.ckpt" \
  --eta0 $ETA0 --kappa0 $KAP0 --out "$OUT/calib_m2.json" | tee "$OUT/calib_m2.stdout"

echo "[paramdev] calib m3 (field jerk feedback)"
$PY scripts/coffee/cfk_param_calib.py m3 \
  --explore-json "$S/coffee/rollout/ATY-exp6/log/coffee_ATY_explore_exp6.json" \
  --jerk-star $JSTAR --eta-current $ETA0 --kappa0 $KAP0 \
  --out "$OUT/calib_m3.json" | tee "$OUT/calib_m3.stdout"

# --- launch three rescue runs (tmux, session-safe, no shell-chain backgrounding) ---
i=0
for m in m1 m2 m3; do
  eta=$(python3 -c "import json;print('%.6g'%json.load(open('$OUT/calib_$m.json'))['eta'])")
  kap=$(python3 -c "import json;print('%.6g'%json.load(open('$OUT/calib_$m.json'))['kappa'])")
  gpu=$i
  echo "[paramdev] spawn $m gpu=$gpu eta=$eta kap=$kap"
  tmux new-session -d -s paramdev_$m \
    "cd $ROOT && bash scripts/coffee/cfk_paramdev_probe.sh $gpu $m $eta $kap >> $OUT/${m}.console.log 2>&1"
  i=$((i+1))
done
echo "[paramdev] LAUNCH_DONE $(date '+%F %T')"
