#!/bin/bash
# thm3_stage3.sh -- stage 3 of the THM3 plan (user 2026-09-23, autonomous):
# kappa cross-round recalibration exploration on the "new round-1 ckpt" pair
# (B2-retrained DP + A3 dyn, the healthy-recipe pair from stage 1/2).
#   calib     <GPU> : (a) eta rcalib on the new pair, ETA0=1.0, band
#                     [0.009,0.011] (same method as the p2 chain);
#                     (b) kappa v2 C-anchored calibration (C = step-averaged
#                     uncapped KL, kappa' = kappa * C_target/C_mean; C_target
#                     measured on the BASE pair at the base's rcalib eta
#                     4.6928). Writes PROBE_AB/stage3_dose.json.
#   failedset <GPU> : eval the B2 DP once over the 100 fixed scenes
#                     (42..141) -> failed set (same protocol as a round's
#                     phase A). Writes PROBE_AB/stage3_failed.json.
#   arm <GPU> <KAPPA> <TAG> : rescue x5 on that failed set with the
#                     calibrated eta + given kappa, NEW explore recording
#                     (--stop-on-first-success), report pass@5/jerk/inject.
# usage: bash scripts/threading/thm3_stage3.sh <calib|failedset|arm...> ...
set -u
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_22_threading_mg_p2
S=$CAMP/THREADING-MG-p2-s233/threading
EXP=$CAMP/PROBE_AB
cd "$ROOT" || exit 1
CORE=$S/rollout/threading_core.hdf5
ESBASE=$S/train/DP/DP-base/checkpoints/599.ckpt
VIBBASE=$(ls -t $S/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
B2DP=$EXP/train_DP_B2/checkpoints/599.ckpt
A3VIB=$EXP/train_A3  # newest scout_vib.ckpt under it
BASE_ETA=4.692821504030686   # p2 s233 r1 rcalib (ETA0=1.0) -- the base anchor
for f in "$CORE" "$ESBASE" "$VIBBASE" "$B2DP"; do
  [ -f "$f" ] || { echo "[s3] FATAL missing $f"; exit 1; }
done
A3VIBC=$(ls -t $A3VIB/*/scout_vib.ckpt 2>/dev/null | head -1)
[ -n "$A3VIBC" ] || { echo "[s3] FATAL no A3 vib ckpt under $A3VIB (run arm A3 first)"; exit 1; }

CMD=${1:?calib|failedset|arm}
case "$CMD" in
calib)
  GPU=${2:?gpu}
  echo "[s3] (a) eta rcalib on new pair (ETA0=1.0) start $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/coffee/cfk_rcalib.py \
    --eval-config configs/eval_threading_entropy.yaml \
    --dp-ckpt "$B2DP" --vib-ckpt "$A3VIBC" --core-hdf5 "$CORE" \
    --eta-prev 1.0 --kappa-prev 2.5 \
    --out $EXP/stage3_eta.json > $EXP/stage3_eta.stdout 2>&1
  RC=$?
  [ $RC -ne 0 ] && { echo "[s3] eta rcalib FAILED rc=$RC"; tail -8 $EXP/stage3_eta.stdout; exit 1; }
  ETA=$($PY -c "import json;print('%.6g'%json.load(open('$EXP/stage3_eta.json'))['eta'])")
  RV=$($PY -c "import json;d=json.load(open('$EXP/stage3_eta.json'));print('R_prev=%.4g R_final=%.4g converged=%s n_upd=%d'%(d['R_prev'],d['R_mean'],d['converged'],d['n_updates']))")
  echo "[s3] eta rcalib: $RV -> eta=$ETA"
  echo "[s3] (b) kappa v2 C-anchored calib start $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/threading/thm2_c_calib.py \
    --eval-config configs/eval_threading_entropy.yaml \
    --core-hdf5 "$CORE" \
    --base-dp-ckpt "$ESBASE" --base-vib-ckpt "$VIBBASE" --base-eta $BASE_ETA \
    --round-dp-ckpt "$B2DP" --round-vib-ckpt "$A3VIBC" --round-eta "$ETA" \
    --kappa0 2.5 --out $EXP/stage3_kcalib.json > $EXP/stage3_kcalib.stdout 2>&1
  RC=$?
  [ $RC -ne 0 ] && { echo "[s3] kappa calib FAILED rc=$RC"; tail -8 $EXP/stage3_kcalib.stdout; exit 1; }
  KAP=$($PY -c "import json;d=json.load(open('$EXP/stage3_kcalib.json'));print('%.6g'%(d.get('kappa') if isinstance(d.get('kappa'),(int,float)) else d['kappa'][-1]))")
  echo "[s3] kappa calib -> kappa=$KAP"
  $PY - "$ETA" "$KAP" <<'PYEOF'
import json, sys
json.dump({"eta": float(sys.argv[1]), "kappa": float(sys.argv[2])},
          open("/root/workspace/baojiachun/scout/data/2026_9_22_threading_mg_p2/PROBE_AB/stage3_dose.json", "w"), indent=1)
print("[s3] dose json written")
PYEOF
  echo "[s3] CALIB ALL DONE eta=$ETA kappa=$KAP $(date '+%F %T')"
  ;;
failedset)
  GPU=${2:?gpu}
  OUT=$EXP/stage3_failedset
  mkdir -p "$OUT"
  echo "[s3] B2 DP eval on 100 fixed scenes start $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp \
    PYTHONUNBUFFERED=1 timeout -k 30 7200 \
    $PY -m scout.eval.run_rollout \
    --config configs/eval_threading_entropy.yaml --task threading --exp-num 0 \
    --base-dp-ckpt "$B2DP" --core-hdf5 "$CORE" \
    --guide off --seed 42 --eval-seed 42 \
    --explore-mode rescue --eval-only --save-failed-set "$EXP/stage3_failed.json" \
    --n-envs 25 --no-wandb \
    --output-dir "$OUT" > "$OUT/eval.stdout" 2>&1
  RC=$?
  [ $RC -ne 0 ] && { echo "[s3] eval FAILED rc=$RC"; tail -8 "$OUT/eval.stdout"; exit 1; }
  $PY -c "import json;d=json.load(open('$EXP/stage3_failed.json'));print('[s3] failed set: %d failed of %d (SR %.3f)'%(len(d['failed_init_indices']),d['n_eval'],d['baseline_solved']/d['n_eval']))"
  echo "[s3] FAILEDSET ALL DONE $(date '+%F %T')"
  ;;
arm)
  GPU=${2:?gpu}; KAP=${3:?kappa}; TAG=${4:?tag}
  [ -f $EXP/stage3_dose.json ] || { echo "[s3-arm] FATAL no stage3_dose.json (run calib first)"; exit 1; }
  [ -f $EXP/stage3_failed.json ] || { echo "[s3-arm] FATAL no stage3_failed.json (run failedset first)"; exit 1; }
  ETA=$($PY -c "import json;print('%.6g'%json.load(open('$EXP/stage3_dose.json'))['eta'])")
  OUT=$EXP/stage3_arm_$TAG
  mkdir -p "$OUT"
  echo "[s3-arm] $TAG: eta=$ETA kappa=$KAP rescue x5 (stop-on-first-success) start $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp \
    PYTHONUNBUFFERED=1 timeout -k 30 14400 \
    $PY -m scout.eval.run_rollout \
    --config configs/eval_threading_entropy.yaml --task threading --exp-num 0 \
    --base-dp-ckpt "$B2DP" --vib-ckpt "$A3VIBC" --core-hdf5 "$CORE" \
    --guide atypical --guidance-scale "$ETA" --atypical-cap "$KAP" \
    --seed 42 --eval-seed 42 \
    --explore-mode rescue --failed-set-json "$EXP/stage3_failed.json" \
    --explore-try-times 5 --stop-on-first-success \
    --n-envs 25 --no-wandb \
    --output-dir "$OUT" --output-success "$OUT/success.hdf5" --output-all "$OUT/all.hdf5" \
    > "$OUT/probe.stdout" 2>&1
  rc=$?
  echo "[s3-arm] $TAG rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { tail -8 "$OUT/probe.stdout"; exit 1; }
  J=$(ls "$OUT"/log/*_explore_*.json 2>/dev/null | head -1)
  [ -n "$J" ] && $PY - "$J" <<'EOF'
import json, sys
d = json.load(open(sys.argv[1]))
print("[s3-arm] RESULT n_failed=%s rescued=%s pass_at_5=%s avg_jerk=%s collected=%s stop_first=%s" %
      (d.get("n_failed"), d.get("exploration_rescued"), d.get("pass_at_5"),
       d.get("avg_jerk"), d.get("collected_trajs"), d.get("stop_on_first_success")))
EOF
  grep -o 'mean_inject=[0-9.]*' "$OUT/probe.stdout" | tail -1 || true
  echo "[s3-arm] $TAG ALL DONE $(date '+%F %T')"
  ;;
*) echo "unknown cmd $CMD"; exit 1 ;;
esac
