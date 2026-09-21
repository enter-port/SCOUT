#!/bin/bash
# thread_dose_probe.sh -- online dose-response probe (2026-09-17, user-approved
# telemetry-first recalibration after the 0.1-2.0 grid showed dose-invariance).
# ONE guided explore-only rollout on the s233 FROZEN failed set (subset via
# --scene-slice), ATY or ORBIT at a given raw eta. Single process (no shards)
# so the [guidance-telemetry] mean_inject lines survive in probe.stdout.
# usage: bash scripts/threading/thread_dose_probe.sh <GPU> <ATY|ORBIT> <ETA> [TRIES] [SLICE]
set -u
GPU=${1:?gpu}; ARM=${2:?ATY|ORBIT}; ETA=${3:?eta}; TRIES=${4:-2}; SLICE=${5:-0:4}
KAP=${6:-2.5}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
S=$ROOT/data/2026_9_16_threading_p1/THREADING-MG-p1-s233
OUT=$ROOT/data/2026_9_16_threading_p1/DOSEPROBE/${ARM}_eta${ETA}_k${KAP}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$(ls -t $S/threading/train/DP/DP-base/checkpoints/*.ckpt | head -1)
VIBC=$(ls -t $S/threading/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$S/threading/rollout/threading_core.hdf5
FAILED=$S/threading/rollout/ATY-exp1/failed.json
GUIDE=atypical; GARGS=(--atypical-cap "$KAP")
if [ "$ARM" = ORBIT ]; then
  GUIDE=orbit
  GARGS=(--atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05
         --orbit-sigma-decay 0.5 --orbit-round 1 --orbit-fb-clamp soft
         --orbit-noise-anneal 2)
fi
echo "[dose-probe] $ARM eta=$ETA gpu=$GPU tries=$TRIES slice=$SLICE dp=$DPCKPT vib=$VIBC"
timeout -k 30 2700 env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl \
  TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout \
  --config configs/eval_threading_entropy.yaml --task threading --exp-num 0 \
  --base-dp-ckpt "$DPCKPT" --vib-ckpt "$VIBC" --core-hdf5 "$CORE" \
  --guide "$GUIDE" --guidance-scale "$ETA" ${GARGS[@]+"${GARGS[@]}"} \
  --seed 42 --eval-seed 42 \
  --explore-mode rescue --failed-set-json "$FAILED" \
  --explore-try-times "$TRIES" --scene-slice "$SLICE" \
  --n-envs 13 --no-wandb \
  --output-dir "$OUT" --output-success "$OUT/success.hdf5" --output-all "$OUT/all.hdf5" \
  > "$OUT/probe.stdout" 2>&1
rc=$?
echo "[dose-probe] $ARM eta=$ETA rc=$rc $(date '+%F %T')"
grep -c "guidance-telemetry" "$OUT/probe.stdout" || true
exit $rc
