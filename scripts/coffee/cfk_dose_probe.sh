#!/bin/bash
# cfk_dose_probe.sh -- online dose-response probe for COFFEE-MG-p1
# (2026-09-20). COPY of scripts/coffee_prep/ckp_dose_probe.sh with these
# deltas ONLY: task/paths coffee (campaign 2026_9_20_coffee_mg_p1) and the
# failed-set source = GRID_s233/PRESEED/failed.json (the canonical preseeded
# eval's frozen failure set). Everything else identical: ONE guided
# explore-only rollout on the s233 FROZEN failed set (subset via
# --scene-slice SLOT:SHARDS -- 0:4 = shard 0 of 4, i.e. ~1/4 of the failure
# set), ATY or ORBIT at a given raw eta. Single process (no shards) so the
# [guidance-telemetry] mean_inject lines survive in probe.stdout.
# timeout 4500s kept from coffee_prep (coffee horizon 400, conservative cap).
# usage: bash scripts/coffee/cfk_dose_probe.sh <GPU> <ATY|ORBIT> <ETA> [TRIES] [SLICE] [KAP]
set -u
GPU=${1:?gpu}; ARM=${2:?ATY|ORBIT}; ETA=${3:?eta}; TRIES=${4:-2}; SLICE=${5:-0:4}
KAP=${6:-2.5}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S=$CAMP/COFFEE-MG-p1-s233
OUT=$CAMP/DOSEPROBE/${ARM}_eta${ETA}_k${KAP}
mkdir -p "$OUT"
cd "$ROOT" || exit 1
DPCKPT=$(ls -t $S/coffee/train/DP/DP-base/checkpoints/*.ckpt | head -1)
VIBC=$(ls -t $S/coffee/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$S/coffee/rollout/coffee_core.hdf5
FAILED=$CAMP/GRID_s233/PRESEED/failed.json
[ -f "$FAILED" ] || { echo "[dose-probe] FATAL: preseeded failed set missing: $FAILED"; exit 1; }
GUIDE=atypical; GARGS=(--atypical-cap "$KAP")
if [ "$ARM" = ORBIT ]; then
  GUIDE=orbit
  GARGS=(--atypical-cap "$KAP" --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05
         --orbit-sigma-decay 0.5 --orbit-round 1 --orbit-fb-clamp soft
         --orbit-noise-anneal 2)
fi
echo "[dose-probe] $ARM eta=$ETA kap=$KAP gpu=$GPU tries=$TRIES slice=$SLICE dp=$DPCKPT vib=$VIBC"
timeout -k 30 4500 env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl \
  TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout \
  --config configs/eval_coffee_entropy.yaml --task coffee --exp-num 0 \
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
