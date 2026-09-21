#!/bin/bash
# tp13_probe.sh -- TRANSPORT-9-13 guidance dose probe on the s233 round0 trio
# (2026-09-13; user flow "先探针后开臂" -- raw th95-family doses are NOT
# calibrated for transport, VIB never trained on transport before).
# Runs ON the port-1022 container (6 GPUs), inside tmux tp13_probe:
#   phase 1 (blocking): base eval, guide=off, 100 fixed scenes seed42..141,
#     --save-failed-set -> failed_full.json  (ALSO = the first-ever transport
#     base-SR reading + smoke test of the transport rollout path);
#   phase 2: slice first PROBE_N=15 failed scenes -> failed15.json, then
#     spawn 5 rescue arms x5 tries on separate GPUs:
#       s0 (placebo, guide off) / a025 / a05 / a10 (atypical k2.5 raw s)
#       / orb (orbit s0.5 + lam.5 delta.25 sigma.05 decay.5 anneal2 fb-soft);
#     each wandb-minimal -> project TRANSPORT-9-13-probe (mean_inject/jerk
#     telemetry lives there -- the dose sanity band).
# Collect with tp13_probe_collect.sh once arms finish (jsons authoritative).
# gpu map: base-eval=0  s0=1  a025=2  a05=3  a10=4  orb=6  (GPU5/7 avoided:
# 1022's GPU5/GPU7 carry uncorrected ECC errors).
set -u
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
DR=$ROOT/data/2026_9_13_transport/TRANSPORT-s233
DATA=$DR/transport
CORE=$DATA/rollout/transport_core.hdf5
TDP=$DATA/train/DP; TDYN=$DATA/train/dyn
PDIR=$ROOT/data/2026_9_13_transport/probe_s233
PROBE_N=${PROBE_N:-15}
SEED=42
cd "$ROOT" || exit 1
mkdir -p "$PDIR"

export MUJOCO_GL=egl TMPDIR=/tmp CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONUNBUFFERED=1
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

# ---- resolve trio (round0 must be COMPLETE) ----
grep -q "ROUND0 transport seed=233 TOTAL" "$DATA/round.log" 2>/dev/null \
  || { echo "[probe] FATAL: s233 round0 TOTAL line missing"; exit 1; }
DPCKPT=$(ls -t "$TDP"/DP-base/checkpoints/*.ckpt 2>/dev/null | head -1)
VIBCKPT=$(ls -t "$TDYN"/dyn-base/*/scout_vib.ckpt 2>/dev/null | head -1)
[ -n "$DPCKPT" ] && [ -n "$VIBCKPT" ] || { echo "[probe] FATAL: trio incomplete (dp=$DPCKPT vib=$VIBCKPT)"; exit 1; }
echo "[probe] trio: dp=$DPCKPT"; echo "[probe] trio: vib=$VIBCKPT"

# ---- phase 1: base eval (guide off) + freeze full failed set ----
if [ -f "$PDIR/failed_full.json" ]; then
  echo "[probe] phase1: failed_full.json exists -- skip base eval"
else
  echo "[probe] phase1: base eval (guide off, 100 scenes seed42, n_envs 25) START $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=0 SCOUT_RENDER_GPU=0 $PY -m scout.eval.run_rollout \
    --config configs/eval_transport_entropy.yaml --task transport --exp-num 0 \
    --base-dp-ckpt "$DPCKPT" \
    --core-hdf5 "$CORE" \
    --guide off --seed $SEED --eval-seed $SEED \
    --explore-mode rescue --eval-only --save-failed-set "$PDIR/failed_full.json" \
    --n-envs 25 \
    --wandb-minimal \
    --output-dir "$PDIR/base" \
    --output-success "$PDIR/base/success.hdf5" \
    --output-all "$PDIR/base/all.hdf5" \
    --wandb-name PROBE-base-eval \
    --wandb-project TRANSPORT-9-13-probe \
    > "$PDIR/base_eval.stdout" 2>&1
  rc=$?
  echo "[probe] phase1 rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[probe] FATAL: base eval failed -- see $PDIR/base_eval.stdout"; exit 1; }
fi

# ---- slice first PROBE_N failed scenes ----
$PY - "$PDIR/failed_full.json" "$PDIR/failed15.json" "$PROBE_N" <<'PYEOF'
import sys, json
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
raw = json.load(open(src))
if isinstance(raw, dict):
    key = "failed" if "failed" in raw else list(raw.keys())[0]
    total = len(raw[key]); raw[key] = raw[key][:n]; k = key
else:
    total = len(raw); raw = raw[:n]; k = None
json.dump(raw, open(dst, "w"))
print(f"[probe-slice] {total} failed -> first {n} -> {dst}")
PYEOF
[ -f "$PDIR/failed15.json" ] || { echo "[probe] FATAL: slice failed"; exit 1; }

# ---- phase 2: 5 rescue arms, one GPU each (explicit per-arm tmux) ----
# spec format: TAG:GPU:ARGS  (ARGS hold the arm's guide flags; no colons in them)
for spec in \
  "s0:1:--guide off" \
  "a025:2:--guide atypical --atypical-cap 2.5 --guidance-scale 0.25" \
  "a05:3:--guide atypical --atypical-cap 2.5 --guidance-scale 0.5" \
  "a10:4:--guide atypical --atypical-cap 2.5 --guidance-scale 1.0" \
  "orb:6:--guide orbit --atypical-cap 2.5 --guidance-scale 0.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05 --orbit-sigma-decay 0.5 --orbit-round 1 --orbit-fb-clamp soft --orbit-noise-anneal 2" \
  ; do
  TAG=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; GARGS=${rest#*:}
  DIR=$PDIR/$TAG; mkdir -p "$DIR"
  VIB=""
  if [ "$TAG" != "s0" ]; then VIB="--vib-ckpt $VIBCKPT"; fi
  if tmux has-session -t "tp13_probe_$TAG" 2>/dev/null; then
    echo "[probe] tmux tp13_probe_$TAG exists -- skip"; continue
  fi
  tmux new-session -d -s "tp13_probe_$TAG" \
    "cd $ROOT && env CUDA_VISIBLE_DEVICES=$G SCOUT_RENDER_GPU=$G $PY -m scout.eval.run_rollout \
      --config configs/eval_transport_entropy.yaml --task transport --exp-num 0 \
      --base-dp-ckpt $DPCKPT --core-hdf5 $CORE \
      $GARGS --seed $SEED --eval-seed $SEED \
      --explore-mode rescue --failed-set-json $PDIR/failed15.json \
      --explore-try-times 5 --n-envs 25 --flush-every 100 $VIB \
      --wandb-minimal --wandb-project TRANSPORT-9-13-probe --wandb-name probe-$TAG \
      --output-dir $DIR --output-success $DIR/success.hdf5 --output-all $DIR/all.hdf5 \
      > $DIR/rollout.stdout 2>&1"
  echo "[probe] spawned tp13_probe_$TAG @GPU$G ($GARGS)"
done
echo "[probe] phase2: 5 arms spawned $(date '+%F %T'); collect via tp13_probe_collect.sh when jsons land"
