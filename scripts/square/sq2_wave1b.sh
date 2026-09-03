#!/bin/bash
# SQUARE beat-SOE wave-1b (2026-08-31): s233 arms with explore_detail (the
# 0831 grid deleted per-scene data; curves + per-scene sets needed for mixing).
# orbit uses sigma=0.25 (the known-good s233 config, aggregate 36/62).
set -uo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
CORE=$R26/core_rebuild
VIB=$R26/SQUARE-entropy-s233/square/train/dyn/dyn-base/20260826-112119/scout_vib.ckpt
DP=$R26/SQUARE-entropy-s233/square/train/DP/DP-base/checkpoints/599.ckpt

run () {  # $1=name $2=guide  rest: extras
  local name=$1 guide=$2; shift 2
  local out=data/particle/$name
  mkdir -p "$out"
  local extra=()
  if [ "$guide" != "off" ]; then extra+=(--vib-ckpt "$VIB"); fi
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name guide=$guide $*"
  if [ -n "${DRYRUN:-}" ]; then
    echo "DRYRUN: guide=$guide out=$out extras=$*"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -u -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$DP" \
    --core-hdf5 "$CORE/square_core_s233.hdf5" \
    --guide "$guide" --try-times 10 --explore-try-times 10 \
    --seed 42 --eval-seed 42 --explore-mode rescue \
    --failed-set-json "$out/failed_set.json" --save-failed-set "$out/failed_set.json" \
    --n-envs 50 --no-wandb --output-dir "$out" ${extra[@]+"${extra[@]}"} "$@" \
    >> "data/particle/$name.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name rc=$?"
}

case "${1:-}" in
  q0) run sq2_orb025_s233 orbit    --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 ;;
  q1) run sq2_att_s233    atypical --atypical-cap 2.5 ;;
  q4) run sq2_plc_s233    off ;;
  q6) run sq2_par_s233    particle --atypical-cap 2.5 --pg-lambda 0.25 --pg-start 0 ;;
  *) echo "usage: GPU=<g> bash $0 {q0|q1|q4|q6}"; exit 2 ;;
esac
