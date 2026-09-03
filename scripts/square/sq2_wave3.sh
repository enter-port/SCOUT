#!/bin/bash
# SQUARE beat-SOE wave-3 (2026-08-31): orbit sector='det' arms (B2) -- per-
# (scene,try) deterministic tangent direction, stratified angular coverage.
# Same triple as the known-good orbit sigma=0.25/kappa=2.5 grid cells so the
# ONLY delta vs those numbers is sector iid->det.
set -uo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
CORE=$R26/core_rebuild

run () {  # $1=name $2=seed
  local name=$1 seed=$2
  local out=data/particle/$name
  mkdir -p "$out"
  local dp=$R26/SQUARE-entropy-s$seed/square/train/DP/DP-base/checkpoints/599.ckpt
  local ts
  case $seed in
    233)   ts=20260826-112119 ;;
    2333)  ts=20260826-112147 ;;
    23333) ts=20260829-025739 ;;
  esac
  local vib=$R26/SQUARE-entropy-s$seed/square/train/dyn/dyn-base/$ts/scout_vib.ckpt
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name seed=$seed sector=det"
  if [ -n "${DRYRUN:-}" ]; then echo "DRYRUN: guide=orbit sector=det out=$out"; return 0; fi
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -u -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$dp" \
    --core-hdf5 "$CORE/square_core_s$seed.hdf5" \
    --guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 \
    --orbit-sector det --orbit-sector-seed 42 \
    --vib-ckpt "$vib" \
    --try-times 10 --explore-try-times 10 \
    --seed 42 --eval-seed 42 --explore-mode rescue \
    --failed-set-json "$out/failed_set.json" --save-failed-set "$out/failed_set.json" \
    --n-envs 50 --no-wandb --output-dir "$out" \
    >> "data/particle/$name.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name rc=$?"
}

case "${1:-}" in
  q0) run sq2_orbdet_s2333  2333 ;;
  q1) run sq2_orbdet_s23333 23333 ;;
  q4) run sq2_orbdet_s233   233 ;;
  q6) run sq2_orbk15_s233   233 ;;
  *) echo "usage: GPU=<g> bash $0 {q0|q1|q4|q6}"; exit 2 ;;
esac
