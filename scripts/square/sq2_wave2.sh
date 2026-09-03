#!/bin/bash
# SQUARE beat-SOE wave-2 data (2026-08-31): kappa-stratified orbit arms --
# radial coverage inside orbit (all existing CLI; pooling across kappa is the
# stratification). kappa=1.5 (near excursions) and 4.0 (deep excursions) on the
# two hard seeds; kappa=2.5 already measured (grid + wave-1b).
set -uo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
CORE=$R26/core_rebuild

run () {  # $1=name $2=seed $3=kappa
  local name=$1 seed=$2 kap=$3
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
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name kappa=$kap seed=$seed"
  if [ -n "${DRYRUN:-}" ]; then echo "DRYRUN: guide=orbit kappa=$kap out=$out"; return 0; fi
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -u -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$dp" \
    --core-hdf5 "$CORE/square_core_s$seed.hdf5" \
    --guide orbit --atypical-cap "$kap" --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 \
    --vib-ckpt "$vib" \
    --try-times 10 --explore-try-times 10 \
    --seed 42 --eval-seed 42 --explore-mode rescue \
    --failed-set-json "$out/failed_set.json" --save-failed-set "$out/failed_set.json" \
    --n-envs 50 --no-wandb --output-dir "$out" \
    >> "data/particle/$name.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name rc=$?"
}

case "${1:-}" in
  q0) run sq2_orbk15_s2333   2333  1.5 ;;
  q1) run sq2_orbk15_s23333  23333 1.5 ;;
  q4) run sq2_orbk40_s2333   2333  4.0 ;;
  q6) run sq2_orbk40_s23333  23333 4.0 ;;
  *) echo "usage: GPU=<g> bash $0 {q0|q1|q4|q6}"; exit 2 ;;
esac
