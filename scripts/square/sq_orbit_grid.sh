#!/bin/bash
# SQUARE orbit sigma=0.25 grid (user order 2026-08-31): multiple seeds x chain
# rounds 1-3. Chain-round convention: r1 = base triple (DP-base/599 + dyn-base),
# r2 = exp1 triple (299 + dyn-SCOUT-exp1), r3 = exp2 triple. Each cell = one
# run_rollout: eval 100 scenes (unguided) -> save failed set -> orbit rescue x10.
# Idempotent: if failed_set.json exists the eval is skipped (crash-resume).
# s233 r1 already measured (sq_orb_s025, 36/62) -- NOT rerun.
# Usage: GPU=<g> bash /tmp/sq_orb_grid.sh cell <seed> <r1|r2|r3>
#        GPU=<g> bash /tmp/sq_orb_grid.sh q1|q2|q3
#        DRYRUN=1 GPU=1 bash /tmp/sq_orb_grid.sh cell 2333 r1   (argv check)
set -uo pipefail   # no -e: a queue continues past a single failed cell
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy

# seed -> (dyn-base, dyn-exp1, dyn-exp2) timestamp dirs (from 2026-08-31 inventory)
DYN_TS_233="20260826-112119 20260826-230101 20260827-061505"
DYN_TS_2333="20260826-112147 20260826-220844 20260827-042648"
DYN_TS_23333="20260829-025739 20260829-090657 20260829-155319"

assets () {  # $1=seed $2=round-index(1|2|3) -> stdout: "<DP ckpt> <VIB ckpt>"
  local E=$R26/SQUARE-entropy-s$1/square
  local ridx=$2
  local ts dp vib
  case $1 in
    233)   set -- $DYN_TS_233 ;;
    2333)  set -- $DYN_TS_2333 ;;
    23333) set -- $DYN_TS_23333 ;;
  esac
  shift $((ridx - 1)); ts=$1
  case $ridx in
    1) dp=$E/train/DP/DP-base/checkpoints/599.ckpt;       vib=$E/train/dyn/dyn-base/$ts/scout_vib.ckpt ;;
    2) dp=$E/train/DP/DP-SCOUT-exp1/checkpoints/299.ckpt;  vib=$E/train/dyn/dyn-SCOUT-exp1/$ts/scout_vib.ckpt ;;
    3) dp=$E/train/DP/DP-SCOUT-exp2/checkpoints/299.ckpt;  vib=$E/train/dyn/dyn-SCOUT-exp2/$ts/scout_vib.ckpt ;;
  esac
  echo "$dp $vib"
}

run_cell () {  # $1=seed $2=round-index
  local name=sq_orb025_s$1_r$2
  local out=data/particle/$name
  mkdir -p "$out"
  local av; av=$(assets "$1" "$2")
  local dp vib; read -r dp vib <<< "$av"
  echo "[$(date +%m-%d\ %H:%M:%S)] CELL $name DP=$dp VIB=$vib"
  if [ -n "${DRYRUN:-}" ]; then
    echo "DRYRUN argv: $PY -m scout.eval.run_rollout --config configs/eval_square_entropy.yaml --task square --exp-num 0 --base-dp-ckpt $dp --core-hdf5 $R26/SQUARE-entropy-s$1/square/rollout/square_core.hdf5 --guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 --vib-ckpt $vib --seed 42 --eval-seed 42 --explore-mode rescue --failed-set-json $out/failed_set.json --save-failed-set $out/failed_set.json --try-times 10 --explore-try-times 10 --n-envs 50 --no-wandb --output-dir $out"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES="$GPU" SCOUT_RENDER_GPU="$GPU" "$PY" -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$dp" \
    --core-hdf5 "$R26/SQUARE-entropy-s$1/square/rollout/square_core.hdf5" \
    --guide orbit --atypical-cap 2.5 \
    --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 \
    --vib-ckpt "$vib" \
    --seed 42 --eval-seed 42 \
    --explore-mode rescue \
    --failed-set-json "$out/failed_set.json" --save-failed-set "$out/failed_set.json" \
    --try-times 10 --explore-try-times 10 --n-envs 50 --no-wandb \
    --output-dir "$out" >> "data/particle/$name.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M:%S)] CELL $name rc=$? (log data/particle/$name.log)"
}

case "${1:-}" in
  cell) shift; GPU="${GPU:-1}" run_cell "$1" "$2" ;;
  q1) GPU="${GPU:-1}" run_cell 2333 1; run_cell 23333 2; run_cell 233 3 ;;
  q2) GPU="${GPU:-4}" run_cell 23333 1; run_cell 233 2; run_cell 2333 3 ;;
  q3) GPU="${GPU:-6}" run_cell 2333 2; run_cell 23333 3 ;;
  *) echo "usage: GPU=<g> $0 {cell <seed> <1|2|3>|q1|q2|q3}"; exit 2 ;;
esac
