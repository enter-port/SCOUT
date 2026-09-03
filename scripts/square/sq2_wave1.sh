#!/bin/bash
# SQUARE round1 beat-SOE campaign, wave 1 (2026-08-31): mechanism x seed matrix
# on the two hard seeds (s2333 gap -18, s23333 gap -13 vs SOE eta=1 r1 pass@10
# .74/.67). Arms: placebo (unguided retry) / atypical / orbit sigma=0.15 (dose
# recal; s23333 r1 was supersampled at 1.63) / particle G1 (lambda=0.25, start=0).
# All runs reuse the 0831 grid failed sets -> eval skipped, pure rescue x10.
# Usage: GPU=<g> bash /tmp/sq2_wave1.sh <q0|q1|q4|q6> ; DRYRUN=1 for argv check
set -uo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
CORE=$R26/core_rebuild

vibck () {  # $1=seed -> dyn-base ckpt
  local ts
  case $1 in
    233)   ts=20260826-112119 ;;
    2333)  ts=20260826-112147 ;;
    23333) ts=20260829-025739 ;;
  esac
  echo $R26/SQUARE-entropy-s$1/square/train/dyn/dyn-base/$ts/scout_vib.ckpt
}

run () {  # $1=name $2=seed $3=guide  rest: extra args
  local name=$1 seed=$2 guide=$3; shift 3
  local out=data/particle/$name
  mkdir -p "$out"
  local dp=$R26/SQUARE-entropy-s$seed/square/train/DP/DP-base/checkpoints/599.ckpt
  local extra=()
  if [ "$guide" != "off" ]; then extra+=(--vib-ckpt "$(vibck $seed)"); fi
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name guide=$guide $*"
  if [ -n "${DRYRUN:-}" ]; then
    echo "DRYRUN argv: $PY -u -m scout.eval.run_rollout --config configs/eval_square_entropy.yaml --task square --exp-num 0 --base-dp-ckpt $dp --core-hdf5 $CORE/square_core_s$seed.hdf5 --guide $guide --atypical-cap 2.5 --try-times 10 --explore-try-times 10 --seed 42 --eval-seed 42 --explore-mode rescue --failed-set-json $out/failed_set.json --save-failed-set $out/failed_set.json --n-envs 50 --no-wandb --output-dir $out ${extra[*]} $*"
    return 0
  fi
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -u -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$dp" \
    --core-hdf5 "$CORE/square_core_s$seed.hdf5" \
    --guide "$guide" --try-times 10 --explore-try-times 10 \
    --seed 42 --eval-seed 42 --explore-mode rescue \
    --failed-set-json "$out/failed_set.json" --save-failed-set "$out/failed_set.json" \
    --n-envs 50 --no-wandb --output-dir "$out" ${extra[@]+"${extra[@]}"} "$@" \
    >> "data/particle/$name.log" 2>&1
  echo "[$(date +%m-%d\ %H:%M:%S)] RUN $name rc=$? (log data/particle/$name.log)"
}

q0 () { run sq2_plc_s2333   2333  off                    ;
        run sq2_orb015_s2333 2333 orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.15 ; }
q1 () { run sq2_plc_s23333  23333 off                    ;
        run sq2_att_s23333  23333 atypical --atypical-cap 2.5 ; }
q4 () { run sq2_att_s2333   2333  atypical --atypical-cap 2.5 ;
        run sq2_par_s2333   2333  particle --atypical-cap 2.5 --pg-lambda 0.25 --pg-start 0 ; }
q6 () { run sq2_orb015_s23333 23333 orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.15 ;
        run sq2_par_s23333  23333 particle --atypical-cap 2.5 --pg-lambda 0.25 --pg-start 0 ; }

case "${1:-}" in
  q0|q1|q4|q6) "$1" ;;
  *) echo "usage: GPU=<g> bash $0 {q0|q1|q4|q6}"; exit 2 ;;
esac
