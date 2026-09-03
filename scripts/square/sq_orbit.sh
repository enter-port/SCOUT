#!/bin/bash
# SQUARE orbit (user 2026-08-31, after G1/G2 results): lam=0.5, delta=0.25,
# sigma = calibrated via probe6, frozen 62-scene failure set, rescue x10, env50.
# Usage (server):  bash /tmp/sq_orbit.sh run <name> <sigma> <gpu>
#                  bash /tmp/sq_orbit.sh probe6 <name> <sigma> <gpu>
set -euo pipefail
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
E=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square
VIB="$E/train/dyn/dyn-base/20260826-112119/scout_vib.ckpt"
cmd="$1"; shift
run () {  # $1=name $2=sigma $3=gpu $4=failed-set-json
  mkdir -p "data/particle/$1"
  env CUDA_VISIBLE_DEVICES="$3" SCOUT_RENDER_GPU="$3" \
    /root/workspace/baojiachun/.venv/bin/python -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$E/train/DP/DP-base/checkpoints/599.ckpt" \
    --core-hdf5 "$E/rollout/square_core.hdf5" \
    --guide orbit --atypical-cap 2.5 \
    --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma "$2" \
    --vib-ckpt "$VIB" \
    --seed 42 --eval-seed 42 \
    --explore-mode rescue --failed-set-json "$4" \
    --try-times 10 --explore-try-times 10 --n-envs 50 --no-wandb \
    --output-dir "data/particle/$1"
  echo "SQ_ORBIT_$1_DONE rc=$?"
}
case "$cmd" in
  probe6) run "$1" "$2" "$3" data/particle/sq_failed_set_probe6.json ;;
  run)    run "$1" "$2" "$3" data/particle/sq_failed_set_base_s233.json ;;
  *) echo "usage: $0 {probe6|run} <name> <sigma> <gpu>"; exit 2 ;;
esac
