#!/bin/bash
# r_probe_rollout_run.sh -- rollout-convention R probe driver (2026-09-22).
# Static lane partition of data/R_probe_20260922/jobs.txt (no cross-node
# flock): lane i takes jobs whose line index % n_lanes == i.
# usage: bash r_probe_rollout_run.sh <gpu_id> <lane_idx> <n_lanes>
set -u
ROOT=/root/workspace/baojiachun/scout
cd "$ROOT" || exit 1
OUT=data/R_probe_20260922
GPU=$1; LANE=$2; NL=$3
mapfile -t LINES < <(grep -n . "$OUT/jobs.txt" | awk -F: -v l=$LANE -v n=$NL '(($1-1)%n)==l{sub(/^[0-9]+:/,"");print}')
echo "[lane $LANE] ${#LINES[@]} jobs on GPU$GPU"
for JOB in "${LINES[@]}"; do
  IFS='|' read -r task seed venv vibcfg evalcfg cr core dp vib eta0 <<< "$JOB"
  tag=${task}_s${seed}
  PJ=$OUT/rollout_$tag.json
  if [ -s "$PJ" ]; then echo "[lane $LANE] SKIP $tag (exists)"; continue; fi
  PYB=/root/workspace/baojiachun/$venv/bin/python
  echo "[$(date '+%T')] GPU$GPU START $tag"
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU \
    "$PYB" scripts/coffee/r_probe_rollout.py \
      --eval-config "configs/$evalcfg" \
      --dp-ckpt "$dp" --vib-ckpt "$vib" --core-hdf5 "$core" \
      --eta0 "$eta0" --kappa0 2.5 --out "$PJ" \
      > "$OUT/rollout_$tag.stdout" 2>&1
  echo "[$(date '+%T')] GPU$GPU rc=$? $tag"
done
echo "[lane $LANE] DONE $(date '+%T')"
