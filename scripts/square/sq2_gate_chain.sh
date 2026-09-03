#!/bin/bash
# sq2 gate chain (2026-08-31 corrected protocol): run two gate arms
# sequentially on one GPU. Each arm = 20env gate subset (front-20 failures)
# x 10 tries, explore seed 43, via /tmp/sq2_conf.sh GATE=1.
# Usage: GPU=<g> bash /tmp/sq2_gate_chain.sh <arm1> <arm2>
set -uo pipefail
G=${GPU:?}; A1=${1:?}; A2=${2:?}
cd /root/workspace/baojiachun/scout-rand
for a in "$A1" "$A2"; do
  echo "[$(date '+%m-%d %H:%M:%S')] gate-chain GPU$G arm=$a START"
  # SEED=42: gate screening uses the historical RNG stream (scene set AND retry
  # RNG both 42 -- passes the failed-set base_seed guard). Fresh-seed
  # confirmation (stage 2) uses --rescue-seed via SEED=43 once that flag exists.
  SPLIT="$a:10" SEED=42 GATE=1 GPU=$G bash /tmp/sq2_conf.sh 233
  rc=$?
  echo "[$(date '+%m-%d %H:%M:%S')] gate-chain GPU$G arm=$a rc=$rc"
done
echo "[$(date '+%m-%d %H:%M:%S')] gate-chain GPU$G ALL_DONE arms=$A1,$A2"
