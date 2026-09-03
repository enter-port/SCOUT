#!/bin/bash
# chain233.sh -- one full seeded chain (user 2026-08-21):
#   round 0 (seeded 20/200 split + base DP + dyn-base; idempotent, SHARED by
#   the seed's SCOUT and DP arms) then rounds 1..NROUNDS serially via round.sh.
# Usage:  chain233.sh <task> <SCOUT|DP>
# Env:    GPU TSEED DATA_ROOT (required; passed through to round.sh)
#         NROUNDS=6 DYN_FREEZE_AFTER=3 NEXPLORE=100
# A DP-arm chain must not race the SCOUT arm on round 0: start the SCOUT chain
# first (it OWNS round 0); launch the DP chain after round0's TOTAL line
# appears in $DATA_ROOT/<task>/round.log (its BASE 0 pass then skips instantly).
set -u
TASK=${1:?usage: chain233.sh <task> <SCOUT|DP>}
A=${2:?usage: chain233.sh <task> <SCOUT|DP>}
GPU=${GPU:?set GPU=<id>} TSEED=${TSEED:?set TSEED} DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
NROUNDS=${NROUNDS:-6}
REPO=/root/workspace/baojiachun/scout
cd "$REPO" || exit 1

bash soe_scripts/round.sh "$TASK" BASE 0 || { echo "[chain] round0 FAILED"; exit 1; }
for n in $(seq 1 "$NROUNDS"); do
  bash soe_scripts/round.sh "$TASK" "$A" "$n" || { echo "[chain] round $n FAILED"; exit 1; }
done
echo "[chain] $TASK a=$A seed=$TSEED ALL DONE $(date '+%F %T')"
