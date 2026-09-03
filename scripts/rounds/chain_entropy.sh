#!/bin/bash
# chain_entropy.sh -- FORMAL entropy experiment chain (user 2026-08-24):
#   round 0 (seeded 20/200 split + base DP 600ep + dyn-base; idempotent,
#   SHARED by the seed's SCOUT and DP arms) then rounds 1..NROUNDS serially
#   via round_entropy.sh (SCOUT explores with the entropy cost, --guide
#   atypical; DP arm unguided).
# Usage:  chain_entropy.sh <task> <SCOUT|DP>
# Env:    GPU TSEED DATA_ROOT (required; passed through to round_entropy.sh)
#         WPROJ=<wandb project, e.g. CAN-8-24-entropy-s233> (required --
#         one project per seed, both arms inside it)
#         NROUNDS=6 ATT_CAP=2.5 DYN_FREEZE_AFTER=6
# The DP-arm chain must not race the SCOUT arm on round 0: start the SCOUT
# chain first (it OWNS round 0); launch the DP chain via wait_dp_entropy.sh
# after round0's TOTAL line appears in $DATA_ROOT/<task>/round.log (its own
# BASE 0 pass then skips instantly).
set -u
TASK=${1:?usage: chain_entropy.sh <task> <SCOUT|DP>}
A=${2:?usage: chain_entropy.sh <task> <SCOUT|DP>}
GPU=${GPU:?set GPU=<id>} TSEED=${TSEED:?set TSEED} DATA_ROOT=${DATA_ROOT:?set DATA_ROOT}
WPROJ=${WPROJ:?set WPROJ=CAN-8-24-entropy-s<seed>}
NROUNDS=${NROUNDS:-6}
REPO=/root/workspace/baojiachun/scout-entropy
cd "$REPO" || exit 1

bash soe_scripts/round_entropy.sh "$TASK" BASE 0 || { echo "[chain] round0 FAILED"; exit 1; }
for n in $(seq 1 "$NROUNDS"); do
  bash soe_scripts/round_entropy.sh "$TASK" "$A" "$n" || { echo "[chain] round $n FAILED"; exit 1; }
done
echo "[chain] $TASK a=$A seed=$TSEED ALL DONE $(date '+%F %T')"
