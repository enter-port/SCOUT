#!/bin/bash
# wait_dp_entropy.sh -- wait for the seed's round-0 TOTAL line in round.log,
# then start the DP-arm chain of the FORMAL entropy experiment (its own
# BASE 0 pass skips instantly; no race with the SCOUT arm that owns round 0).
# Usage: wait_dp_entropy.sh <task> <seed> <gpu> <data_root> <wproj>
set -u
TASK=${1:?}; SEED=${2:?}; GPU=${3:?}; DATA_ROOT=${4:?}; WPROJ=${5:?}
LOG=$DATA_ROOT/$TASK/round.log
REPO=/root/workspace/baojiachun/scout-entropy
cd "$REPO" || exit 1
echo "[wait] waiting for ROUND0 $TASK seed=$SEED TOTAL in $LOG"
while ! grep -q "ROUND0 $TASK seed=$SEED TOTAL" "$LOG" 2>/dev/null; do
  sleep 300
done
echo "[wait] round0 seed=$SEED done -- launching DP chain (GPU$GPU)"
exec env GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
  bash soe_scripts/chain_entropy.sh "$TASK" DP
