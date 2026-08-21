#!/bin/bash
# wait_launch_dp.sh -- wait for the seed's round-0 TOTAL line in round.log,
# then start the DP-arm chain (its own BASE 0 pass skips instantly; no race
# with the SCOUT arm that owns round 0).
# Usage: wait_launch_dp.sh <task> <seed> <gpu> <data_root>
set -u
TASK=${1:?}; SEED=${2:?}; GPU=${3:?}; DATA_ROOT=${4:?}
LOG=$DATA_ROOT/$TASK/round.log
REPO=/root/workspace/baojiachun/scout
cd "$REPO" || exit 1
echo "[wait] waiting for ROUND0 seed=$SEED TOTAL in $LOG"
while ! grep -q "ROUND0 $TASK seed=$SEED TOTAL" "$LOG" 2>/dev/null; do
  sleep 300
done
echo "[wait] round0 seed=$SEED done -- launching DP chain (GPU$GPU)"
exec env GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT bash soe_scripts/chain233.sh "$TASK" DP
