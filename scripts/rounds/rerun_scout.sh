#!/bin/bash
# rerun_scout.sh -- re-run a contiguous range of SCOUT-arm rounds after a
# data wipe (round.sh's walk-back picks the right DP/dyn predecessors
# automatically). Usage: rerun_scout.sh <task> <seed> <gpu> <first> <last> <data_root>
set -u
TASK=${1:?}; SEED=${2:?}; GPU=${3:?}; FIRST=${4:?}; LAST=${5:?}; DATA_ROOT=${6:?}
REPO=/root/workspace/baojiachun/scout
cd "$REPO" || exit 1
for n in $(seq "$FIRST" "$LAST"); do
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT bash soe_scripts/round.sh "$TASK" SCOUT "$n" || exit 1
done
echo "[rerun] $TASK SCOUT seed=$SEED rounds $FIRST..$LAST ALL DONE $(date '+%F %T')"
