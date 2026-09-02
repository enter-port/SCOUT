#!/bin/bash
# th_chain.sh -- TOOLHANG-9-1 orbit campaign chain wrapper (2026-09-01).
# Rounds 1-7 full (eval + sharded rescue explore + DP 300ep + dyn 100ep),
# round 8 eval-only. Both arms of a seed share ONE wandb project
# (TOOLHANG-9-1-orbit-s<seed>) and ONE DATA_ROOT (round0 assets live there).
#
# usage: SEED=233 GPU=1 ARM=SCOUT bash th_chain.sh
#   (or: SEED=233 GPU=1 ARM=BASE bash th_chain.sh  -> round 0 only)
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|SCOUT|DP>}
ROOT=/root/workspace/baojiachun/scout-orbit
DATA_ROOT=$ROOT/data/2026_9_1_toolhang/TOOLHANG-s$SEED
WPROJ=TOOLHANG-9-1-orbit-s$SEED
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/tool_hang/round.log   # shared across this seed's arms; TOTAL
                                    # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
    bash soe_scripts/round_orbit_th.sh tool_hang BASE 0
  rc=$?
  echo "[chain] round0 rc=$rc $(date '+%F %T')"
  exit $rc
fi

for N in 1 2 3 4 5 6 7; do
  if done_round "$N"; then
    echo "[chain] round $N already COMPLETE (TOTAL in round.log) -- skip"
    continue
  fi
  echo "[chain] round $N START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
    bash soe_scripts/round_orbit_th.sh tool_hang "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/tool_hang/round.log)"; exit $rc; }
done
N=8
if done_round "$N"; then
  echo "[chain] round 8 (eval-only) already COMPLETE -- skip"
else
echo "[chain] round 8 (eval-only) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
  bash soe_scripts/round_orbit_th.sh tool_hang "$ARM" 8 eval-only
rc=$?
echo "[chain] round 8 rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL 8 ROUNDS DONE $(date '+%F %T')"
