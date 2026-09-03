#!/bin/bash
# th_chain.sh -- TOOLHANG-9-1 orbit campaign chain wrapper (2026-09-01).
# Rounds 1..NROUNDS-1 full (eval + sharded rescue explore + DP 300ep +
# dyn 100ep), round $NROUNDS eval-only. Default NROUNDS=8 (original 09-01
# spec); the 2026-09-03 restart passes NROUNDS=6. Both arms of a seed share
# ONE wandb project (TOOLHANG-9-1-orbit-s<seed>) and ONE DATA_ROOT (round0
# assets live there).
#
# usage: SEED=233 GPU=1 ARM=SCOUT NROUNDS=6 bash th_chain.sh
#   (or: SEED=233 GPU=1 ARM=BASE bash th_chain.sh  -> round 0 only)
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|SCOUT|DP>}
NROUNDS=${NROUNDS:-8}   # total rounds incl. the final eval-only round
ROOT=/root/workspace/baojiachun/scout-orbit
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_1_toolhang/TOOLHANG-s$SEED}
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

for N in $(seq 1 $((NROUNDS-1))); do
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
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-only) already COMPLETE -- skip"
else
echo "[chain] round $NROUNDS (eval-only) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
  bash soe_scripts/round_orbit_th.sh tool_hang "$ARM" "$NROUNDS" eval-only
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
