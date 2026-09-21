#!/bin/bash
# thread_chain.sh -- THREADING-MG-p1 chain wrapper (2026-09-16).
# COPY of scripts/transport/tp15_chain.sh with these deltas ONLY:
#   * task = threading (MimicGen threading_d0), round script =
#     scripts/threading/round_threading.sh (.venv_mg inside);
#   * DATA_ROOT default data/2026_9_16_threading_p1/THREADING-MG-p1-s$SEED
#     (fresh campaign; round0 runs FRESH via the BASE branch -- core20 is
#     pre-materialized by thread_prep_data.sh so BASE only trains DP+dyn);
#   * WPROJ = THREADING-MG-p1-s$SEED;
#   * ETRIES default 5 (pass@5 campaign, user order 2026-09-16).
# Everything else identical to tp15: rounds 1..NROUNDS-1 full (eval + sharded
# rescue explore xETRIES + DP 300ep b256 + dyn 100ep), round NROUNDS eval-pk
# (SR + pass@ETRIES); done_round idempotence via the shared round.log TOTAL
# lines; final doses pinned by thread_start_arms.sh AFTER the grid.
#
# usage: SEED=<233|2333|23333> GPU=<id> ARM=<BASE|ATY|ORBIT|DP> NROUNDS=6 \
#          bash scripts/threading/thread_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333|23333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|ATY|ORBIT|DP>}
NROUNDS=${NROUNDS:-6}
export ETRIES=${ETRIES:-5}   # pass@5 campaign (user 2026-09-16); inherited by round_threading.sh
ROOT=/root/workspace/baojiachun/scout
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_16_threading_p1/THREADING-MG-p1-s$SEED}
WPROJ=${WPROJ:-THREADING-MG-p1-s$SEED}
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/threading/round.log   # shared across this seed's arms; TOTAL
                                    # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ CORE_N=20 \
    bash scripts/threading/round_threading.sh threading BASE 0
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
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
    bash scripts/threading/round_threading.sh threading "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/threading/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
# eval-pk (口径定案): final round = full rescue rollout (SR + sharded explore
# xETRIES -> pass@ETRIES in merged json), retrains skipped inside round_threading.sh.
echo "[chain] round $NROUNDS (eval-pk) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
  bash scripts/threading/round_threading.sh threading "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
