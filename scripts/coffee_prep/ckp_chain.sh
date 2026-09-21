#!/bin/bash
# ckp_chain.sh -- COFFEEPREP-MG-p1 chain wrapper (2026-09-20).
# COPY of scripts/threading/thread_chain.sh with these deltas ONLY:
#   * task = coffee_prep (MimicGen coffee_preparation_d0), round script =
#     scripts/coffee_prep/round_ckp.sh (.venv_mg inside);
#   * DATA_ROOT default data/2026_9_19_coffeeprep_mg_p1/COFFEEPREP-MG-p1-s$SEED
#     (fresh campaign; round0 runs FRESH via the BASE branch -- core20 is
#     pre-materialized by ckp_prep_data.sh so BASE only trains DP+dyn);
#   * WPROJ = COFFEEPREP-MG-p1-s$SEED.
# Everything else identical to threading: rounds 1..NROUNDS-1 full (eval +
# sharded rescue explore xETRIES + DP 300ep b256 + dyn 100ep), round NROUNDS
# eval-pk (SR + pass@ETRIES); done_round idempotence via the shared round.log
# TOTAL lines; final doses pinned by ckp_start_arms.sh AFTER the grid.
#
# usage: SEED=<233|2333|23333> GPU=<id> ARM=<BASE|ATY|ORBIT|DP> NROUNDS=6 \
#          bash scripts/coffee_prep/ckp_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333|23333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|ATY|ORBIT|DP>}
NROUNDS=${NROUNDS:-6}
export ETRIES=${ETRIES:-5}   # pass@5 campaign (threading precedent); inherited by round_ckp.sh
ROOT=/root/workspace/baojiachun/scout
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_19_coffeeprep_mg_p1/COFFEEPREP-MG-p1-s$SEED}
WPROJ=COFFEEPREP-MG-p1-s$SEED
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/coffee_prep/round.log   # shared across this seed's arms; TOTAL
                                      # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ CORE_N=20 \
    bash scripts/coffee_prep/round_ckp.sh coffee_prep BASE 0
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
    bash scripts/coffee_prep/round_ckp.sh coffee_prep "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/coffee_prep/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
# eval-pk (口径定案): final round = full rescue rollout (SR + sharded explore
# xETRIES -> pass@ETRIES in merged json), retrains skipped inside round_ckp.sh.
echo "[chain] round $NROUNDS (eval-pk) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
  bash scripts/coffee_prep/round_ckp.sh coffee_prep "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
