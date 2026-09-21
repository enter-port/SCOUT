#!/bin/bash
# tp15_chain.sh -- TRANSPORT-9-15-p1 (pass@1) chain wrapper (2026-09-15).
# COPY of scripts/transport/tp13_chain.sh with these deltas ONLY:
#   * DATA_ROOT default data/2026_9_15_transport_p1/TRANSPORT-s$SEED -- base
#     three-piece COPIED from 2026_9_13_transport by tp15_prep_base.sh (no
#     fresh round0; the ROUND0 TOTAL gate line is written by the prep);
#   * WPROJ = TRANSPORT-9-15-p1-s$SEED (new wandb project, old one untouched);
#   * ETRIES pinned to 1 (pass@1 protocol, user order 2026-09-15;
#     round_tp.sh default is 5).
# Everything else identical to tp13: rounds 1..NROUNDS-1 full (eval + sharded
# rescue explore xETRIES + DP 300ep b256 + dyn 100ep), round NROUNDS eval-pk;
# final doses still pinned by tp15_start_arms.sh AFTER prep.
#
# usage: SEED=<233|2333|23333> GPU=<id> ARM=<ATY|ORBIT|DP> NROUNDS=6 \
#          bash scripts/transport/tp15_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333|23333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|ATY|ORBIT|DP>}
NROUNDS=${NROUNDS:-6}
export ETRIES=${ETRIES:-1}   # pass@1 campaign (tp15); inherited by round_tp.sh
ROOT=/root/workspace/baojiachun/scout
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_15_transport_p1/TRANSPORT-s$SEED}
WPROJ=TRANSPORT-9-15-p1-s$SEED
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/transport/round.log   # shared across this seed's arms; TOTAL
                                    # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ CORE_N=20 \
    bash scripts/transport/round_tp.sh transport BASE 0
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
    bash scripts/transport/round_tp.sh transport "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/transport/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
# eval-pk (口径定案): final round = full rescue rollout (SR + sharded explore
# xETRIES -> pass@ETRIES in merged json), retrains skipped inside round_tp.sh.
echo "[chain] round $NROUNDS (eval-pk) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
  bash scripts/transport/round_tp.sh transport "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
