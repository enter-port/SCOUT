#!/bin/bash
# aty_chain.sh -- CAN-aty-try1-s233 chain wrapper (2026-09-03 user order).
# Structure copy of th_chain.sh driving round_aty.sh (the CAN-9-2-orbit-s233
# round_orbit.sh variant whose SCOUT arm runs the orbit PHASE-1 CLIMB ONLY,
# i.e. the atypical cost -- lam=0/delta=0/sigma=0, dose eta_tilde=0.33
# dimless + kappa=2.5 kept verbatim from the 9-2 phase-1). Three user deltas
# vs CAN-9-2-orbit-s233: phase-1-only guidance, ETRIES=1 (pass@10 -> pass@1:
# each failed eval init tried ONCE), SHARD_P=1 (single explore worker).
# Both the round0 assets and the wandb project belong to this campaign alone
# (fresh DATA_ROOT; round0 is pre-seeded by can_aty_launch.sh from the
# CAN-9-2 root -- same seed 233, deterministic round0 => byte-equivalent).
#
# usage: SEED=233 GPU=0 ARM=SCOUT bash aty_chain.sh
#   (or: SEED=233 GPU=0 ARM=BASE bash aty_chain.sh  -> round 0 only)
set -uo pipefail
SEED=${SEED:?set SEED=<int>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|SCOUT>}
NROUNDS=${NROUNDS:-6}   # total rounds incl. the final eval-only round
ROOT=/root/workspace/baojiachun/scout-orbit
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_3_atytry1/ATY-s$SEED}
WPROJ=${WPROJ:-CAN-aty-try1-s$SEED}
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/can/round.log   # shared across this seed's arms; TOTAL
                              # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
    bash soe_scripts/round_aty.sh can BASE 0
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
    ETRIES=1 SHARD_P=1 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 \
    bash soe_scripts/round_aty.sh can "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/can/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-only) already COMPLETE -- skip"
else
echo "[chain] round $NROUNDS (eval-only) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
  ETRIES=1 SHARD_P=1 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 \
  bash soe_scripts/round_aty.sh can "$ARM" "$NROUNDS" eval-only
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
