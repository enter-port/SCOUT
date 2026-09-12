#!/bin/bash
# orb233x_chain.sh -- CAN-9-2-orbit-s2333 / CAN-9-2-orbit-s23333 chain
# wrapper (2026-09-08 user order: replicate CAN-9-2-orbit-s233 at seeds
# 2333 / 23333; everything identical except the seed). Structure copy of
# aty_chain.sh driving round_orbit_canpk.sh (= the CAN-9-2-orbit-s233
# round_orbit.sh + the new eval-pk final-round mode).
# Deltas vs the 9-2 reference chain:
#   * seed 2333 / 23333: TSEED controls the seeded 20-of-200 core split +
#     DP training.seed + dyn cfg.seed, so round0 is a FRESH base build
#     (split + base DP 600ep + dyn-base) -- exactly what the 9-2 chain
#     itself ran for s233 (no equivalent pre-built base exists for these
#     seeds);
#   * final round runs MODE=eval-pk (SR + pass@ETRIES=10 from the same
#     two-phase rescue rollout, no accum rebuild / no retrain / no backfill
#     into accum) per the 2026-09-07 user order "末轮 eval-only = SR +
#     pass@{ETRIES} 两样都要" (th95 had to patch pass@5 in after the fact
#     -- do not repeat). The r6 pass@10 uses the exp5 ckpt, the same ckpt
#     generation as the 9-2 chain's r5/r6 evals -> directly comparable.
#   * everything else verbatim 9-2: ETRIES=10 (pass@10), SHARD_P=2 x 25
#     envs, EVALNENV=25, DYN_FREEZE_AFTER=6 (dyn retrained every full
#     round 1..5), orbit v3 dose (eta_tilde 0.33 dimless / kappa 2.5 /
#     sigma 0.16 x 0.5^(r-1) / fb-clamp soft / noise-anneal 2 / lam 0.5 /
#     delta 0.25 -- defaults inside round_orbit_canpk.sh), XMODE=soe,
#     DP retrain 300ep / dyn 100ep, eval scenes 42..141, wandb one project
#     per seed (CAN-9-2-orbit-s<SEED>).
# ARM=ALL (used by the launch scripts): fresh round0 then the SCOUT arm,
# in one tmux session -- round0 here is a real ~3h build, so the launch
# must not block on it. ARM=BASE / ARM=SCOUT stay available for manual
# resume (done_round idempotency via the round.log TOTAL lines).
#
# usage: SEED=2333 GPU=1 ARM=ALL bash orb233x_chain.sh    # round0 then SCOUT r1..r6
#   (or: SEED=2333 GPU=1 ARM=BASE bash orb233x_chain.sh  -> round 0 only)
#   (or: SEED=2333 GPU=1 ARM=SCOUT bash orb233x_chain.sh -> rounds 1..6)
set -uo pipefail
SEED=${SEED:?set SEED=<int>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<ALL|BASE|SCOUT>}
NROUNDS=${NROUNDS:-6}   # total rounds incl. the final eval-pk round
ROOT=/root/workspace/baojiachun/scout-orbit
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_2_orbchain/ORBIT-s$SEED}
WPROJ=${WPROJ:-CAN-9-2-orbit-s$SEED}
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/can/round.log   # shared across this seed's arms; TOTAL
                              # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = ALL ]; then
  echo "[chain] round0 START $(date '+%F %T') (fresh base build: split + DP 600ep + dyn-base)"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
    bash soe_scripts/round_orbit_canpk.sh can BASE 0
  rc=$?
  echo "[chain] round0 rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] round0 FAILED -- SCOUT arm NOT started"; exit $rc; }
  ARM=SCOUT   # fall through into the full-round loop below
fi

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
    bash soe_scripts/round_orbit_canpk.sh can BASE 0
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
    ETRIES=10 SHARD_P=2 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 \
    bash soe_scripts/round_orbit_canpk.sh can "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/can/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
echo "[chain] round $NROUNDS (eval-pk: SR + pass@10) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ \
  ETRIES=10 SHARD_P=2 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 \
  bash soe_scripts/round_orbit_canpk.sh can "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
