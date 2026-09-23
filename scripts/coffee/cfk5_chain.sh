#!/bin/bash
# cfk5_chain.sh -- COFFEE-MG-p5 chain wrapper (2026-09-23).
# COPY of scripts/coffee/cfk4_chain.sh with these deltas ONLY:
#   * DATA_ROOT default data/2026_9_23_coffee_mg_p5/COFFEE-MG-p5-s$SEED
#     (fresh campaign; core + DP-base PRE-COPIED from p1 and dyn-base
#     RE-TRAINED at beta=1e-5 by cfk5_launch.sh -- chains start at
#     round 1 after launch's round0+BASE_ETA steps);
#   * WPROJ = COFFEE-MG-p5-s$SEED;
#   * round script = scripts/coffee/round_cfk5.sh (ATY arm carries the
#     per-round eta R-anchored + kappa v2 C-anchored recalibration; DP arm
#     guide=off);
#   * exports ETA0=5.6 (rcalib eta anchor), BETA=1.0e-5 (every dyn
#     training) and passes BASE_ETA through (REQUIRED -- the kappa C-anchor
#     on the base pair, measured by cfk5_launch.sh).
# Everything else identical: rounds 1..NROUNDS-1 full, round NROUNDS
# eval-pk; done_round idempotence via the shared round.log TOTAL lines.
#
# usage: SEED=<233|2333|23333> GPU=<id> ARM=<ATY|DP> NROUNDS=6 BASE_ETA=<float> \
#          bash scripts/coffee/cfk5_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333|23333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<ATY|DP>}
NROUNDS=${NROUNDS:-6}
export ETRIES=${ETRIES:-5}   # pass@5 (coffee campaign convention)
export ETA0=${ETA0:-5.6}     # rcalib eta anchor for round 1 (p1 grid verdict)
export BETA=${BETA:-1.0e-5}  # [p5] every dyn training (THM3 A3 recipe)
export BASE_ETA=${BASE_ETA:?set BASE_ETA=<base-pair calibrated eta (cfk5_launch.sh)>}
ROOT=/root/workspace/baojiachun/scout
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_23_coffee_mg_p5/COFFEE-MG-p5-s$SEED}
WPROJ=COFFEE-MG-p5-s$SEED
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/coffee/round.log   # shared across this seed's arms; TOTAL
                                 # lines mark completed (arm, round) pairs
[ -f "$RL" ] || { echo "[chain] FATAL: no round.log -- base not staged? run cfk5_launch.sh"; exit 1; }
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

for N in $(seq 1 $((NROUNDS-1))); do
  if done_round "$N"; then
    echo "[chain] round $N already COMPLETE (TOTAL in round.log) -- skip"
    continue
  fi
  echo "[chain] round $N START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
    bash scripts/coffee/round_cfk5.sh coffee "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/coffee/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
# eval-pk (口径定案): final round = full rescue rollout (SR + sharded explore
# xETRIES -> pass@ETRIES in merged json), retrains skipped inside round_cfk5.sh.
echo "[chain] round $NROUNDS (eval-pk) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
  bash scripts/coffee/round_cfk5.sh coffee "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
