#!/bin/bash
# thm3_chain.sh -- THREADING-THM3 chain wrapper (2026-09-23).
# COPY of scripts/threading/thm2_chain.sh with these deltas ONLY:
#   * DATA_ROOT default data/2026_9_23_threading_thm3/THREADING-THM3-s$SEED
#     (base three-piece STAGED by thm3_launch.sh from the p2 campaign -- the
#     p2 dyn-base is already beta=1e-5/300ep = the A3 recipe, so it is
#     COPIED, not retrained; no round0 on this campaign);
#   * WPROJ = THREADING-THM3-s$SEED;
#   * round script = scripts/threading/round_thm3.sh (stop-on-first-success
#     explore + DP 600ep b64 + dyn 300ep + per-round eta rcalib AND kappa v2
#     C-anchored recalibration inside);
#   * BASE_ETA forwarded (per-seed p2 rcalib anchor for the kappa C_target).
# Everything else identical: rounds 1..NROUNDS-1 full, round NROUNDS
# eval-pk; done_round idempotence via the shared round.log TOTAL lines.
#
# usage: SEED=<233|2333|23333> GPU=<id> ARM=<BASE|ATY|DP> NROUNDS=6 \
#          bash scripts/threading/thm2_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333|23333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|ATY|DP>}
NROUNDS=${NROUNDS:-6}
export ETRIES=${ETRIES:-5}   # pass@5 campaign (threading convention)
export ETA0=${ETA0:-1.0}     # rcalib eta anchor for round 1
export ATT_CAP=${ATT_CAP:-2.5}   # kappa anchor (carried over unchanged)
export BETA=${BETA:-1.0e-5}  # VIB beta for every dyn training
export CALIB_MODE=${CALIB_MODE:-rc}
case "$CALIB_MODE" in rc|pr) ;; *) echo "CALIB_MODE must be rc or pr"; exit 1 ;; esac
export CALIB_P=${CALIB_P:-6} CALIB_R=${CALIB_R:-0.01}
export CALIB_BAND=${CALIB_BAND:-0.1} CALIB_KAPPA0=${CALIB_KAPPA0:-2.5}
if [ "$ARM" = ATY ] && [ "$CALIB_MODE" = rc ]; then
  export BASE_ETA=${BASE_ETA:?set BASE_ETA for CALIB_MODE=rc}
else
  export BASE_ETA=${BASE_ETA:-}
fi
ROOT=${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_23_threading_thm3/THREADING-THM3-s$SEED}
WPROJ=${WPROJ:-THREADING-THM3-s$SEED}
CONSOLE=$DATA_ROOT/chain_${ARM}.console.log
mkdir -p "$DATA_ROOT"
exec >> "$CONSOLE" 2>&1

cd "$ROOT" || exit 1
RL=$DATA_ROOT/threading/round.log   # shared across this seed's arms; TOTAL
                                    # lines mark completed (arm, round) pairs
done_round(){ grep -q "a=$ARM seed=$SEED round=$1 TOTAL" "$RL" 2>/dev/null; }

if [ "$ARM" = BASE ]; then
  echo "[chain] round0 START $(date '+%F %T')"
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ CORE_N=20 BETA=$BETA \
    bash scripts/threading/round_thm3.sh threading BASE 0
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
    ETA0=$ETA0 ATT_CAP=$ATT_CAP BETA=$BETA BASE_ETA=$BASE_ETA \
    bash scripts/threading/round_thm3.sh threading "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/threading/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
# eval-pk (口径定案): final round = full rescue rollout (SR + sharded explore
# xETRIES -> pass@ETRIES in merged json), retrains skipped inside round_thm2.sh.
echo "[chain] round $NROUNDS (eval-pk) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM CORE_N=20 \
  ETA0=$ETA0 ATT_CAP=$ATT_CAP BETA=$BETA BASE_ETA=$BASE_ETA \
  bash scripts/threading/round_thm3.sh threading "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
