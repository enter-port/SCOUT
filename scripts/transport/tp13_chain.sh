#!/bin/bash
# tp13_chain.sh -- TRANSPORT-9-13-orbit chain wrapper (2026-09-13).
# COPY of scripts/toolhang/th95_chain.sh with these deltas:
#   * ROOT = /root/workspace/baojiachun/scout (main repo, post 09-12
#     consolidation; shared CPFS, visible from port-1022 AND port-1024);
#   * task = transport; round script = scripts/transport/round_tp.sh;
#   * DATA_ROOT default data/2026_9_13_transport/TRANSPORT-s$SEED (fresh
#     round0 per seed: seeded 20/200 split + DP 600ep b64 + dyn-base);
#   * WPROJ = TRANSPORT-9-13-orbit-s$SEED; run names DP-round{i}/ATY-round{i}/
#     ORBIT-round{i} via WNAME_BASE=$ARM;
#   * CORE_N=20 exported into every round call (round0 split size).
# THREE arms: DP(guide off) / ATY(atypical raw + k2.5 + gst50) / ORBIT(orbit
# same raw dose + lam.5/delta.25/sigma-decay0.5/anneal2/fb-soft) -- FINAL
# DOSES are pinned by tp13_start_arms.sh AFTER the round0 dose probe (user
# decision point), NOT here.
# Rounds 1..NROUNDS-1 full (eval + sharded rescue explore xETRIES + DP 300ep
# b256 + dyn 100ep), round $NROUNDS eval-pk (SR + pass@ETRIES, retrains
# skipped). Arms share ONE wandb project and ONE DATA_ROOT (each arm
# accumulates under its own rollout/<ARM>-exp<N> dirs).
#
# usage: SEED=<233|2333|23333> GPU=<id> ARM=<BASE|ATY|ORBIT|DP> NROUNDS=6 \
#          bash scripts/transport/tp13_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233|2333|23333>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|ATY|ORBIT|DP>}
NROUNDS=${NROUNDS:-6}
ROOT=/root/workspace/baojiachun/scout
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_13_transport/TRANSPORT-s$SEED}
WPROJ=TRANSPORT-9-13-orbit-s$SEED
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
