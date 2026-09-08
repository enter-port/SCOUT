#!/bin/bash
# th95_chain.sh -- TOOLHANG-9-5-orbit-s233 chain wrapper (2026-09-05).
# COPY of th94_chain.sh with these deltas:
#   * ROOT = scout-th95 (plain extract of orbit-dev@cfca66e);
#   * THREE arms: DP(guide off) / ATY(atypical raw s0.5/k2.5/gst50) /
#     ORBIT(orbit raw s0.5 + lam.5/delta.25/sig.05 x0.5^(r-1)/anneal2/fb-soft);
#   * DATA_ROOT default data/2026_9_5_toolhang/TOOLHANG-s$SEED (base assets
#     pre-copied from TOOLHANG-9-4 root: 40-demo core + DP-base 599.ckpt +
#     dyn-base 20260904-170403);
#   * WPROJ = TOOLHANG-9-5-orbit-s$SEED; run names DP-round{i}/ATY-round{i}/
#     ORBIT-round{i} via WNAME_BASE=$ARM;
#   * round script = round_th95.sh; ETRIES=5 (pass@5), SHARD_P=8/arm.
# Rounds 1..NROUNDS-1 full (eval + sharded rescue explore + DP 300ep +
# dyn 100ep), round $NROUNDS eval-pk (2026-09-08 口径定案: full rescue
# rollout, SR + pass@ETRIES, retrains skipped). Arms share ONE wandb project
# and ONE DATA_ROOT (each arm accumulates under its own rollout/<ARM>-exp<N>
# dirs).
#
# usage: SEED=233 GPU=<id> ARM=<BASE|ATY|ORBIT|DP> NROUNDS=6 \
#          bash scripts/toolhang/th95_chain.sh
set -uo pipefail
SEED=${SEED:?set SEED=<233>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|ATY|ORBIT|DP>}
NROUNDS=${NROUNDS:-6}
ROOT=/root/workspace/baojiachun/scout-th95
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_5_toolhang/TOOLHANG-s$SEED}
WPROJ=TOOLHANG-9-5-orbit-s$SEED
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
    bash scripts/toolhang/round_th95.sh tool_hang BASE 0
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
  GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM \
    bash scripts/toolhang/round_th95.sh tool_hang "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/tool_hang/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-pk) already COMPLETE -- skip"
else
# eval-pk (user order 2026-09-08, 口径定案): final round = full rescue
# rollout (SR + sharded explore xETRIES -> pass@ETRIES in merged json),
# retrains skipped inside the round script.
echo "[chain] round $NROUNDS (eval-pk) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM \
  bash scripts/toolhang/round_th95.sh tool_hang "$ARM" "$NROUNDS" eval-pk
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
