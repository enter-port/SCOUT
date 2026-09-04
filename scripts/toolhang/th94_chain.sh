#!/bin/bash
# th94_chain.sh -- TOOLHANG-9-4-orbit-s233 chain wrapper (2026-09-04).
# Rounds 1..NROUNDS-1 full (eval + sharded rescue explore + DP 300ep +
# dyn 100ep), round $NROUNDS eval-only. Arms share ONE wandb project
# (TOOLHANG-9-4-orbit-s233; run names DP-round{i} / SCOUT-round{i} via
# WNAME_BASE) and ONE DATA_ROOT (the TH9-4 round0 assets live there:
# 40-demo core + DP-base 599.ckpt + dyn-base 20260904-170403).
# Runs in the scout-th94 worktree (orbit-dev: refactor + TrajSpool OOM fix);
# SCOUT arm = atypical raw dose s1.0/cap2.5/gst50 (config FINAL 1.0/50).
#
# usage: SEED=233 GPU=1 ARM=SCOUT NROUNDS=6 bash scripts/toolhang/th94_chain.sh
#   (or: SEED=233 GPU=1 ARM=BASE bash ...  -> round 0 only, idempotent-skip)
set -uo pipefail
SEED=${SEED:?set SEED=<233>}
GPU=${GPU:?set GPU=<cuda id>}
ARM=${ARM:?set ARM=<BASE|SCOUT|DP>}
NROUNDS=${NROUNDS:-6}
ROOT=/root/workspace/baojiachun/scout-th94
DATA_ROOT=${DATA_ROOT:-$ROOT/data/2026_9_4_toolhang/TOOLHANG-s$SEED}
WPROJ=TOOLHANG-9-4-orbit-s$SEED
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
    bash scripts/toolhang/round_th94.sh tool_hang BASE 0
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
    bash scripts/toolhang/round_th94.sh tool_hang "$ARM" "$N" full
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $DATA_ROOT/tool_hang/round.log)"; exit $rc; }
done
N=$NROUNDS
if done_round "$N"; then
  echo "[chain] round $NROUNDS (eval-only) already COMPLETE -- skip"
else
echo "[chain] round $NROUNDS (eval-only) START $(date '+%F %T')"
GPU=$GPU TSEED=$SEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ WNAME_BASE=$ARM \
  bash scripts/toolhang/round_th94.sh tool_hang "$ARM" "$NROUNDS" eval-only
rc=$?
echo "[chain] round $NROUNDS rc=$rc $(date '+%F %T')"
[ $rc -ne 0 ] && exit $rc
fi
echo "[chain] ALL $NROUNDS ROUNDS DONE $(date '+%F %T')"
