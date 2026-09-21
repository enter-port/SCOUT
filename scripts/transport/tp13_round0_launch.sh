#!/bin/bash
# tp13_round0_launch.sh -- TRANSPORT-9-13 round0-only orchestrator (2026-09-13).
# Spawns ONE tmux session running the chain BASE branch (real round0: seeded
# 20/200 split + base DP 600ep b64 + dyn-base) for a seed. ARMS ARE NOT
# STARTED HERE -- the guidance dose for transport is picked from the round0
# probe by the user; arms start via tp13_start_arms.sh afterwards.
# Run inside its own tmux (never via blocking ssh): this script itself spawns
# the tmux and returns.
#
# usage: bash scripts/transport/tp13_round0_launch.sh <SEED> <GPU> [<SSH_PORT>]
#   port defaults to 1022; s23333 runs on the port-1024 container (same shared
#   CPFS repo/data -- the ONLY difference is which container's GPUs/tmux).
set -u
SEED=${1:?usage: tp13_round0_launch.sh <SEED> <GPU>}
GPU=${2:?usage: tp13_round0_launch.sh <SEED> <GPU>}
ROOT=/root/workspace/baojiachun/scout
DR=$ROOT/data/2026_9_13_transport/TRANSPORT-s$SEED
S=tp13_r0_s$SEED

if tmux has-session -t "$S" 2>/dev/null; then
  echo "[launch] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
  exit 0
fi
tmux new-session -d -s "$S" \
  "cd $ROOT && SEED=$SEED GPU=$GPU ARM=BASE DATA_ROOT=$DR NROUNDS=6 CORE_N=20 bash scripts/transport/tp13_chain.sh"
echo "[launch] round0 spawned: $S (SEED=$SEED GPU=$GPU) $(date '+%F %T')"
echo "[launch] console: $DR/chain_BASE.console.log"
