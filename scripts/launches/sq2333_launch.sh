#!/bin/bash
# sq2333_launch.sh -- SQUARE-9-1-orbit-s2333 launch orchestrator (2026-09-09
# user order). Replicates SQUARE-9-1-orbit-s233 (single SCOUT arm, square,
# 6 rounds, orbit v3 dose, pass@10 rescue protocol) at seed 2333; deltas =
# DP training batch 256 (this campaign only) + final round eval-pk (09-07
# user order). Round0 is a FRESH base build (no pre-built s2333 square base
# exists): seeded 20-of-200 split + base DP 600ep (batch 256) + dyn-base --
# so the launcher only spawns the tmux session (ARM=ALL chain: round0 then
# SCOUT r1..r6) and returns. GPU1 assigned (the can s2333 chain's card,
# free after that chain finishes -- the sq233x_wait_launch.sh watcher gates
# on exactly that). Fresh DATA_ROOT + own wandb project.
set -u
ROOT=/root/workspace/baojiachun/scout-orbit
SEED=2333
GPU=1
DATA_ROOT=$ROOT/data/2026_9_1_orbchain/ORBIT-s$SEED
WPROJ=SQUARE-9-1-orbit-s$SEED
cd "$ROOT" || exit 1
# NOTE: no `unset DATA_ROOT` here: this launcher SETS DATA_ROOT itself and
# expands it into the tmux string, and the script runs with set -u --
# unsetting would abort with an unbound variable (can-replicate review P0).

if tmux has-session -t sq2333_scout 2>/dev/null; then
  echo "[launch] tmux session sq2333_scout ALREADY EXISTS -- NOT starting"
  exit 1
fi

tmux new-session -d -s sq2333_scout \
  "cd $ROOT && SEED=$SEED GPU=$GPU ARM=ALL DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ NROUNDS=6 ETRIES=10 SHARD_P=4 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 bash soe_scripts/orb_sq233x_chain.sh"
echo "[launch] spawned tmux sq2333_scout @GPU$GPU (round0 fresh batch256 -> SCOUT r1..r6, r6=eval-pk) $(date '+%F %T')"
echo "[launch] console: $DATA_ROOT/chain_ALL.console.log"
