#!/bin/bash
# orb2333_launch.sh -- CAN-9-2-orbit-s2333 launch orchestrator (2026-09-08
# user order). Replicates CAN-9-2-orbit-s233 (single SCOUT arm, can, 6
# rounds, orbit v3 dose, pass@10 rescue protocol) at seed 2333; the ONLY
# protocol delta is the final round = eval-pk (SR + pass@10) per the
# 2026-09-07 user order. Round0 is a FRESH base build (no pre-built s2333
# base exists; the 9-2 chain likewise built its own): seeded 20-of-200
# split + base DP 600ep + dyn-base, ~3h -- so the launcher only spawns the
# tmux session (ARM=ALL chain: round0 then SCOUT r1..r6) and returns.
# GPU1 assigned (idle, ECC clean; th95 arms on 0/2/4, SOE chain on 6,
# GPU7 defective). Fresh DATA_ROOT + own wandb project: touches no
# running chain.
set -u
ROOT=/root/workspace/baojiachun/scout-orbit
SEED=2333
GPU=1
DATA_ROOT=$ROOT/data/2026_9_2_orbchain/ORBIT-s$SEED
WPROJ=CAN-9-2-orbit-s$SEED
cd "$ROOT" || exit 1
# NOTE: no `unset DATA_ROOT` here (unlike can_aty_launch.sh): this launcher
# SETS DATA_ROOT itself (below) and expands it into the tmux string, and the
# script runs with set -u -- unsetting would abort with an unbound variable.

if tmux has-session -t orb2333_scout 2>/dev/null; then
  echo "[launch] tmux session orb2333_scout ALREADY EXISTS -- NOT starting"
  exit 1
fi

tmux new-session -d -s orb2333_scout \
  "cd $ROOT && SEED=$SEED GPU=$GPU ARM=ALL DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ NROUNDS=6 ETRIES=10 SHARD_P=2 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 bash soe_scripts/orb233x_chain.sh"
echo "[launch] spawned tmux orb2333_scout @GPU$GPU (round0 fresh -> SCOUT r1..r6, r6=eval-pk) $(date '+%F %T')"
echo "[launch] console: $DATA_ROOT/chain_ALL.console.log"
