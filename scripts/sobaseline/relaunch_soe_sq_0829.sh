#!/bin/bash
# relaunch_soe_sq_0829.sh -- restart the 3 SOE square chains with the corrected
# project name SQUARE-8-29-SOE-s{seed} (user order 08-29; previous SQUARE-29-SOE
# launch stopped ~45min into round0, partials + stale wandb BASE runs deleted).
set -eu
cd /root/workspace/baojiachun
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
mkdir -p soe_data/2026_8_29_soe/logs
tmux new -d -s soe_sq233   "cd /root/workspace/baojiachun && GPU=2 TSEED=233   DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s233   WPROJ=SQUARE-8-29-SOE-s233   DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500 VISGATE=0 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq233_chain.console.log"
tmux new -d -s soe_sq2333  "cd /root/workspace/baojiachun && GPU=3 TSEED=2333  DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s2333  WPROJ=SQUARE-8-29-SOE-s2333  DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500 VISGATE=0 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq2333_chain.console.log"
tmux new -d -s soe_sq23333 "cd /root/workspace/baojiachun && GPU=4 TSEED=23333 DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s23333 WPROJ=SQUARE-8-29-SOE-s23333 DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500 VISGATE=0 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq23333_chain.console.log"
sleep 3
tmux ls | grep soe_sq
echo "[relaunch] 3 SOE square chains started with WPROJ=SQUARE-8-29-SOE-s{seed}"
