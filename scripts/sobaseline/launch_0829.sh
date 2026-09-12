#!/bin/bash
# launch_0829.sh -- 2026-08-29 campaign launch (user order):
#   1. SOE square  seeds 233/2333/23333  -> WPROJ SQUARE-8-29-SOE-s{seed} (renamed by user order)
#   2. SOE can    seed  23333            -> WPROJ CAN-8-24-SOE-s23333 (aligned with the 8-24 can campaign)
#   3. SCOUT square s23333 SCOUT+DP arms -> WPROJ SQUARE-8-26-entropy-s23333 (aligned with the 8-26 square campaign;
#      user wrote "s2333" -- corrected to s23333, "SQUARE-8-26-entropy-s2333" is already occupied by the finished seed-2333 project)
# GPU map: 0=sq23333_SCOUT  1=sq23333_waitDP(DP arm)  2=SOE-sq233  3=SOE-sq2333  4=SOE-sq23333  5=SOE-can-s23333
# (GPU7 forbidden: 6 uncorrected ECC errors)
# Disk hygiene (v2): SOE SAVE0/SAVE=200 (ckpt archive 4x smaller; policy_last + full eval/metrics identical)
#   + PRUNE_ACCUM=1 (superseded demo_plus_core removed after next round consumed it; raw demo.hdf5 kept)
#   SCOUT POST_PRUNE=1 (round's success_accum/all_accum removed after its retrain; sources success.hdf5/all.hdf5 kept).
set -eu
cd /root/workspace/baojiachun
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
mkdir -p soe_data/2026_8_29_soe/logs scout-entropy/data/2026_8_26_entropy/logs

echo "[launch] SCOUT square s23333 (SCOUT arm owns round0; DP arm waits for ROUND0 TOTAL)"
tmux new -d -s sq23333_SCOUT "cd /root/workspace/baojiachun/scout-entropy && POST_PRUNE=1 GPU=0 TSEED=23333 DATA_ROOT=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s23333 WPROJ=SQUARE-8-26-entropy-s23333 bash soe_scripts/chain_entropy.sh square SCOUT 2>&1 | tee -a /root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/logs/sq23333_SCOUT.console.log"
tmux new -d -s sq23333_waitDP "cd /root/workspace/baojiachun/scout-entropy && POST_PRUNE=1 bash soe_scripts/wait_dp_entropy.sh square 23333 1 /root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s23333 SQUARE-8-26-entropy-s23333 2>&1 | tee -a /root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/logs/sq23333_waitDP.console.log"

echo "[launch] SOE square s233/s2333/s23333 (HORIZON=500 VISGATE=0 DATASETS=square)"
tmux new -d -s soe_sq233   "cd /root/workspace/baojiachun && GPU=2 TSEED=233   DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s233   WPROJ=SQUARE-8-29-SOE-s233   DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500 VISGATE=0 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq233_chain.console.log"
tmux new -d -s soe_sq2333  "cd /root/workspace/baojiachun && GPU=3 TSEED=2333  DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s2333  WPROJ=SQUARE-8-29-SOE-s2333  DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500 VISGATE=0 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq2333_chain.console.log"
tmux new -d -s soe_sq23333 "cd /root/workspace/baojiachun && GPU=4 TSEED=23333 DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s23333 WPROJ=SQUARE-8-29-SOE-s23333 DATASETS=/root/workspace/baojiachun/soe_data/datasets/square HORIZON=500 VISGATE=0 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh square 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq23333_chain.console.log"

echo "[launch] SOE can s23333 (can protocol: HORIZON=300 VISGATE=1; joins 2026_8_26_soe dir)"
tmux new -d -s soe_can23333 "cd /root/workspace/baojiachun && GPU=5 TSEED=23333 DATA_ROOT=/root/workspace/baojiachun/soe_data/2026_8_26_soe/SOE-s23333 WPROJ=CAN-8-24-SOE-s23333 SAVE0=200 SAVE=200 PRUNE_ACCUM=1 bash SOE_scripts_2/chain_soe.sh can 2>&1 | tee -a /root/workspace/baojiachun/soe_data/2026_8_26_soe/logs/soe23333_chain.console.log"

sleep 5
tmux ls
echo "[launch] all 6 sessions started"
