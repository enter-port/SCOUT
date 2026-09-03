#!/bin/bash
# resume_watch.sh -- the 14:23 relaunch of the 233333 arms used a nested-
# quoting for-loop whose $n breaks after the in-flight direct round finishes;
# those sessions will exit on their own once the current round completes.
# This watcher waits for each session to disappear, then resumes the
# remaining rounds via resume_chain.sh (which re-runs a failed round cleanly).
set -u
cd /root/workspace/baojiachun/scout || exit 1
D33=/root/workspace/baojiachun/scout/data/2026_8_21/CAN-exp1-233333
W=CAN-2026-8-21-s233333

while tmux has-session -t chain_233333_SCOUT 2>/dev/null; do sleep 60; done
echo "[watch $(date '+%F %T')] 233333-SCOUT session gone -- resuming rounds 3..6 (GPU2)"
env GPU=2 TSEED=233333 DATA_ROOT=$D33 WPROJ=$W \
  bash soe_scripts/resume_chain.sh can SCOUT 3 6 0 >> data/logs/chain_233333_SCOUT.log 2>&1

while tmux has-session -t chain_233333_DP 2>/dev/null; do sleep 60; done
echo "[watch $(date '+%F %T')] 233333-DP session gone -- resuming rounds 4..6 (GPU3)"
env GPU=3 TSEED=233333 DATA_ROOT=$D33 WPROJ=$W \
  bash soe_scripts/resume_chain.sh can DP 4 6 0 >> data/logs/chain_233333_DP.log 2>&1

echo "[watch $(date '+%F %T')] all 233333 arms resumed-and-done"
