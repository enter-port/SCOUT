#!/bin/bash
# stop_soe_sq_0829.sh -- user order 08-29: rename SQUARE-29-SOE -> SQUARE-8-29-SOE.
# Stops the 3 misnamed SOE square chains (kill order: tmux session -> leftover
# python, per the standing order), removes their ~45min partial round0 outputs.
# Does NOT touch soe_can23333 / sq23333_SCOUT / sq23333_waitDP (correct names).
set -u
for s in soe_sq233 soe_sq2333 soe_sq23333; do
  tmux kill-session -t "$s" 2>/dev/null && echo "killed tmux $s" || echo "no tmux $s"
done
sleep 3
# kill any orphaned train/tail processes from these chains (by PID, no pkill -f self-match)
for pid in $(ps -eo pid,args | grep -E '2026_8_29_soe/SOE-s(233|2333|23333)' | grep -v grep | awk '{print $1}'); do
  kill "$pid" 2>/dev/null && echo "killed leftover pid $pid"
done
sleep 2
echo "== remaining 0829_soe processes (should be empty):"
ps -eo pid,args | grep -E '2026_8_29_soe/SOE-s' | grep -v grep || echo "(none)"
echo "== removing partial outputs"
rm -rf /root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s233 \
       /root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s2333 \
       /root/workspace/baojiachun/soe_data/2026_8_29_soe/SOE-s23333 \
       /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq233_chain.console.log \
       /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq2333_chain.console.log \
       /root/workspace/baojiachun/soe_data/2026_8_29_soe/logs/soe_sq23333_chain.console.log
ls /root/workspace/baojiachun/soe_data/2026_8_29_soe/ 2>/dev/null || echo "(empty)"
echo "== GPUs after stop:"
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | sed -n '3p;4p;5p'
echo "== untouched chains still alive:"
tmux ls 2>/dev/null | grep -E 'soe_can23333|sq23333'
