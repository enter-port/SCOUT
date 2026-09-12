#!/bin/bash
# sq233x_wait_launch.sh -- deferred launcher for the SQUARE-9-1-orbit
# seed replicates (2026-09-09 user order: start them AFTER both can
# replicate chains finish). Runs inside tmux session sq233x_wait:
#   1. poll until BOTH can round.log files contain the round-6 TOTAL line
#      ("a=SCOUT seed=<s> round=6 TOTAL") -- the exact-completion anchor
#      the can chains emit; note s2333/s23333 are distinguished by the
#      trailing space in the pattern (each file only holds its own seed
#      anyway);
#   2. wait until GPU1 and GPU3 have drained (<2 GB each, 2h ceiling) --
#      the can chains' cards, which the square chains inherit;
#   3. run the two square launchers (each spawns its own tmux chain and
#      returns; each is a no-op if its session already exists).
# READ-ONLY w.r.t. the can chains: it only greps their round.log.
# Append-only log: data/2026_9_1_orbchain/sq233x_wait.console.log
set -u
ROOT=/root/workspace/baojiachun/scout-orbit
LOG=$ROOT/data/2026_9_1_orbchain/sq233x_wait.console.log
mkdir -p "$(dirname "$LOG")"
exec >> "$LOG" 2>&1
echo "[wait] watcher started $(date '+%F %T') (gates: can s2333 round6 TOTAL + can s23333 round6 TOTAL, then GPU1/3 drain)"

can_done(){
  grep -q "a=SCOUT seed=$1 round=6 TOTAL" \
    "$ROOT/data/2026_9_2_orbchain/ORBIT-s$1/can/round.log" 2>/dev/null
}

while :; do
  A=pending; B=pending
  can_done 2333  && A=DONE
  can_done 23333 && B=DONE
  echo "[wait] $(date '+%F %T') can_s2333=$A can_s23333=$B"
  [ "$A" = DONE ] && [ "$B" = DONE ] && break
  sleep 600
done

echo "[wait] both can chains COMPLETE -- waiting for GPU1/3 to drain $(date '+%F %T')"
i=0
while [ $i -lt 24 ]; do
  M1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)
  M3=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 3)
  if [ "${M1:-99999}" -lt 2000 ] && [ "${M3:-99999}" -lt 2000 ]; then
    echo "[wait] GPUs drained (GPU1=${M1}MiB GPU3=${M3}MiB) $(date '+%F %T')"
    break
  fi
  i=$((i+1))
  echo "[wait] GPUs not drained (GPU1=${M1}MiB GPU3=${M3}MiB) -- retry 5min ($(date '+%F %T'))"
  sleep 300
done
if [ $i -ge 24 ]; then
  echo "[wait] WARN: GPUs not drained after 2h -- launching anyway per user order $(date '+%F %T')"
fi

echo "[wait] launching SQUARE-9-1-orbit-s23333 (GPU3) $(date '+%F %T')"
bash "$ROOT/soe_scripts/sq23333_launch.sh"
sleep 60
echo "[wait] launching SQUARE-9-1-orbit-s2333 (GPU1) $(date '+%F %T')"
bash "$ROOT/soe_scripts/sq2333_launch.sh"
echo "[wait] both launchers returned -- watcher exiting $(date '+%F %T')"
