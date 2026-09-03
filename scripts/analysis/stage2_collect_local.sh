#!/bin/bash
# stage-2 poll-collect standalone (launcher's phase 2 died silently).
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -p 1022 root@106.14.2.243"
BASE=/root/workspace/baojiachun/scout-rand/data/particle/sq2_conf_s233_seed43_gate0
mkdir -p "$(dirname "$0")/../experiments/sq2_stage2"
for i in $(seq 1 44); do
  sleep 300
  n=$(timeout 60 $SSH "ls $BASE/{orb025,par,orb015}/log/square_SCOUT_rollout_exp0.json 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r')
  echo "poll $i: jsons=${n:-probe-failed} $(date +%H:%M:%S)"
  if [ "${n:-0}" -ge 3 ]; then
    echo "=== all 3 stage-2 jsons present, fetching"
    for arm in orb025 par orb015; do
      timeout 90 $SSH "cat $BASE/$arm/log/square_SCOUT_rollout_exp0.json" > "$(dirname "$0")/../experiments/sq2_stage2/$arm.json" 2>/dev/null
      echo "  $arm: $(wc -c < "$(dirname "$0")/../experiments/sq2_stage2/$arm.json") bytes"
    done
    echo "=== STAGE2_DATA_READY"
    exit 0
  fi
done
echo "=== TIMEOUT with incomplete jsons"
exit 2
