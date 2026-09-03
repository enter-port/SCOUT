#!/bin/bash
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -p 1022 root@106.14.2.243"
BASE=/root/workspace/baojiachun/scout-rand/data/particle/sq2_conf_s233_seed42_gate1
D="$(cd "$(dirname "$0")" && pwd)/../experiments/sq2_gate"
mkdir -p "$D"
for i in $(seq 1 20); do
  sleep 300
  n=$(timeout 60 $SSH "ls $BASE/{ray,ray0}/log/square_SCOUT_rollout_exp0.json 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r')
  echo "poll $i: jsons=${n:-probe-failed} $(date +%H:%M:%S)"
  if [ "${n:-0}" -ge 2 ]; then
    for arm in ray ray0; do
      timeout 90 $SSH "cat $BASE/$arm/log/square_SCOUT_rollout_exp0.json" > "$D/$arm.json" 2>/dev/null
      echo "  $arm: $(wc -c < "$D/$arm.json") bytes"
    done
    echo "=== RAY_GATE_DATA_READY"
    exit 0
  fi
done
echo "=== TIMEOUT"
exit 2
