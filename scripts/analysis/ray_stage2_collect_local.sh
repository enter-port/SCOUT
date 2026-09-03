#!/bin/bash
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -p 1022 root@106.14.2.243"
BASE=/root/workspace/baojiachun/scout-rand/data/particle/sq2_conf_s233_seed43_gate0
D=experiments/sq2_stage2_ray
mkdir -p "$D"
for i in $(seq 1 70); do
  sleep 300
  n=$(timeout 60 $SSH "ls $BASE/ray/log/square_SCOUT_rollout_exp0.json 2>/dev/null | wc -l; ls -la $BASE/orb025/log/square_SCOUT_rollout_exp0.json 2>/dev/null | awk '{print \$5}'" 2>/dev/null | tr -d '\r')
  echo "poll $i: $n $(date +%H:%M:%S)"
  ray_done=$(echo "$n" | head -1)
  if [ "${ray_done:-0}" -ge 1 ]; then
    timeout 90 $SSH "cat $BASE/ray/log/square_SCOUT_rollout_exp0.json" > "$D/ray.json" 2>/dev/null
    timeout 90 $SSH "cat $BASE/orb025/log/square_SCOUT_rollout_exp0.json" > "$D/orb025_k10.json" 2>/dev/null
    echo "  ray: $(wc -c < "$D/ray.json") bytes; orb025_k10: $(wc -c < "$D/orb025_k10.json") bytes (0=control still running)"
    if [ "$(wc -c < "$D/orb025_k10.json")" -gt 1000 ]; then
      echo "=== RAY_STAGE2_DATA_READY"
      exit 0
    fi
  fi
done
echo "=== TIMEOUT"
exit 2
