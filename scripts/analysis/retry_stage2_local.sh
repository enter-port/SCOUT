#!/bin/bash
# SQUARE beat-SOE stage-2 launcher + collector (2026-09-01, local).
# Control channel to the server is flaky tonight (hung ssh data paths), so:
# short-lived probes with hard timeouts, idempotent launch guard, then a
# poll-collect phase with fresh connections. Notifies via task completion.
set -u
SSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -o ServerAliveInterval=10 -o ServerAliveCountMax=3 -p 1022 root@106.14.2.243"
BASE=/root/workspace/baojiachun/scout-rand/data/particle/sq2_conf_s233_seed43_gate0

echo "=== phase 1: launch (idempotent, network-retrying) $(date +%H:%M:%S)"
launched=no
for attempt in 1 2 3 4 5 6; do
  probe=$(timeout 90 $SSH "tmux ls 2>/dev/null | grep -c sq2s2; ls $BASE/*/log/*_rollout_exp0.json 2>/dev/null | wc -l" 2>&1)
  rc=$?
  n_sess=$(echo "$probe" | head -1 | tr -d '\r')
  n_json=$(echo "$probe" | tail -1 | tr -d '\r')
  echo "attempt $attempt: probe rc=$rc sess=${n_sess:-?} json=${n_json:-?}"
  if [ "$rc" -ne 0 ]; then echo "  ssh stalled, sleep 300s"; sleep 300; continue; fi
  if [ "${n_sess:-0}" -ge 3 ] || [ "${n_json:-0}" -ge 3 ]; then echo "  already launched, skip"; launched=yes; break; fi
  dry=$(timeout 120 $SSH "cd /root/workspace/baojiachun/scout-rand; DRYRUN=1 SPLIT=orb025:4 SEED=43 GATE=0 GPU=9 bash /tmp/sq2_conf.sh 233; DRYRUN=1 SPLIT=par:3 SEED=43 GATE=0 GPU=9 bash /tmp/sq2_conf.sh 233; DRYRUN=1 SPLIT=orb015:3 SEED=43 GATE=0 GPU=9 bash /tmp/sq2_conf.sh 233" 2>&1)
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "  dryrun ssh stalled rc=$rc, sleep 300s"; sleep 300; continue; fi
  echo "$dry" | grep -E 'failed inits|DRYRUN:' | sed 's/args=.*--guide/args=...--guide/'
  if [ "$(echo "$dry" | grep -c '62 failed inits')" -ne 3 ]; then
    echo "  DRYRUN BAD (expected 3x '62 failed inits'), ABORT for manual inspection"
    break
  fi
  timeout 60 $SSH "L=/root/workspace/baojiachun/scout-rand/data/particle/sq2_stage2.log; tmux new-session -d -s sq2s2a 'cd /root/workspace/baojiachun/scout-rand && SPLIT=orb025:4 SEED=43 GATE=0 GPU=0 bash /tmp/sq2_conf.sh 233 >> '\$L' 2>&1'; tmux new-session -d -s sq2s2b 'cd /root/workspace/baojiachun/scout-rand && SPLIT=par:3 SEED=43 GATE=0 GPU=1 bash /tmp/sq2_conf.sh 233 >> '\$L' 2>&1'; tmux new-session -d -s sq2s2c 'cd /root/workspace/baojiachun/scout-rand && SPLIT=orb015:3 SEED=43 GATE=0 GPU=4 bash /tmp/sq2_conf.sh 233 >> '\$L' 2>&1'; sleep 3; tmux ls | grep sq2s2"
  rc=$?
  if [ "$rc" -ne 0 ]; then echo "  launch ssh stalled rc=$rc (remote may have launched; next probe detects), sleep 120s"; sleep 120; continue; fi
  echo "  LAUNCHED_OK (attempt $attempt)"
  launched=yes
  break
done

if [ "$launched" != "yes" ]; then
  echo "=== PHASE1 FAILED: launch not confirmed after retries. Manual handling needed."
  exit 1
fi

echo "=== phase 2: poll-collect (fresh ssh each probe, 5min interval, max 200min) $(date +%H:%M:%S)"
mkdir -p "$(dirname "$0")/../experiments/sq2_stage2"
for i in $(seq 1 40); do
  sleep 300
  n=$(timeout 60 $SSH "ls $BASE/{orb025,par,orb015}/log/square_SCOUT_rollout_exp0.json 2>/dev/null | wc -l" 2>/dev/null | tr -d '\r')
  echo "poll $i: jsons=${n:-probe-failed}"
  if [ "${n:-0}" -ge 3 ]; then
    echo "=== all 3 stage-2 jsons present, fetching $(date +%H:%M:%S)"
    for arm in orb025 par orb015; do
      timeout 90 $SSH "cat $BASE/$arm/log/square_SCOUT_rollout_exp0.json" > "$(dirname "$0")/../experiments/sq2_stage2/$arm.json" 2>/dev/null
      echo "  $arm: $(wc -c < "$(dirname "$0")/../experiments/sq2_stage2/$arm.json") bytes"
    done
    echo "=== STAGE2_DATA_READY"
    exit 0
  fi
done
echo "=== PHASE2 TIMEOUT (200min) with incomplete jsons"
exit 2
