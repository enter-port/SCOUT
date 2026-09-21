#!/bin/bash
# threading_launch.sh -- THREADING-MG-p1 MASTER ORCHESTRATOR (2026-09-16).
# Run ONCE inside tmux (session thread_launch); drives the whole pipeline to
# 9 running chains, per user order "一直跑到9条训练全部跑起来":
#   A) per-seed core20 prep (thread_prep_data.sh x3: split + abs conversion
#      + reset-reproducibility probe)
#   B) round0 x3 seeds in parallel (tmux thread_s<seed>_BASE, GPU 1/2/3;
#      DP-base 600ep b64 + dyn-base) -> wait for the 3 ROUND0 TOTAL lines
#   C) grid search on the s233 base: the single parameter = shared raw
#      guidance_scale, etas 0.1/0.25/0.5/1.0/2.0 x {ATY,ORBIT} (two
#      sequential-per-arm streams on GPU 4 and 5; each cell <=30min hard
#      cap; verdict = ONE shared eta for both guided arms, argmax of the
#      per-eta SUM of the two arms' pass@5, tie -> closest to 0.5 in log
#      space; user 2026-09-17 三个臂同一组参数)
#   D) start 9 chains (thread_start_arms.sh per seed, GPUs rotated 1..6:
#      s233->1,2,3 s2333->4,5,6 s23333->1,2,3; render sharing per tp15
#      precedent)
# usage: tmux new-session -d -s thread_launch 'bash scripts/threading/threading_launch.sh'
set -uo pipefail
ROOT=/root/workspace/baojiachun/scout
CAMP=$ROOT/data/2026_9_16_threading_p1
SEEDS="233 2333 23333"
ETAS="0.1 0.25 0.5 1.0 2.0"
GRIDROOT=$CAMP/GRID_s233
ROUND0_TIMEOUT_S=$(( 8*3600 ))    # 8h hard wait for the three round0s
cd "$ROOT" || exit 1

log(){ echo "[$(date '+%F %T')] [launch] $*"; }

# ---------------- A) per-seed core20 prep ---------------------------------- #
for s in $SEEDS; do
  log "prep s$s ..."
  bash scripts/threading/thread_prep_data.sh "$s" || { log "FATAL: prep s$s failed"; exit 1; }
done

# ---------------- B) round0 x3 (parallel) + wait --------------------------- #
GPU_OF_S233=1; GPU_OF_S2333=2; GPU_OF_S23333=3
for s in $SEEDS; do
  g=$(eval echo \${GPU_OF_S$s})
  RL=$CAMP/THREADING-MG-p1-s$s/threading/round.log
  grep -q "ROUND0 threading seed=$s TOTAL" "$RL" 2>/dev/null && { log "round0 s$s already done -- skip"; continue; }
  if tmux has-session -t "thread_s${s}_BASE" 2>/dev/null; then
    log "tmux thread_s${s}_BASE exists -- reuse"
  else
    tmux new-session -d -s "thread_s${s}_BASE" \
      "cd $ROOT && SEED=$s GPU=$g ARM=BASE NROUNDS=6 bash scripts/threading/thread_chain.sh"
    log "round0 s$s launched @GPU$g (tmux thread_s${s}_BASE)"
  fi
done
T0=$(date +%s)
while true; do
  done_n=0
  for s in $SEEDS; do
    grep -q "ROUND0 threading seed=$s TOTAL" "$CAMP/THREADING-MG-p1-s$s/threading/round.log" 2>/dev/null && done_n=$((done_n+1))
  done
  [ "$done_n" = 3 ] && { log "all 3 round0 TOTAL lines present"; break; }
  [ $(( $(date +%s) - T0 )) -gt "$ROUND0_TIMEOUT_S" ] && { log "FATAL: round0 wait timeout (${done_n}/3 done)"; exit 1; }
  sleep 120
done
for s in $SEEDS; do tmux kill-session -t "thread_s${s}_BASE" 2>/dev/null; done

# ---------------- C) grid on the s233 base --------------------------------- #
mkdir -p "$GRIDROOT"
# P1-1 (review 2026-09-16): never parse stale rows across relaunches -- the
# probe outputs are throwaway; archive any previous csv before the streams.
mv -f "$GRIDROOT/grid_results.csv" "$GRIDROOT/grid_results.csv.$(date +%Y%m%d_%H%M%S).bak" 2>/dev/null || true
GLOG=$GRIDROOT/grid_streams.log
log "grid start: etas=[$ETAS] x {ATY@GPU4, ORBIT@GPU5}; log=$GLOG"
(
  for e in $ETAS; do
    ARM=ATY ETA=$e GPU=4 CELL="ATY_eta${e}" bash scripts/threading/thread_grid_probe.sh
  done
) >> "$GLOG" 2>&1 &
GRID_ATY_PID=$!
(
  for e in $ETAS; do
    ARM=ORBIT ETA=$e GPU=5 CELL="ORBIT_eta${e}" bash scripts/threading/thread_grid_probe.sh
  done
) >> "$GLOG" 2>&1 &
GRID_ORB_PID=$!
wait $GRID_ATY_PID $GRID_ORB_PID
log "grid done"

BEST=$(/root/workspace/baojiachun/.venv_mg/bin/python - "$GRIDROOT/grid_results.csv" <<'PYEOF'
import sys, math, csv, collections
rows = list(csv.DictReader(open(sys.argv[1])))
by_eta = collections.defaultdict(dict)
for r in rows:
    if r["pass5"] not in ("", None):
        by_eta[float(r["eta"])][r["arm"]] = float(r["pass5"])
# ONE shared dose (user 2026-09-17: 三个臂同一组参数, seed-233 calibration):
# eta* = argmax_eta [pass@5_ATY(eta) + pass@5_ORBIT(eta)]; tie -> eta closest
# to 0.5 in log space. Per-arm readouts stay in the csv for re-picking.
for eta in sorted(by_eta):
    pa = by_eta[eta]
    print(f"[grid-parse] eta={eta}: ATY={pa.get('ATY','-')} ORBIT={pa.get('ORBIT','-')}")
cand = [(per_arm["ATY"] + per_arm["ORBIT"], eta)
        for eta, per_arm in by_eta.items()
        if "ATY" in per_arm and "ORBIT" in per_arm]
if not cand:
    raise SystemExit("[grid-parse] FATAL: no eta with BOTH arms completed")
mx = max(s for s, _ in cand)
tied = [e for s, e in cand if s == mx]
eta_star = min(tied, key=lambda e: abs(math.log(e / 0.5)))
print(f"[grid-parse] SHARED eta*={eta_star} (combined={mx}, "
      f"ATY={by_eta[eta_star]['ATY']}, ORBIT={by_eta[eta_star]['ORBIT']}, tied={tied})")
print(f"BEST_ETA={eta_star}")
PYEOF
) || { log "FATAL: grid parse failed"; exit 1; }
echo "$BEST"
BEST_ETA=$(echo "$BEST" | grep '^BEST_ETA=' | cut -d= -f2)
[ -n "$BEST_ETA" ] || { log "FATAL: best eta parse empty"; exit 1; }
log "grid verdict: SHARED eta=$BEST_ETA for BOTH guided arms (seed-233 calibration)"

# ---------------- D) 9 chains across GPU1-6 (render shared) ---------------- #
# ONE shared dose for all arms (user 2026-09-17); the DP arm ignores it.
export ATY_SCALE=$BEST_ETA
export ORB_SIGMA=0.05
bash scripts/threading/thread_start_arms.sh 233   1 2 3
bash scripts/threading/thread_start_arms.sh 2333  4 5 6
bash scripts/threading/thread_start_arms.sh 23333 1 2 3
log "ALL 9 CHAINS LAUNCHED (shared eta=$BEST_ETA)"
echo "LAUNCH_DONE $(date '+%F %T')"
