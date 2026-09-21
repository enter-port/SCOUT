#!/bin/bash
# ckp_launch.sh -- COFFEEPREP-MG-p1 MASTER ORCHESTRATOR (2026-09-20).
# COPY lineage: scripts/threading/threading_launch.sh, adapted for the
# autonomous two-machine flow (user order 2026-09-20: full pipeline, no
# stops; eta ties resolved by the agent). Run ONCE inside tmux on 1022:
#   A) wait for the pool download (nohup curl may still be running) + verify
#   B) per-seed core20 prep (ckp_prep_data.sh x3: split + abs conversion +
#      reset-reproducibility probe)
#   C) round0 x3 seeds in parallel (tmux ckp_s<seed>_BASE on 1022 GPU 0/1/2;
#      DP-base 600ep b64 + dyn-base) -> wait for the 3 ROUND0 TOTAL lines
#   D) grid on 1022 LOCAL (tmux ckp_grid, GPUs 0-6; 1022->1024 ssh has no
#      host key for a bare BatchMode call -- grid stays on the orchestrator's
#      machine; driver: grad probe -> preseed eval -> dose probes -> placebo
#      -> coarse ATY -> adaptive loop -> kappa fill -> ORBIT -> verdict.json)
#   E) parse verdict.json (local read) -> ATY_SCALE / ATT_CAP / ORB_SIGMA / ORB_LAM
#   F) start 9 chains: 1022 s233->GPU0/1/2, s2333->GPU3/4/5;
#      1024 s23333->GPU0/1/2 (ssh with accept-new + known_hosts inside
#      baojiachun). One chain per GPU.
# usage: tmux new-session -d -s ckp_launch 'bash scripts/coffee_prep/ckp_launch.sh'
set -uo pipefail
ROOT=/root/workspace/baojiachun/scout
CAMP=$ROOT/data/2026_9_19_coffeeprep_mg_p1
SEEDS="233 2333 23333"
GRIDROOT=$CAMP/GRID_s233
ROUND0_TIMEOUT_S=$(( 16*3600 ))
GRID_TIMEOUT_S=$(( 44*3600 ))
SSH4="ssh -o BatchMode=yes -o ConnectTimeout=15 -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/root/workspace/baojiachun/.ssh_known_hosts_1024 -p 1024 root@106.14.2.243"
cd "$ROOT" || exit 1

log(){ echo "[$(date '+%F %T')] [launch] $*"; }

# ---------------- A) pool download wait + verify ---------------------------- #
POOL=$ROOT/data/robomimic/coffee_prep/mg/coffee_preparation_d0.hdf5
EXPECT_BYTES=7187662878
log "A: waiting for pool download (7.19G from hf-mirror) ..."
T0=$(date +%s)
while true; do
  SZ=$(stat -c%s "$POOL" 2>/dev/null || echo 0)
  [ "$SZ" -ge "$EXPECT_BYTES" ] && { log "A: pool complete ($SZ bytes)"; break; }
  if ! pgrep -f "curl.*coffee_preparation_d0" > /dev/null 2>&1; then
    log "A: no active download and size $SZ < $EXPECT_BYTES -- (re)starting download"
    bash scripts/coffee_prep/ckp_download_pool.sh >> "$CAMP/pool_download.log" 2>&1 \
      || { log "FATAL: pool download failed"; exit 1; }
    break
  fi
  [ $(( $(date +%s) - T0 )) -gt "$(( 4*3600 ))" ] && { log "FATAL: pool download wait timeout (4h)"; exit 1; }
  sleep 60
done
bash scripts/coffee_prep/ckp_download_pool.sh >> "$CAMP/pool_download.log" 2>&1 \
  || { log "FATAL: pool verify failed"; exit 1; }
log "A: pool verified"

# ---------------- B) per-seed core20 prep ------------------------------------ #
for s in $SEEDS; do
  log "B: prep s$s ..."
  bash scripts/coffee_prep/ckp_prep_data.sh "$s" || { log "FATAL: prep s$s failed"; exit 1; }
done

# ---------------- C) round0 x3 (parallel on 1022) + wait --------------------- #
for spec in "233:0" "2333:1" "23333:2"; do
  s=${spec%%:*}; g=${spec##*:}
  RL=$CAMP/COFFEEPREP-MG-p1-s$s/coffee_prep/round.log
  grep -q "ROUND0 coffee_prep seed=$s TOTAL" "$RL" 2>/dev/null && { log "C: round0 s$s already done -- skip"; continue; }
  if tmux has-session -t "ckp_s${s}_BASE" 2>/dev/null; then
    log "C: tmux ckp_s${s}_BASE exists -- reuse"
  else
    tmux new-session -d -s "ckp_s${s}_BASE" \
      "cd $ROOT && SEED=$s GPU=$g ARM=BASE NROUNDS=6 bash scripts/coffee_prep/ckp_chain.sh"
    log "C: round0 s$s launched @1022 GPU$g (tmux ckp_s${s}_BASE)"
  fi
done
T0=$(date +%s)
while true; do
  done_n=0
  for s in $SEEDS; do
    grep -q "ROUND0 coffee_prep seed=$s TOTAL" "$CAMP/COFFEEPREP-MG-p1-s$s/coffee_prep/round.log" 2>/dev/null && done_n=$((done_n+1))
  done
  [ "$done_n" = 3 ] && { log "C: all 3 round0 TOTAL lines present"; break; }
  [ $(( $(date +%s) - T0 )) -gt "$ROUND0_TIMEOUT_S" ] && { log "FATAL: round0 wait timeout (${done_n}/3 done)"; exit 1; }
  sleep 300
done
for s in $SEEDS; do tmux kill-session -t "ckp_s${s}_BASE" 2>/dev/null || true; done

# ---------------- D) grid on 1024 -------------------------------------------- #
if [ -f "$GRIDROOT/verdict.json" ] && ! grep -q '"error"' "$GRIDROOT/verdict.json" 2>/dev/null; then
  log "D: grid verdict already present -- skip"
else
  mkdir -p "$GRIDROOT"
  if tmux has-session -t ckp_grid 2>/dev/null; then
    log "D: tmux ckp_grid already exists (1022 local) -- reuse"
  else
    tmux new-session -d -s ckp_grid "GPUS=\"0 1 2 3 4 5 6\" bash scripts/coffee_prep/ckp_grid_driver.sh >> $GRIDROOT/driver_console.log 2>&1" \
      || { log "FATAL: failed to spawn ckp_grid locally"; exit 1; }
    log "D: grid driver spawned on 1022 (tmux ckp_grid, GPUs 0-6)"
  fi
  T0=$(date +%s)
  while true; do
    if [ -f "$GRIDROOT/verdict.json" ] && ! grep -q '"error"' "$GRIDROOT/verdict.json" 2>/dev/null; then
      log "D: grid verdict ready"; break
    fi
    if ! tmux has-session -t ckp_grid 2>/dev/null; then
      log "FATAL: ckp_grid tmux gone and no verdict (see $GRIDROOT/driver_console.log)"; exit 1
    fi
    [ $(( $(date +%s) - T0 )) -gt "$GRID_TIMEOUT_S" ] && { log "FATAL: grid wait timeout (44h)"; exit 1; }
    sleep 600
  done
fi

# ---------------- E) parse verdict ------------------------------------------- #
[ -f "$GRIDROOT/verdict.json" ] || { log "FATAL: verdict.json missing"; exit 1; }
PARAMS=$(/root/workspace/baojiachun/.venv_mg/bin/python - "$GRIDROOT/verdict.json" <<'PYEOF'
import sys, json
d = json.load(open(sys.argv[1]))
if "error" in d:
    raise SystemExit("verdict error: " + d["error"])
print(f"ATY_SCALE={d['eta_star']}")
print(f"ATT_CAP={d['kappa_star']}")
print(f"ORB_SIGMA={d.get('sigma_star', 0.05)}")
print(f"ORB_LAM={d.get('lam_star', 0.5)}")
print(f"ATY_MAX_P5={d.get('aty_max_pass5')}")
print(f"ORBIT_MAX_P5={d.get('orbit_max_pass5')}")
PYEOF
) || { log "FATAL: verdict parse failed"; exit 1; }
log "E: verdict params:"
echo "$PARAMS" | while IFS= read -r l; do log "E:   $l"; done
eval "$PARAMS"
[ -n "${ATY_SCALE:-}" ] || { log "FATAL: ATY_SCALE empty"; exit 1; }

# ---------------- F) 9 chains (1022 G0-5 + 1024 G0-2) ------------------------- #
log "F: starting arms s233 (1022 GPU0/1/2) + s2333 (1022 GPU3/4/5)"
ATY_SCALE=$ATY_SCALE ATT_CAP=$ATT_CAP ORB_SIGMA=$ORB_SIGMA ORB_LAM=$ORB_LAM \
  bash scripts/coffee_prep/ckp_start_arms.sh 233  0 1 2 \
  || { log "FATAL: start_arms s233 failed"; exit 1; }
ATY_SCALE=$ATY_SCALE ATT_CAP=$ATT_CAP ORB_SIGMA=$ORB_SIGMA ORB_LAM=$ORB_LAM \
  bash scripts/coffee_prep/ckp_start_arms.sh 2333 3 4 5 \
  || { log "FATAL: start_arms s2333 failed"; exit 1; }
log "F: starting arms s23333 (1024 GPU0/1/2) via ssh"
$SSH4 "cd $ROOT && ATY_SCALE=$ATY_SCALE ATT_CAP=$ATT_CAP ORB_SIGMA=$ORB_SIGMA ORB_LAM=$ORB_LAM bash scripts/coffee_prep/ckp_start_arms.sh 23333 0 1 2" \
  || { log "FATAL: start_arms s23333 (1024) failed"; exit 1; }

log "ALL 9 CHAINS LAUNCHED (eta=$ATY_SCALE kappa=$ATT_CAP sigma=$ORB_SIGMA lam=$ORB_LAM)"
echo "LAUNCH_DONE $(date '+%F %T')"
