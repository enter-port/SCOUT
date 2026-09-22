#!/bin/bash
# thm2_launch.sh -- THREADING-MG-p2 one-shot orchestrator (2026-09-22).
# Runs ON THE PORT-1024 HOST (dsw-801389-*) where all 6 chains live:
#   A) per-seed staging: core20 hdf5 + DP-base 599.ckpt copied with cp -L
#      from THREADING-MG-p1 (size-verified, idempotent). The dyn-base is NOT
#      copied -- it is RE-TRAINED with beta=1e-5 (user order 2026-09-22).
#   B) round0 (= dyn-base x3 seeds, beta=1e-5) in parallel via tmux
#      thm2_base_s<seed> on GPU 0/1/2; wait for the 3 "ROUND0 threading
#      seed=<s> TOTAL" lines, then kill those sessions.
#   C) spawn the 6 chains, one GPU each: s233 -> G0(DP)/G1(ATY),
#      s2333 -> G2/G3, s23333 -> G4/G5. Dose: eta_0=1.0, kappa_0=2.5,
#      per-round R-anchored eta re-calibration inside round_thm2.sh.
#
# usage: tmux new-session -d -s thm2_launch 'bash scripts/threading/thm2_launch.sh'
#        COPY_ONLY=1 bash scripts/threading/thm2_launch.sh   # stage only
set -u
set -o pipefail
ROOT=/root/workspace/baojiachun/scout
SRC=$ROOT/data/2026_9_16_threading_p1
DST=$ROOT/data/2026_9_22_threading_mg_p2
SEEDS="233 2333 23333"
ROUND0_TIMEOUT_S=$(( 3*3600 ))
export ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5
cd "$ROOT" || exit 1

log(){ echo "[$(date '+%F %T')] [thm2-launch] $*"; }

# --- gate: GPUs 0-5 must be idle (shared-server rule) --- #
for g in 0 1 2 3 4 5; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -lt 1000 ] || { echo "[thm2-launch] FATAL GPU$g busy (${used}MiB)"; exit 1; }
done

# --- A) stage core20 + DP-base per seed (idempotent, size-verified) --- #
for s in $SEEDS; do
  S1=$SRC/THREADING-MG-p1-s$s/threading
  S2=$DST/THREADING-MG-p2-s$s/threading
  DPNEW=$(ls -t $S1/train/DP/DP-base/checkpoints/*.ckpt 2>/dev/null | head -1)
  [ -n "$DPNEW" ] || { echo "[thm2-launch] FATAL no DP-base ckpt for s$s"; exit 1; }
  [ -f "$S1/rollout/threading_core.hdf5" ] || { echo "[thm2-launch] FATAL missing source core for s$s"; exit 1; }
  mkdir -p "$S2/rollout" "$S2/train/DP/DP-base/checkpoints" "$S2/train/dyn"
  copy_chk(){  # copy with SIZE verification (a half-written leftover re-copies)
    local srcf=$1 dstf=$2
    if [ ! -f "$dstf" ] || [ "$(stat -c%s "$srcf")" != "$(stat -c%s "$dstf")" ]; then
      cp -L "$srcf" "$dstf" || { echo "[thm2-launch] FATAL copy failed: $srcf"; exit 1; }
    fi
  }
  copy_chk "$S1/rollout/threading_core.hdf5" "$S2/rollout/threading_core.hdf5"
  copy_chk "$DPNEW" "$S2/train/DP/DP-base/checkpoints/$(basename $DPNEW)"
  for aux in config.yaml train.log; do
    [ -f "$S1/train/DP/DP-base/$aux" ] && { [ -f "$S2/train/DP/DP-base/$aux" ] \
      || cp -L "$S1/train/DP/DP-base/$aux" "$S2/train/DP/DP-base/"; }
  done
  log "s$s base ready: core=$(stat -c%s $S2/rollout/threading_core.hdf5)B dp=$(basename $(ls -t $S2/train/DP/DP-base/checkpoints/*.ckpt | head -1))"
done

if [ "${COPY_ONLY:-0}" = 1 ]; then
  log "COPY_ONLY=1 -- base staged, not spawning"
  exit 0
fi

# --- B) round0 = dyn-base x3 with beta=1e-5 (parallel, GPU 0/1/2) --- #
g=0
for s in $SEEDS; do
  RL=$DST/THREADING-MG-p2-s$s/threading/round.log
  if grep -q "ROUND0 threading seed=$s TOTAL" "$RL" 2>/dev/null; then
    log "dyn-base s$s already done -- skip"; g=$((g+1)); continue
  fi
  if tmux has-session -t "thm2_base_s$s" 2>/dev/null; then
    log "tmux thm2_base_s$s exists -- reuse"
  else
    tmux new-session -d -s "thm2_base_s$s" \
      "cd $ROOT && SEED=$s GPU=$g ARM=BASE NROUNDS=6 ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 bash scripts/threading/thm2_chain.sh"
    log "dyn-base s$s launched @GPU$g (tmux thm2_base_s$s, beta=$BETA)"
  fi
  g=$((g+1))
done
T0=$(date +%s)
while true; do
  done_n=0
  for s in $SEEDS; do
    grep -q "ROUND0 threading seed=$s TOTAL" "$DST/THREADING-MG-p2-s$s/threading/round.log" 2>/dev/null && done_n=$((done_n+1))
  done
  [ "$done_n" = 3 ] && { log "all 3 dyn-base TOTAL lines present"; break; }
  [ $(( $(date +%s) - T0 )) -gt "$ROUND0_TIMEOUT_S" ] && { log "FATAL: dyn-base wait timeout (${done_n}/3 done)"; exit 1; }
  sleep 60
done
for s in $SEEDS; do tmux kill-session -t "thm2_base_s$s" 2>/dev/null; done

# --- C) spawn the 6 chains (GPU map: s233 G0/G1, s2333 G2/G3, s23333 G4/G5) --- #
g=0
for s in $SEEDS; do
  for ARM in DP ATY; do
    tmux new-session -d -s "thm2_${ARM,,}_s$s" \
      "cd $ROOT && SEED=$s GPU=$g ARM=$ARM NROUNDS=6 DATA_ROOT=$DST/THREADING-MG-p2-s$s WPROJ=THREADING-MG-p2-s$s ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5 bash scripts/threading/thm2_chain.sh"
    log "spawned thm2_${ARM,,}_s$s gpu=$g"
    g=$((g+1))
  done
done
log "LAUNCH_DONE $(date '+%F %T')"
