#!/bin/bash
# cfk4_launch.sh -- COFFEE-MG-p4 one-shot orchestrator (2026-09-22).
# 2 arms (DP baseline + ATY with per-round R-anchored eta re-calibration
#   eta_i = eta_{i-1}*0.01/R_mean, converge R_mean in [0.009,0.011];
#   user-specified 2026-09-22)
# x 3 seeds (233/2333/23333), 6 rounds, base DP+DYN REUSED from p1
# (no round0; three-piece pre-copied idempotently with cp -L to
# dereference possible ckpt symlinks).
# tmux sessions: cfk4_<arm>_s<seed> on 1022 GPU0-5.
set -u
set -o pipefail
ROOT=/root/workspace/baojiachun/scout
SRC=$ROOT/data/2026_9_20_coffee_mg_p1
DST=$ROOT/data/2026_9_22_coffee_mg_p4
cd "$ROOT" || exit 1

# --- gate: GPUs 0-5 must be idle (shared-server rule) ---
for g in 0 1 2 3 4 5; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -lt 1000 ] || { echo "[cfk4] FATAL GPU$g busy (${used}MiB)"; exit 1; }
done

# --- pre-copy the base three-piece per seed (idempotent; DP-base carries
#     ONLY the newest ckpt + config/train.log -- the round script resolves
#     inputs via newest_ckpt, and the intermediate 99-499 stay in p1) ---
for SEED in 233 2333 23333; do
  S1=$SRC/COFFEE-MG-p1-s$SEED/coffee
  S2=$DST/COFFEE-MG-p4-s$SEED/coffee
  DPNEW=$(ls -t $S1/train/DP/DP-base/checkpoints/*.ckpt 2>/dev/null | head -1)
  [ -n "$DPNEW" ] || { echo "[cfk4] FATAL no DP-base ckpt for s$SEED"; exit 1; }
  [ -n "$(ls -t $S1/train/dyn/dyn-base/*/scout_vib.ckpt 2>/dev/null | head -1)" ] \
    || { echo "[cfk4] FATAL no dyn-base ckpt for s$SEED"; exit 1; }
  [ -f "$S1/rollout/coffee_core.hdf5" ] || { echo "[cfk4] FATAL missing source core for s$SEED"; exit 1; }
  mkdir -p "$S2/rollout" "$S2/train/DP/DP-base/checkpoints" "$S2/train/dyn"
  copy_chk(){  # copy with SIZE verification (a half-written leftover re-copies)
    local srcf=$1 dstf=$2
    if [ ! -f "$dstf" ] || [ "$(stat -c%s "$srcf")" != "$(stat -c%s "$dstf")" ]; then
      cp -L "$srcf" "$dstf" || { echo "[cfk4] FATAL copy failed: $srcf"; exit 1; }
    fi
  }
  copy_chk "$S1/rollout/coffee_core.hdf5" "$S2/rollout/coffee_core.hdf5"
  copy_chk "$DPNEW" "$S2/train/DP/DP-base/checkpoints/$(basename $DPNEW)"
  for aux in config.yaml train.log; do
    [ -f "$S1/train/DP/DP-base/$aux" ] && { [ -f "$S2/train/DP/DP-base/$aux" ] \
      || cp -L "$S1/train/DP/DP-base/$aux" "$S2/train/DP/DP-base/"; }
  done
  [ -d "$S2/train/dyn/dyn-base" ] || cp -rL "$S1/train/dyn/dyn-base" "$S2/train/dyn/dyn-base"
  # seed the shared round.log so cfk4_chain.sh finds it (chains append)
  [ -f "$S2/round.log" ] || echo "[$(date '+%F %T')] [cfk4] base three-piece copied from COFFEE-MG-p1-s$SEED (core + DP-base $DPNEW + dyn-base; no round0)" >> "$S2/round.log"
  echo "[cfk4] s$SEED base ready: core=$(stat -c%s $S2/rollout/coffee_core.hdf5)B dp=$(ls -t $S2/train/DP/DP-base/checkpoints/*.ckpt | head -1 | xargs basename) vib=$(ls -t $S2/train/dyn/dyn-base/*/scout_vib.ckpt | head -1 | xargs basename)"
done

if [ "${COPY_ONLY:-0}" = 1 ]; then
  echo "[cfk4] COPY_ONLY=1 -- base staged, not spawning"
  exit 0
fi

# --- spawn 6 chains (tmux; GPU map: s233 G0/G1, s2333 G2/G3, s23333 G4/G5) ---
g=0
for SEED in 233 2333 23333; do
  for ARM in DP ATY; do
    tmux new-session -d -s cfk4_${ARM,,}_s$SEED \
      "cd $ROOT && SEED=$SEED GPU=$g ARM=$ARM NROUNDS=6 bash scripts/coffee/cfk4_chain.sh"
    echo "[cfk4] spawned cfk4_${ARM,,}_s$SEED gpu=$g"
    g=$((g+1))
  done
done
echo "[cfk4] LAUNCH_DONE $(date '+%F %T')"
