#!/bin/bash
# thm3_launch.sh -- THREADING-THM3 one-shot orchestrator (2026-09-23, user
# plan stage 4a, autonomous session). Runs ON THE PORT-1024 HOST:
#   A) per-seed staging from the p2 campaign: core20 hdf5 + DP-base 599.ckpt
#      (cp -L, size-verified, idempotent) + dyn-base WHOLE DIR copy (the p2
#      dyn-base is already beta=1e-5 / 300ep = the A3 recipe -- no round0
#      retrain on this campaign);
#   B) spawn the 6 chains, one GPU each: s233 -> G0(DP)/G1(ATY),
#      s2333 -> G2/G3, s23333 -> G4/G5. Dose: ETA0=1.0, kappa0=2.5; per-round
#      eta rcalib AND kappa v2 C-anchored recalibration inside round_thm3.sh
#      (BASE_ETA = the per-seed p2 rcalib anchor). explore records with
#      --stop-on-first-success; DP retrain 600ep b64; dyn retrain 300ep.
#
# usage: tmux new-session -d -s thm3_launch 'bash scripts/threading/thm3_launch.sh'
#        COPY_ONLY=1 bash scripts/threading/thm3_launch.sh   # stage only
set -u
set -o pipefail
ROOT=/root/workspace/baojiachun/scout
SRC=$ROOT/data/2026_9_22_threading_mg_p2
DST=$ROOT/data/2026_9_23_threading_thm3
SEEDS="233 2333 23333"
export ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5
# per-seed kappa C-anchor: the p2 campaign's r1 rcalib eta on the base pair
BASE_ETA_233=4.692821504030686
BASE_ETA_2333=2.814421224071731
BASE_ETA_23333=2.040034419235227
cd "$ROOT" || exit 1

log(){ echo "[$(date '+%F %T')] [thm3-launch] $*"; }

# --- gate: GPUs 0-5 must be idle (shared-server rule) --- #
for g in 0 1 2 3 4 5; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -lt 1000 ] || { echo "[thm3-launch] FATAL GPU$g busy (${used}MiB)"; exit 1; }
done

# --- A) stage core20 + DP-base + dyn-base per seed (idempotent) --- #
for s in $SEEDS; do
  S1=$SRC/THREADING-MG-p2-s$s/threading
  S2=$DST/THREADING-THM3-s$s/threading
  DPNEW=$(ls -t $S1/train/DP/DP-base/checkpoints/*.ckpt 2>/dev/null | head -1)
  VIBSRC=$(ls -t $S1/train/dyn/dyn-base/*/scout_vib.ckpt 2>/dev/null | head -1)
  [ -n "$DPNEW" ] || { echo "[thm3-launch] FATAL no DP-base ckpt for s$s"; exit 1; }
  [ -n "$VIBSRC" ] || { echo "[thm3-launch] FATAL no dyn-base vib for s$s"; exit 1; }
  [ -f "$S1/rollout/threading_core.hdf5" ] || { echo "[thm3-launch] FATAL missing source core for s$s"; exit 1; }
  mkdir -p "$S2/rollout" "$S2/train/DP/DP-base/checkpoints" "$S2/train/dyn/dyn-base"
  copy_chk(){  # copy with SIZE verification (a half-written leftover re-copies)
    local srcf=$1 dstf=$2
    if [ ! -f "$dstf" ] || [ "$(stat -c%s "$srcf")" != "$(stat -c%s "$dstf")" ]; then
      cp -L "$srcf" "$dstf" || { echo "[thm3-launch] FATAL copy failed: $srcf"; exit 1; }
    fi
  }
  copy_chk "$S1/rollout/threading_core.hdf5" "$S2/rollout/threading_core.hdf5"
  copy_chk "$DPNEW" "$S2/train/DP/DP-base/checkpoints/$(basename $DPNEW)"
  # dyn-base: ckpt under its timestamp dir (preserves newest_vib structure)
  # + the vib config; train.log optional
  VIBTS=$(basename "$(dirname "$VIBSRC")")
  mkdir -p "$S2/train/dyn/dyn-base/$VIBTS"
  copy_chk "$VIBSRC" "$S2/train/dyn/dyn-base/$VIBTS/scout_vib.ckpt"
  [ -f "$S1/train/dyn/dyn-base/config.yaml" ] && copy_chk "$S1/train/dyn/dyn-base/config.yaml" "$S2/train/dyn/dyn-base/config.yaml"
  for aux in config.yaml train.log; do
    [ -f "$S1/train/DP/DP-base/$aux" ] && { [ -f "$S2/train/DP/DP-base/$aux" ] \
      || cp -L "$S1/train/DP/DP-base/$aux" "$S2/train/DP/DP-base/"; }
  done
  log "s$s base ready: core=$(stat -c%s $S2/rollout/threading_core.hdf5)B dp=$(basename $DPNEW) dynbase=$VIBTS"
done

if [ "${COPY_ONLY:-0}" = 1 ]; then
  log "COPY_ONLY=1 -- base staged, not spawning"
  exit 0
fi

# --- B) spawn the 6 chains (GPU map: s233 G0/G1, s2333 G2/G3, s23333 G4/G5) --- #
g=0
for s in $SEEDS; do
  BE=$(eval "echo \${BASE_ETA_$s}")
  for ARM in DP ATY; do
    tmux new-session -d -s "thm3_${ARM,,}_s$s" \
      "cd $ROOT && SEED=$s GPU=$g ARM=$ARM NROUNDS=6 DATA_ROOT=$DST/THREADING-THM3-s$s WPROJ=THREADING-THM3-s$s ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5 BASE_ETA=$BE bash scripts/threading/thm3_chain.sh"
    log "spawned thm3_${ARM,,}_s$s gpu=$g base_eta=$BE"
    g=$((g+1))
  done
done
log "LAUNCH_DONE $(date '+%F %T')"
