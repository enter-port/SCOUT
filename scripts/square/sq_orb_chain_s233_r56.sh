#!/bin/bash
# SQUARE s233 orbit chain RESUME for rounds 5-6 (user order 2026-09-02:
# round-5 explore was killed by a host memory blast (0 exploration
# successes), products + wandb run purged, redo from round 5).
# Identical to /tmp/sq_orb_chain.sh except the loop starts at 5.
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
TSEED=233
ROOT=/root/workspace/baojiachun/scout-rand
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
DATA_ROOT=$ROOT/data/2026_9_1_orbchain/ORBIT-s233
T=$DATA_ROOT/square
WPROJ=SQUARE-9-1-orbit-s233

mkdir -p "$T/rollout" "$T/train/DP/DP-base/checkpoints" "$T/train/dyn/dyn-base"
[ -f "$T/rollout/square_core.hdf5" ] || cp "$R26/core_rebuild/square_core_s233.hdf5" "$T/rollout/square_core.hdf5"
ln -sf "$R26/SQUARE-entropy-s233/square/train/DP/DP-base/checkpoints/599.ckpt" \
      "$T/train/DP/DP-base/checkpoints/599.ckpt"
ln -sfn "$R26/SQUARE-entropy-s233/square/train/dyn/dyn-base/20260826-112119" \
       "$T/train/dyn/dyn-base/20260826-112119"

cd "$ROOT" || exit 1
for N in 5 6; do
  echo "[chain] round $N START $(date '+%F %T')"
  GPU=$GPU TSEED=$TSEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ ATT_CAP=2.5 \
    bash soe_scripts/round_orbit.sh square SCOUT $N
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $T/round.log)"; exit $rc; }
done
echo "[chain] ALL ROUNDS DONE $(date '+%F %T')"
