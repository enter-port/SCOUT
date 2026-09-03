#!/bin/bash
# SQUARE s2333 orbit-native 6-round chain (user order 2026-09-02; replica of
# s233 sq_orb_chain.sh with seed swapped, GPU0).
# Best orbit params: sigma=0.25 kappa=2.5 lam=0.5 delta=0.25 (sector=iid, climb=grad).
# Base trio = 2026_8_26_entropy assets (DP-base 599.ckpt + dyn-base
# 20260826-112147 + core_rebuild/square_core_s2333.hdf5), linked/copied into
# the chain layout -- NO round0. Rounds 1-6: rollout(eval seed42 x100 +
# rescue x10 orbit) -> DP retrain 300ep -> dyn retrain 100ep, per SOE budget.
# Usage: GPU=<id> bash /tmp/sq_orb_chain_2333.sh
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
TSEED=2333
ROOT=/root/workspace/baojiachun/scout-rand
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
DATA_ROOT=$ROOT/data/2026_9_1_orbchain/ORBIT-s2333
T=$DATA_ROOT/square
WPROJ=SQUARE-9-1-orbit-s2333

mkdir -p "$T/rollout" "$T/train/DP/DP-base/checkpoints" "$T/train/dyn/dyn-base"
[ -f "$T/rollout/square_core.hdf5" ] || cp "$R26/core_rebuild/square_core_s2333.hdf5" "$T/rollout/square_core.hdf5"
ln -sf "$R26/SQUARE-entropy-s2333/square/train/DP/DP-base/checkpoints/599.ckpt" \
      "$T/train/DP/DP-base/checkpoints/599.ckpt"
ln -sfn "$R26/SQUARE-entropy-s2333/square/train/dyn/dyn-base/20260826-112147" \
       "$T/train/dyn/dyn-base/20260826-112147"

cd "$ROOT" || exit 1
for N in 1 2 3 4 5 6; do
  echo "[chain] round $N START $(date '+%F %T')"
  GPU=$GPU TSEED=$TSEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ ATT_CAP=2.5 \
    bash soe_scripts/round_orbit.sh square SCOUT $N
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $T/round.log)"; exit $rc; }
done
echo "[chain] ALL 6 ROUNDS DONE $(date '+%F %T')"
