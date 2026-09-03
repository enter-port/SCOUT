#!/bin/bash
# CAN s233 orbit-native 6-round chain (user order 2026-09-02; replica of the
# SQUARE s233 sq_orb_chain.sh on task can, GPU0).
# Best orbit params: sigma=0.25 kappa=2.5 lam=0.5 delta=0.25 (sector=iid, climb=grad).
# Base trio = 2026_8_21_entropy can assets (DP-base 599.ckpt + dyn-base
# 20260824-232156 + core rebuilt via split_core.py n=20 seed=233), linked/copied
# into the chain layout -- NO round0. Rounds 1-6: rollout(eval seed42 x100 +
# rescue x10 orbit) -> DP retrain 300ep -> dyn retrain 100ep, per SOE budget.
# Usage: GPU=<id> bash /tmp/can_orb_chain.sh
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
TSEED=233
ROOT=/root/workspace/baojiachun/scout-rand
R21=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233
DATA_ROOT=$ROOT/data/2026_9_2_orbchain/ORBIT-s233
T=$DATA_ROOT/can
WPROJ=CAN-9-2-orbit-s233

mkdir -p "$T/rollout" "$T/train/DP/DP-base/checkpoints" "$T/train/dyn/dyn-base"
[ -f "$T/rollout/can_core.hdf5" ] || /root/workspace/baojiachun/.venv/bin/python "$ROOT/soe_scripts/split_core.py" \
  /root/workspace/baojiachun/scout/data/robomimic/can/ph/image_v141_abs.hdf5 \
  "$T/rollout/can_core.hdf5" 20 233
ln -sf "$R21/can/train/DP/DP-base/checkpoints/599.ckpt" \
      "$T/train/DP/DP-base/checkpoints/599.ckpt"
ln -sfn "$R21/can/train/dyn/dyn-base/20260824-232156" \
       "$T/train/dyn/dyn-base/20260824-232156"

cd "$ROOT" || exit 1
for N in 1 2 3 4 5 6; do
  echo "[chain] round $N START $(date '+%F %T')"
  GPU=$GPU TSEED=$TSEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ ATT_CAP=2.5 \
    bash soe_scripts/round_orbit.sh can SCOUT $N
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $T/round.log)"; exit $rc; }
done
echo "[chain] ALL 6 ROUNDS DONE $(date '+%F %T')"
