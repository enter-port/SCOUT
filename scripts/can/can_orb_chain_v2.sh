#!/bin/bash
# CAN s233 orbit-native 6-round chain V2 (user order 2026-09-02 19:xx):
# re-run on the orbit-dev SHARDED worker code -- rounds 1-5 full, round 6
# EVAL-ONLY (success-rate measurement of the round-5 policy, no explore, no
# retrains). Replaces the monolithic first attempt (r1-r5 completed, r6 was
# running in FULL mode when superseded); its products remain untouched under
# scout-rand/data/2026_9_2_orbchain/.
# Differences from /tmp/can_orb_chain.sh (v1):
#   * ROOT = scout-orbit (orbit-dev tree with shard_rollout.sh/heartbeat);
#     DATA_ROOT = fresh dir under scout-orbit/data (no collision with v1);
#   * round_orbit.sh = sharded two-phase [1/3]: phase A eval 25-env
#     monolithic -> failed.json; phase B SHARD_P=2 x 25-env workers with
#     RAM gate 500G + global phase-B flock + wandb heartbeat;
#   * round 6 runs with mode arg "eval-only".
# Orbit params unchanged: sigma=0.25 kappa=2.5 lam=0.5 delta=0.25
# (sector=iid, climb=grad). Base trio = 2026_8_21_entropy can assets
# (DP-base 599.ckpt + dyn-base 20260824-232156 + core rebuilt via
# split_core.py n=20 seed=233), NO round0.
# Usage: GPU=<id> bash /tmp/can_orb_chain_v2.sh
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
TSEED=233
ROOT=/root/workspace/baojiachun/scout-orbit
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
  MODE=full; [ "$N" = 6 ] && MODE=eval-only
  echo "[chain] round $N ($MODE) START $(date '+%F %T')"
  GPU=$GPU TSEED=$TSEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ ATT_CAP=2.5 \
    bash soe_scripts/round_orbit.sh can SCOUT "$N" "$MODE"
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $T/round.log)"; exit $rc; }
done
echo "[chain] ALL 6 ROUNDS DONE (r6 eval-only) $(date '+%F %T')"
