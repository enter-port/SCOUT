#!/bin/bash
# can_aty_launch.sh -- CAN-aty-try1-s233 launch orchestrator (2026-09-03
# user order). CAN-9-2-orbit-s233 (single SCOUT arm, seed 233, 6 rounds with
# r6 eval-only) with THREE deltas:
#   1. guidance = orbit phase-1 climb ONLY (= the atypical cost): round_aty.sh
#      runs GUIDE=orbit with lam=0/delta=0/sigma=0, which the code documents
#      as bit-identical to --guide atypical; the 9-2 phase-1 dose is kept
#      verbatim (eta_tilde 0.33 dimless, kappa ATT_CAP=2.5, gst 100 from the
#      can eval config). The bare --guide atypical path has no eta-dimless
#      switch, so the phase-1 dose cannot be carried by it.
#   2. pass@10 -> pass@1: ETRIES=1 -- each failed eval init is tried ONCE.
#   3. explore worker = 1: SHARD_P=1 (x 25 envs per worker).
# Round0 assets (core + DP-base final ckpt + dyn-base) are COPIED from the
# CAN-9-2-orbit-s233 root: same seed 233 and deterministic round0 make them
# the training-identical weights, skipping a redundant 600ep base retrain.
# GPU0 is idle (can_orb_v3 finished 2026-09-03); this campaign does not
# touch any running chain (fresh DATA_ROOT, own wandb project).
set -u
ROOT=/root/workspace/baojiachun/scout-orbit
SRC=$ROOT/data/2026_9_2_orbchain/ORBIT-s233/can
DST=$ROOT/data/2026_9_3_atytry1/ATY-s233/can
cd "$ROOT" || exit 1   # relative soe_scripts/aty_chain.sh call below
mkdir -p "$DST/rollout" "$DST/train/DP" "$DST/train/dyn"
unset DATA_ROOT   # stale export must not redirect the real campaign
export NROUNDS=6 SHARD_P=1 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=1

echo "[launch] seeding round0 assets from CAN-9-2 (same seed 233, deterministic) $(date '+%F %T')"
[ -f "$DST/rollout/can_core.hdf5" ] || cp "$SRC/rollout/can_core.hdf5" "$DST/rollout/"
mkdir -p "$DST/train/DP/DP-base/checkpoints"
[ -f "$DST/train/DP/DP-base/checkpoints/599.ckpt" ] || cp -L "$SRC/train/DP/DP-base/checkpoints/599.ckpt" "$DST/train/DP/DP-base/checkpoints/599.ckpt"
VIBTS=$(ls "$SRC/train/dyn/dyn-base/" | head -1)
mkdir -p "$DST/train/dyn/dyn-base/$VIBTS"
for f in scout_vib.ckpt config.yaml; do
  [ -f "$DST/train/dyn/dyn-base/$VIBTS/$f" ] || cp "$SRC/train/dyn/dyn-base/$VIBTS/$f" "$DST/train/dyn/dyn-base/$VIBTS/"
done
echo "[launch] assets ready: core=$(du -h "$DST/rollout/can_core.hdf5" | cut -f1) dp599=$(du -h "$DST/train/DP/DP-base/checkpoints/599.ckpt" | cut -f1) vib=$VIBTS"

echo "[launch] round0 START $(date '+%F %T') (GPU0, seed 233)"
SEED=233 GPU=0 ARM=BASE bash soe_scripts/aty_chain.sh
rc=$?
echo "[launch] round0 rc=$rc $(date '+%F %T')"
if [ $rc -ne 0 ]; then
  echo "[launch] round0 FAILED -- arm NOT started"
  exit 1
fi

tmux new-session -d -s can_aty_try1_s233 \
  "cd $ROOT && SEED=233 GPU=0 ARM=SCOUT NROUNDS=6 ETRIES=1 SHARD_P=1 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 bash soe_scripts/aty_chain.sh"
echo "[launch] arm spawned: can_aty_try1_s233@GPU0 $(date '+%F %T')"
