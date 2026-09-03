#!/bin/bash
# Prep for the CAN s233 orbit chain: mkdir + core rebuild (seeded 20-of-200
# split, same default_rng(TSEED) selection the DP-base was trained on) +
# base-trio symlinks from the can entropy campaign assets. Idempotent.
set -eu
ROOT=/root/workspace/baojiachun/scout-rand
R21=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233
T=$ROOT/data/2026_9_2_orbchain/ORBIT-s233/can
PY=/root/workspace/baojiachun/.venv/bin/python

mkdir -p "$T/rollout" "$T/train/DP/DP-base/checkpoints" "$T/train/dyn/dyn-base"
if [ ! -f "$T/rollout/can_core.hdf5" ]; then
  $PY "$ROOT/soe_scripts/split_core.py" \
    /root/workspace/baojiachun/scout/data/robomimic/can/ph/image_v141_abs.hdf5 \
    "$T/rollout/can_core.hdf5" 20 233
fi
ln -sf "$R21/can/train/DP/DP-base/checkpoints/599.ckpt" \
      "$T/train/DP/DP-base/checkpoints/599.ckpt"
ln -sfn "$R21/can/train/dyn/dyn-base/20260824-232156" \
       "$T/train/dyn/dyn-base/20260824-232156"

$PY - <<'PYEOF'
import h5py
f = h5py.File("/root/workspace/baojiachun/scout-rand/data/2026_9_2_orbchain/ORBIT-s233/can/rollout/can_core.hdf5")
demos = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[-1]))
steps = sum(f[f"data/{d}/actions"].shape[0] for d in demos)
print(f"CORE-CHECK demos={len(demos)} steps={steps}")
PYEOF
echo "prep can-s233 done"
