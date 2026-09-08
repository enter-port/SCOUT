#!/bin/bash
# th9_4_round0_launch.sh -- TOOLHANG-9-4-orbit-s233 round0-only launcher
# (2026-09-04 user order via /launch-experiment): train base DP 600ep +
# dyn-base on a 40-demo core (user: 40 demos, NOT the standard 20), wandb
# project TOOLHANG-9-4-orbit-s233; NO chain arms after round0.
# Core is pre-split with split_core.py (n=40, seed 233) so the BASE arm of
# round_orbit_th.sh skips the split and trains directly on the 40-demo core
# (zero script changes: WPROJ and DATA_ROOT are env-overridable).
set -u
ROOT=/root/workspace/baojiachun/scout-orbit
DROOT=$ROOT/data/2026_9_4_toolhang/TOOLHANG-s233
cd "$ROOT" || exit 1
unset DATA_ROOT   # stale export must not redirect this campaign
mkdir -p "$DROOT/tool_hang/rollout"
exec >> "$DROOT/round0.console.log" 2>&1

CORE=$DROOT/tool_hang/rollout/tool_hang_core.hdf5
OFFICIAL=/root/workspace/baojiachun/scout/data/robomimic/tool_hang/ph/image_v141_abs.hdf5
PY=/root/workspace/baojiachun/.venv/bin/python

if [ ! -f "$CORE" ]; then
  echo "[launch] splitting 40-of-200 demos (seed 233) -> $CORE $(date '+%F %T')"
  $PY soe_scripts/split_core.py "$OFFICIAL" "$CORE" 40 233 \
    || { echo "[launch] SPLIT FAILED"; exit 1; }
fi
echo "[launch] core: $(du -h "$CORE" | cut -f1)"

echo "[launch] round0 START $(date '+%F %T') (GPU0, DP 600ep + dyn-base, WPROJ=TOOLHANG-9-4-orbit-s233)"
GPU=0 TSEED=233 DATA_ROOT=$DROOT WPROJ=TOOLHANG-9-4-orbit-s233 \
  bash soe_scripts/round_orbit_th.sh tool_hang BASE 0
rc=$?
echo "[launch] round0 rc=$rc $(date '+%F %T')"
