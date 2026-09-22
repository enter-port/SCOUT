#!/bin/bash
# thm2_dryrun_smoke.sh -- server-side DRY_RUN smoke for THREADING-MG-p2.
# Builds a SCRATCH data root in /tmp (symlinked core+DP-base, dummy dyn-base
# vib file) so the real campaign tree stays untouched, then dry-runs the ATY
# and DP round-1 paths and prints the load-bearing lines.
set -u
cd /root/workspace/baojiachun/scout || exit 1
D=/root/workspace/baojiachun/scout/data/2026_9_22_threading_mg_p2/THREADING-MG-p2-s233
S=/tmp/thm2_dryrun/THREADING-MG-p2-s233
rm -rf /tmp/thm2_dryrun
mkdir -p "$S/threading/rollout" "$S/threading/train/DP/DP-base/checkpoints" \
         "$S/threading/train/dyn/dyn-base/DRYRUN"
ln -sf "$D/threading/rollout/threading_core.hdf5" "$S/threading/rollout/threading_core.hdf5"
ln -sf "$D/threading/train/DP/DP-base/checkpoints/599.ckpt" "$S/threading/train/DP/DP-base/checkpoints/599.ckpt"
: > "$S/threading/train/dyn/dyn-base/DRYRUN/scout_vib.ckpt"   # dummy: DRY_RUN never reads it
export DRY_RUN=1

echo "===== [1] ATY round 1 (expect: calib DRY_RUN plan, dose 1.0/2.5) ====="
GPU=1 TSEED=233 DATA_ROOT=$S WPROJ=THREADING-MG-p2-s233 ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5 \
  bash scripts/threading/round_thm2.sh threading ATY 1 full 2>&1 \
  | grep -E "calib-rcalib|=== ROUND|DRY_RUN:.*run_rollout|atypical-cap|guidance-scale|shard_rollout" | head -12

echo "===== [2] DP round 1 (expect: no calib; dose line has no aty_scale) ====="
GPU=0 TSEED=233 DATA_ROOT=$S WPROJ=THREADING-MG-p2-s233 ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5 \
  bash scripts/threading/round_thm2.sh threading DP 1 full 2>&1 \
  | grep -E "calib-rcalib|=== ROUND|guide off|--guide" | head -8

echo "===== [3] ATY round 2 walk-back + prev-json read (expect WARN fallback) ====="
GPU=1 TSEED=233 DATA_ROOT=$S WPROJ=THREADING-MG-p2-s233 ETA0=1.0 ATT_CAP=2.5 BETA=1.0e-5 ETRIES=5 \
  bash scripts/threading/round_thm2.sh threading ATY 2 full 2>&1 \
  | grep -E "calib-rcalib|=== ROUND" | head -6

echo "===== [4] staged real tree: dyn-base config beta + core/DP sizes ====="
grep -E "^beta:|^num_epochs:|^seed:" "$D/threading/train/dyn/dyn-base/config.yaml" 2>/dev/null
echo "round.log bytes: $(stat -c%s $D/threading/round.log 2>/dev/null || echo absent)"
echo "===== done ====="
