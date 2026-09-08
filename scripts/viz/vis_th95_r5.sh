#!/bin/bash
# vis_th95_r5.sh -- 3-arm 3x3 grid trajectory VIDEO, TOOLHANG-9-5-orbit-s233
# round5-final ckpts (user order 2026-09-08). Runs from the scout-viz
# extract (plain extract of orbit-dev @ the commit carrying this file).
#
#   rows    : DP (guide off) / SCOUT (atypical) / SCOUT-orbit (orbit)
#   columns : sideview / topdown / eye_in_hand (tool_hang has NO agentview)
#
# ckpts = each arm's exp5 (the last trained round; r6 was eval-only) and the
# parameter set is the campaign's own r6-eval invocation (orbit_round 6 =>
# sigma_eff = 0.05 * 0.5^5). GPU auto-picked from the idle set {1,3,5}
# (0/2/4 = s2333 campaign arms, 6 = SOE baseline, 7 = ECC FORBIDDEN);
# pin with GPU=<id>. Disk guard: res downgraded to 384 when < 32 GiB free.
#
# Usage:  bash scripts/viz/vis_th95_r5.sh
set -euo pipefail

REPO=/root/workspace/baojiachun/scout-viz
PY=/root/workspace/baojiachun/.venv/bin/python
ROOT=/root/workspace/baojiachun/scout-th95/data/2026_9_5_toolhang/TOOLHANG-s233
DATA=$ROOT/tool_hang
OUT=$ROOT/vis_r5_grid
cd "$REPO"

if [ -z "${GPU:-}" ]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, '$1==1||$1==3||$1==5 {print $2, $1}' | sort -n | head -1 \
    | awk '{print $2}')
fi
[ -n "${GPU:-}" ] || { echo "[vis] FATAL: no GPU in {1,3,5}"; exit 1; }
AVAIL_KIB=$(df -P /root/workspace | tail -1 | awk '{print $4}')
RES=480
[ "$AVAIL_KIB" -lt 33554432 ] && RES=384
echo "[vis] GPU=$GPU res=$RES df_avail=${AVAIL_KIB}KiB out=$OUT $(date '+%F %T')"

ATYV=$(ls -t "$DATA"/train/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)
ORBV=$(ls -t "$DATA"/train/dyn/dyn-ORBIT-exp5/*/scout_vib.ckpt | head -1)

mkdir -p "$OUT"
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp \
  "$PY" -m scout.eval.visualize_arms_video \
    --config configs/eval_tool_hang_entropy.yaml --task tool_hang \
    --core-hdf5 "$DATA/rollout/tool_hang_core.hdf5" \
    --dp-ckpt  "$DATA/train/DP/DP-DP-exp5/checkpoints/299.ckpt" \
    --aty-ckpt "$DATA/train/DP/DP-ATY-exp5/checkpoints/299.ckpt" \
    --aty-vib  "$ATYV" \
    --orb-ckpt "$DATA/train/DP/DP-ORBIT-exp5/checkpoints/299.ckpt" \
    --orb-vib  "$ORBV" \
    --guidance-scale 0.5 --atypical-cap 2.5 \
    --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.05 \
    --orbit-sigma-decay 0.5 --orbit-noise-anneal 2 --orbit-round 6 \
    --orbit-fb-clamp soft \
    --stock-camera sideview --max-steps 700 --max-attempts 40 \
    --seed 42 --res "$RES" \
    --out-dir "$OUT" 2>&1 | tee "$OUT/run.log"
echo "[vis] DONE $(date '+%F %T')"
