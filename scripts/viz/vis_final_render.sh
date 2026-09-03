#!/bin/bash
# vis_final_render.sh -- render trajviz pngs for the vis_final rollout dirs.
# For every selected dir with an all.hdf5 and no trajviz.png yet, run
# scout.eval.visualize_trajectories on the 20 APPENDED rollout demos
# (--min-demo-id 20; core demos occupy ids 0..19).
# Usage: GPU=<id> bash soe_scripts/vis_final_render.sh [dir ...]
#        (default: every data/vis_final/exp*/g*/ dir)
set -u
GPU=${GPU:?set GPU=<cuda id>}
export MUJOCO_GL=egl TMPDIR=/tmp
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
cd "$REPO" || exit 1

DIRS=("$@")
if [ ${#DIRS[@]} -eq 0 ]; then
  mapfile -t DIRS < <(ls -d "$REPO"/data/vis_final/exp*/g*/ 2>/dev/null)
fi
for d in "${DIRS[@]}"; do
  h="${d%/}/all.hdf5"
  [ -f "$h" ] || continue
  if [ -f "${d%/}/trajviz.png" ]; then
    echo "[render] skip ${d%/}"
    continue
  fi
  echo "[render] $h $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl TMPDIR=/tmp "$PY" \
    -m scout.eval.visualize_trajectories \
    "$h" --min-demo-id 20 --out "${d%/}/trajviz.png" \
    > "${d%/}/trajviz.stdout" 2>&1 \
    || { echo "[render] FAILED ${d%/}:"; tail -n 5 "${d%/}/trajviz.stdout"; }
done
echo "[render] done $(date '+%F %T')"
