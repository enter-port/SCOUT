#!/bin/bash
# tool_hang data pipeline: download low_dim -> re-render 84x84 -> abs conversion -> core_20
# Same recipe as can/transport. Camera set per LPB/DP tool_hang config: sideview + robot0_eye_in_hand.
# All steps idempotent (skip if output already valid), log to scout/data/logs/toolhang_pipeline.log.
set -uo pipefail

REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
DATADIR=/root/workspace/baojiachun/scout/data/robomimic/tool_hang/ph
LOGDIR=/root/workspace/baojiachun/scout/data/logs
LOG=$LOGDIR/toolhang_pipeline.log
URL=http://downloads.cs.stanford.edu/downloads/rt_benchmark/tool_hang/ph/low_dim_v141.hdf5
GPUEGL=2   # GPU0/4 busy (square orbit chain); GPU7 ECC-defective, banned

export TMPDIR=/tmp
export MUJOCO_GL=egl
export SCOUT_RENDER_GPU=$GPUEGL

mkdir -p "$DATADIR" "$LOGDIR"
exec >> "$LOG" 2>&1
echo "===== $(date) tool_hang data pipeline start (EGL device=$GPUEGL) ====="

ndemos() { $PY -c "import h5py,sys;print(len([k for k in h5py.File(sys.argv[1],'r')['data'] if k.startswith('demo_')]))" "$1" 2>/dev/null; }

# ---------- [1/4] download low_dim ----------
if [ -s "$DATADIR/low_dim_v141.hdf5" ]; then
  echo "[$(date)] low_dim exists ($(stat -c%s "$DATADIR/low_dim_v141.hdf5") bytes), skip"
else
  echo "[$(date)] downloading $URL ..."
  curl -fsSL --retry 3 -o "$DATADIR/low_dim_v141.hdf5.part" "$URL" || { echo "[$(date)] ERROR: download failed"; exit 1; }
  mv "$DATADIR/low_dim_v141.hdf5.part" "$DATADIR/low_dim_v141.hdf5"
  echo "[$(date)] download done: $(stat -c%s "$DATADIR/low_dim_v141.hdf5") bytes"
fi
ND=$(ndemos "$DATADIR/low_dim_v141.hdf5")
echo "[$(date)] low_dim demos=$ND"
[ "$ND" != "200" ] && { echo "ERROR: expected 200 demos, got $ND"; exit 1; }
$PY -c "
import h5py
f=h5py.File('$DATADIR/low_dim_v141.hdf5','r')
d=f['data']
print('env:', dict(d.attrs).get('env','?'))
print('actions shape:', d['demo_0/actions'].shape)
print('obs keys:', sorted(d['demo_0/obs'].keys()))
" || exit 1

# ---------- [2/4] re-render 84x84 ----------
NR=$(ndemos "$DATADIR/image_v141.hdf5")
if [ "$NR" = "200" ]; then
  echo "[$(date)] image_v141.hdf5 already has 200 demos, skip render"
else
  echo "[$(date)] re-render (sideview + robot0_eye_in_hand, 84x84) ..."
  $PY -m robomimic.scripts.dataset_states_to_obs --done_mode 0 \
      --dataset "$DATADIR/low_dim_v141.hdf5" \
      --output_name image_v141.hdf5 \
      --camera_names sideview robot0_eye_in_hand \
      --camera_height 84 --camera_width 84 \
      || { echo "[$(date)] ERROR: render failed"; exit 1; }
fi
NR=$(ndemos "$DATADIR/image_v141.hdf5")
echo "[$(date)] image demos=$NR"
[ "$NR" != "200" ] && { echo "ERROR: render incomplete ($NR/200)"; exit 1; }
$PY -c "
import h5py
obs=h5py.File('$DATADIR/image_v141.hdf5','r')['data/demo_0/obs']
ks=sorted(obs.keys())
print('image obs keys:', ks)
assert 'sideview_image' in ks and 'robot0_eye_in_hand_image' in ks, 'missing camera key'
print('sideview shape/dtype:', obs['sideview_image'].shape, obs['sideview_image'].dtype)
" || exit 1

# ---------- [3/4] abs conversion ----------
if [ -s "$DATADIR/image_v141_abs.hdf5" ] && [ "$(ndemos "$DATADIR/image_v141_abs.hdf5")" = "200" ]; then
  echo "[$(date)] image_v141_abs.hdf5 valid, skip conversion"
else
  echo "[$(date)] DP conversion -> abs_actions ..."
  cd "$REPO" || exit 1
  $PY -m diffusion_policy.scripts.robomimic_dataset_conversion \
      -i "$DATADIR/image_v141.hdf5" -o "$DATADIR/image_v141_abs.hdf5" -n 4 \
      || { echo "[$(date)] ERROR: conversion failed"; exit 1; }
fi
$PY -c "
import h5py
src=h5py.File('$DATADIR/image_v141.hdf5','r')['data/demo_0/actions'].shape[1]
ab=h5py.File('$DATADIR/image_v141_abs.hdf5','r')['data/demo_0/abs_actions'].shape[1]
print('actions dim=%s abs_actions dim=%s' % (src, ab))
assert ab==src and ab>0, 'abs dim mismatch'
" || exit 1

# ---------- [4/4] core subset ----------
if [ -s "$DATADIR/image_v141_abs_core20.hdf5" ] && [ "$(ndemos "$DATADIR/image_v141_abs_core20.hdf5")" = "20" ]; then
  echo "[$(date)] core20 valid, skip"
else
  echo "[$(date)] core subset (interval 10 -> 20 demos) ..."
  cd "$REPO" || exit 1
  $PY experiments/scripts/extract_core_subset.py "$DATADIR/image_v141_abs.hdf5" "$DATADIR/image_v141_abs_core20.hdf5" 10 \
      || { echo "[$(date)] ERROR: core subset failed"; exit 1; }
fi
NC=$(ndemos "$DATADIR/image_v141_abs_core20.hdf5")
echo "[$(date)] core20 demos=$NC"
[ "$NC" != "20" ] && { echo "ERROR: core20 incomplete"; exit 1; }

echo "===== $(date) TOOLHANG DATA PIPELINE DONE ====="
echo "artifacts: $DATADIR/{low_dim_v141,image_v141,image_v141_abs,image_v141_abs_core20}.hdf5"
