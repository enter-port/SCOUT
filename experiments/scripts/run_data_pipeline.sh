#!/usr/bin/env bash
set -euo pipefail
LOG=/root/workspace/baojiachun/data/PIPELINE.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "================ PIPELINE START $(date) ================"
export MUJOCO_GL=egl
export OMP_NUM_THREADS=4
VPY=/root/workspace/baojiachun/.venv/bin/python
REPO=/root/workspace/baojiachun/scout
DATA=/root/workspace/baojiachun/data/robomimic
SUBSET=/root/workspace/baojiachun/soe_scripts/extract_core_subset.py
NWORK=4
df -h /root/workspace | sed 's/^/[df] /'

run_task () {
  local task=$1 interval=$2
  local d="$DATA/$task/ph"; mkdir -p "$d"
  local low="$d/low_dim_v141.hdf5"
  local img="$d/image_v141.hdf5"
  local abs="$d/image_v141_abs.hdf5"
  local n=$(( 200 / interval ))
  local core="$d/image_v141_abs_core${n}.hdf5"
  echo "------ TASK=$task interval=$interval core_$n ------"
  if [ ! -f "$low" ]; then
    echo "[download] $task"
    wget -q --tries=3 --timeout=60 -O "$low" "http://downloads.cs.stanford.edu/downloads/rt_benchmark/$task/ph/low_dim_v141.hdf5"
  fi
  ls -la "$low"
  if [ ! -f "$img" ]; then
    echo "[re-render] $task"
    ( cd "$d" && "$VPY" -m robomimic.scripts.dataset_states_to_obs --done_mode 0 --dataset "$low" --output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 )
  fi
  ls -la "$img"
  if [ ! -f "$abs" ]; then
    echo "[convert] $task"
    ( cd "$REPO" && "$VPY" -m diffusion_policy.scripts.robomimic_dataset_conversion -i "$img" -o "$abs" -n "$NWORK" )
  fi
  ls -la "$abs"
  echo "[subset] $task -> core_$n"
  "$VPY" "$SUBSET" "$abs" "$core" "$interval"
  ls -la "$core"
  echo "[verify] $task"
  "$VPY" - "$core" << 'PYV'
import sys, h5py
f=h5py.File(sys.argv[1],'r')
d=f['data']; demos=sorted(d.keys(), key=lambda k:int(k.split('_')[-1]))
demo=d[demos[0]]
print('  demos=',len(demos),'| abs_actions=',demo['abs_actions'].shape,'| actions=',demo['actions'].shape,'| agentview=',demo['obs/agentview_image'].shape)
print('  env_args=',str(f['data'].attrs.get('env_args',''))[:140])
f.close()
PYV
  echo "[done] $task -> $core"
}
run_task lift 20
run_task can 10
run_task square 10
echo "================ PIPELINE DONE $(date) ================"
