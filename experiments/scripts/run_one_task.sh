#!/usr/bin/env bash
set -euo pipefail
task=$1; interval=$2; gpu=$3
export MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=$gpu OMP_NUM_THREADS=4
VPY=/root/workspace/baojiachun/.venv/bin/python
REPO=/root/workspace/baojiachun/scout
DATA=/root/workspace/baojiachun/data/robomimic
SUBSET=/root/workspace/baojiachun/soe_scripts/extract_core_subset.py
LOG=$DATA/${task}.log
exec > >(tee -a "$LOG") 2>&1
d="$DATA/$task/ph"; mkdir -p "$d"
low=$d/low_dim_v141.hdf5; img=$d/image_v141.hdf5; abs=$d/image_v141_abs.hdf5
n=$((200/interval)); core=$d/image_v141_abs_core${n}.hdf5
echo "=== $task interval=$interval core_$n gpu=$gpu START $(date) ==="
[ -f "$low" ] || wget -q --tries=3 --timeout=60 -O "$low" "http://downloads.cs.stanford.edu/downloads/rt_benchmark/$task/ph/low_dim_v141.hdf5"
[ -f "$img" ] || ( cd "$d" && "$VPY" -m robomimic.scripts.dataset_states_to_obs --done_mode 0 --dataset "$low" --output_name image_v141.hdf5 --camera_names agentview robot0_eye_in_hand --camera_height 84 --camera_width 84 )
[ -f "$abs" ] || ( cd "$REPO" && "$VPY" -m diffusion_policy.scripts.robomimic_dataset_conversion -i "$img" -o "$abs" -n 4 )
"$VPY" "$SUBSET" "$abs" "$core" "$interval"
echo "=== $task verify ==="
"$VPY" - "$core" << 'PYV'
import sys, h5py
f=h5py.File(sys.argv[1],'r'); demos=sorted(f['data'].keys(), key=lambda k:int(k.split('_')[-1])); demo=f['data'][demos[0]]
print(' demos=',len(demos),'| abs_actions=',demo['abs_actions'].shape,'| agentview=',demo['obs/agentview_image'].shape)
f.close()
PYV
echo "=== $task DONE $(date) -> $core ==="
