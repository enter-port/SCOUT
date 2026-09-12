#!/bin/bash
set -uo pipefail
REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
DATADIR=/root/workspace/baojiachun/data/robomimic/transport/ph
LOG=/root/workspace/baojiachun/data/logs/transport_pipeline.log
TLOG=/root/workspace/baojiachun/data/logs/train_transport.log
GPU=3
exec >> "$LOG" 2>&1
echo "===== $(date) transport pipeline start (GPU=$GPU) ====="

echo "[$(date)] waiting for re-render process to exit..."
while pgrep -f "dataset_states_to_obs.*transport" >/dev/null 2>&1; do sleep 30; done
echo "[$(date)] re-render exited."

ND=$($PY -c "import h5py;print(len([k for k in h5py.File('$DATADIR/image_v141.hdf5','r')['data'] if k.startswith('demo_')]))" 2>/dev/null) || ND=0
echo "[$(date)] image_v141.hdf5 demos=$ND"
if [ "$ND" -ne 200 ]; then echo "ERROR: expected 200 demos, got $ND — re-render likely failed"; exit 1; fi

echo "[$(date)] DP conversion -> abs_actions..."
cd "$REPO"
$PY -m diffusion_policy.scripts.robomimic_dataset_conversion -i "$DATADIR/image_v141.hdf5" -o "$DATADIR/image_v141_abs.hdf5" -n 4
ADIM=$($PY -c "import h5py;print(h5py.File('$DATADIR/image_v141_abs.hdf5','r')['data/demo_0/abs_actions'].shape[1])" 2>/dev/null) || ADIM=0
echo "[$(date)] abs_actions dim=$ADIM"
if [ "$ADIM" -ne 14 ]; then echo "ERROR: expected abs_actions dim 14 (dual-arm), got $ADIM"; exit 1; fi

echo "[$(date)] core subset (interval 10 -> core_20)..."
$PY experiments/scripts/extract_core_subset.py "$DATADIR/image_v141_abs.hdf5" "$DATADIR/image_v141_abs_core20.hdf5" 10

echo "[$(date)] launching base DP training: GPU=$GPU wandb=SCOUT-baseDP-transport 600ep"
CUDA_VISIBLE_DEVICES=$GPU nohup $PY train.py --config-path configs --config-name base_dp_transport_image \
  task.dataset_path="$DATADIR/image_v141_abs_core20.hdf5" \
  logging.name=SCOUT-baseDP-transport logging.project=scout-base-dp \
  training.num_epochs=600 training.device=cuda:0 \
  > "$TLOG" 2>&1 &
TPID=$!
disown 2>/dev/null || true
echo "[$(date)] training launched PID=$TPID log=$TLOG"
echo "===== $(date) pipeline script done (training detached) ====="
