#!/bin/bash
# transport experiment2 bootstrap (user 2026-08-19): round0 from scratch --
# DP-base (600ep on core_20) then dyn-base (300ep, beta=3e-5) -- then launch
# the two 6-round loops (DP@GPU2, SCOUT@GPU0; last round eval-only).
set -u
cd /root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
LOG=data/experiment2/transport/logs
export MUJOCO_GL=egl
export TMPDIR=/tmp
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG/round0.log"; }

say '=== transport round0 [1/2] DP-base 600ep on core_20 (GPU0) START ==='
CUDA_VISIBLE_DEVICES=0 $PY train.py   --config-path configs --config-name base_dp_transport_image   task.dataset_path=data/experiment2/transport/rollout/transport_core.hdf5   training.rollout_every=0   training.sample_every=100   training.num_epochs=600   training.resume=False   training.checkpoint_every=100   training.cudnn_benchmark=true   dataloader.num_workers=8 dataloader.persistent_workers=true   logging.name=DP-base logging.project=TRANSPORT-experiment2   hydra.run.dir=data/experiment2/transport/train/DP/DP-base   > "$LOG/dp_base.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  say 'DP-base workers=8 failed -- retry with num_workers=0'
  CUDA_VISIBLE_DEVICES=0 $PY train.py     --config-path configs --config-name base_dp_transport_image     task.dataset_path=data/experiment2/transport/rollout/transport_core.hdf5     training.rollout_every=0     training.sample_every=100     training.num_epochs=600     training.resume=False     training.checkpoint_every=100     training.cudnn_benchmark=true     dataloader.num_workers=0     logging.name=DP-base logging.project=TRANSPORT-experiment2     hydra.run.dir=data/experiment2/transport/train/DP/DP-base     >> "$LOG/dp_base.log" 2>&1
  RC=$?
fi
say "=== round0 [1/2] DP-base rc=$RC ==="
[ $RC -ne 0 ] && { say 'DP-base FAILED - see dp_base.log'; exit 1; }

say '=== transport round0 [2/2] dyn-base 300ep beta=3e-5 (GPU0) START ==='
CUDA_VISIBLE_DEVICES=0 $PY -m scout.train_vib   --config configs/vib_transport_image_e2.yaml   > "$LOG/dyn_base.log" 2>&1
RC=$?
say "=== round0 [2/2] dyn-base rc=$RC ==="
[ $RC -ne 0 ] && { say 'dyn-base FAILED - see dyn_base.log'; exit 1; }

tmux new-session -d -s e2loop_transport_DP "cd /root/workspace/baojiachun/scout && GPU=2 bash soe_scripts/run_rounds.sh transport DP round_e2.sh 6 2>&1 | tee data/experiment2/logs/e2loop_transport_DP.console.log"
tmux new-session -d -s e2loop_transport_SCOUT "cd /root/workspace/baojiachun/scout && GPU=0 bash soe_scripts/run_rounds.sh transport SCOUT round_e2.sh 6 2>&1 | tee data/experiment2/logs/e2loop_transport_SCOUT.console.log"
say '=== round0 complete -- launched e2loop_transport_{DP,SCOUT} (6 rounds, r6 eval-only) ==='
