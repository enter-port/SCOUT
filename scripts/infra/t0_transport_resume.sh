#!/bin/bash
# transport round0 resume (2026-08-19): DP-base done (599.ckpt); config had a
# stale 580.ckpt reference -> dyn-base died at E_s load. Retrain dyn-base then
# launch the two 6-round loops (DP@GPU2, SCOUT@GPU0; r6=eval-only).
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

say '=== round0-retry [2/2] dyn-base 300ep beta=3e-5 (GPU0) START ==='
CUDA_VISIBLE_DEVICES=0 $PY -m scout.train_vib   --config configs/vib_transport_image_e2.yaml   > "$LOG/dyn_base.log" 2>&1
RC=$?
say "=== round0-retry [2/2] dyn-base rc=$RC ==="
[ $RC -ne 0 ] && { say 'dyn-base FAILED - see dyn_base.log'; exit 1; }

tmux new-session -d -s e2loop_transport_DP "cd /root/workspace/baojiachun/scout && GPU=2 bash soe_scripts/run_rounds.sh transport DP round_e2.sh 6 2>&1 | tee data/experiment2/logs/e2loop_transport_DP.console.log"
tmux new-session -d -s e2loop_transport_SCOUT "cd /root/workspace/baojiachun/scout && GPU=0 bash soe_scripts/run_rounds.sh transport SCOUT round_e2.sh 6 2>&1 | tee data/experiment2/logs/e2loop_transport_SCOUT.console.log"
say '=== round0 complete -- launched e2loop_transport_{DP,SCOUT} ==='
