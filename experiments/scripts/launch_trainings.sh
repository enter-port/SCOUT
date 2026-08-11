#!/usr/bin/env bash
set -euo pipefail
REPO=/root/workspace/baojiachun/scout
VPY=/root/workspace/baojiachun/.venv/bin/python
WENV=/root/workspace/baojiachun/.secrets/wandb.env
LOG=/root/workspace/baojiachun/data
PROJ=scout-base-dp
launch () {
  local task=$1 gpu=$2
  tmux kill-session -t train_$task 2>/dev/null || true
  tmux new-session -d -s train_$task "set -a && . $WENV && set +a && cd $REPO && CUDA_VISIBLE_DEVICES=$gpu $VPY train.py --config-path configs --config-name base_dp_${task}_image logging.project=$PROJ logging.name=SCOUT-baseDP-${task} 2>&1 | tee $LOG/train_${task}.log"
  echo "launched train_$task on GPU $gpu"
}
launch lift 0
launch can 1
launch square 2
sleep 3
tmux ls
