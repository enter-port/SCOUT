#!/bin/bash
# FAST-ROUND timing validation (square): DP retrain -> VIB retrain -> rollout,
# strictly serial, wall-clock stamped. Outputs ONLY under *-fasttest dirs --
# nothing existing is overwritten.
set -u
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout
DATA=$REPO/data
PY=/root/workspace/baojiachun/.venv/bin/python
TASK=square
GPU=4
LOG=$DATA/fast_round_${TASK}.log
log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
cd $REPO || exit 1

DS=$DATA/$TASK/rollout/${TASK}_SCOUT_exp1.hdf5
OUT=$DATA/$TASK/train/DP-${TASK}-SCOUT-exp1-fasttest
VIBDIR=$DATA/$TASK/train/SCOUT-dyn-${TASK}-fasttest
mkdir -p "$OUT" "$VIBDIR"

T0=$(date +%s)
log "=== FAST ROUND ($TASK) START (GPU$GPU) ==="

log "[1/3] DP retrain: 600ep, no mid-eval (rollout_every=0), workers=8, ds=$DS"
CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
  --config-path configs --config-name base_dp_${TASK}_image \
  task.dataset_path=$DS \
  task.train_filter_key=scout_aug \
  training.num_epochs=600 \
  training.resume=False \
  training.rollout_every=0 \
  training.sample_every=100 \
  training.device=cuda:0 \
  dataloader.num_workers=8 \
  dataloader.persistent_workers=true \
  logging.name=DP-${TASK}-SCOUT-exp1-fasttest \
  logging.project=scout-base-dp \
  hydra.run.dir=$OUT \
  > $OUT.train.log 2>&1
RC=$?; T1=$(date +%s)
log "[1/3] DP done rc=$RC in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"
[ $RC -ne 0 ] && { log "DP FAILED - see $OUT.train.log"; exit 1; }

log "[2/3] VIB retrain (feature cache): configs/vib_${TASK}_image_fast.yaml"
CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.train_vib --config configs/vib_${TASK}_image_fast.yaml \
  > $VIBDIR.train.log 2>&1
RC=$?; T2=$(date +%s)
log "[2/3] VIB done rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
[ $RC -ne 0 ] && { log "VIB FAILED - see $VIBDIR.train.log"; exit 1; }

VIBCKPT=$(ls -t $VIBDIR/*/scout_vib.ckpt 2>/dev/null | head -1)
log "[3/3] rollout: baseline 100 + guided explore (dp=$OUT/checkpoints/580.ckpt vib=$VIBCKPT)"
CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_${TASK}.yaml --task $TASK --exp-num 1 \
  --base-dp-ckpt $OUT/checkpoints/580.ckpt \
  --vib-ckpt "$VIBCKPT" \
  --core-hdf5 $DATA/$TASK/rollout/${TASK}_core.hdf5 \
  --guide dyn --cuda-visible-devices $GPU \
  --output-dir $DATA/$TASK/rollout/fasttest \
  --wandb-name ${TASK}-SCOUT-rollout-fasttest \
  > $DATA/$TASK/rollout/fasttest.stdout 2>&1
RC=$?; T3=$(date +%s)
log "[3/3] rollout done rc=$RC in $(( (T3-T2)/60 ))m$(( (T3-T2)%60 ))s"
[ $RC -ne 0 ] && { log "ROLLOUT FAILED - see $DATA/$TASK/rollout/fasttest.stdout"; exit 1; }
log "=== FAST ROUND ($TASK) TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
