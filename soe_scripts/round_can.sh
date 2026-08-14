#!/bin/bash
# SCOUT round driver -- CAN (rollout -> retrain DP -> retrain dyn), strictly
# serial, wall-clock stamped. Clone this file per task (change TASK + core
# hdf5 name) to extend to square / lift / transport.
#
# Canonical data layout / naming (2026-08-14):
#   data/can/train/DP/DP-base              frozen base DP (E0; formerly DP-can-base)
#   data/can/train/DP/DP-{a}-exp{num}      retrained DP,   a in {DP, SCOUT}, num>=1
#   data/can/train/dyn/dyn-base            Step-1 VIB dynamics (full-data train)
#   data/can/train/dyn/dyn-{a}-exp{num}    retrained dyn
#   data/can/rollout/{a}-exp{num}/         success.hdf5 + all.hdf5 + log/*.json
#     success.hdf5 = core + successful EXPLORATION trajs -> DP retrain
#     all.hdf5     = core + every traj of the round          -> dyn retrain
#
# Round chaining: exp{num} rolls out with DP-{a}-exp{num-1} (fallback DP-base)
# and, for a=SCOUT, is guided by dyn-{a}-exp{num-1} (fallback dyn-base). The
# retrained dyn uses E_s from the NEW DP-{a}-exp{num}, so the next round's
# (DP, E_s) pair stays matched. If dyn-base is missing it is trained first
# (configs/vib_can_image.yaml, ~15 min with the feature cache).
#
# Disk note: retrains write training.checkpoint_every=100 (~6 ckpts = ~28 GB
# per round) because the shared CPFS runs near quota; change if space allows.
#
# Usage:
#   bash soe_scripts/round_can.sh <DP|SCOUT> <exp-num>
# Env:
#   GPU=0        CUDA device for all three stages
#   DRY_RUN=1    print the commands instead of executing (no dirs created)
set -u

A=${1:?usage: round_can.sh <DP|SCOUT> <exp-num>}
NUM=${2:?usage: round_can.sh <DP|SCOUT> <exp-num>}
case "$A" in
  DP|SCOUT) ;;
  *) echo "a must be DP or SCOUT (got: $A)"; exit 1 ;;
esac
NUM=$(printf '%d' "$NUM" 2>/dev/null) || { echo "exp-num must be an integer"; exit 1; }
[ "$NUM" -ge 1 ] || { echo "exp-num must be >= 1"; exit 1; }

GPU=${GPU:-0}
DRY_RUN=${DRY_RUN:-0}
export MUJOCO_GL=egl
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout
DATA=$REPO/data
PY=/root/workspace/baojiachun/.venv/bin/python
TASK=can
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
RDIR=$DATA/$TASK/rollout/$A-exp$NUM
OUTDP=$TDP/DP-$A-exp$NUM
OUTDYN=$TDYN/dyn-$A-exp$NUM
if [ "$DRY_RUN" = 1 ]; then
  RLOG=/dev/null; DPLOG=/dev/null; DYNLOG=/dev/null; DBLOG=/dev/null
else
  mkdir -p "$RDIR"
  RLOG=$RDIR/rollout.stdout; DPLOG=$OUTDP.train.log
  DYNLOG=$OUTDYN.train.log;  DBLOG=$TDYN/dyn-base.train.log
fi
LOG=$DATA/$TASK/round.log
cd "$REPO" || exit 1
exec 3>&1   # dry-run command echo escapes stage-log redirects via fd 3

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
RUN(){
  if [ "$DRY_RUN" = 1 ]; then
    printf 'DRY_RUN:' >&3; printf ' %q' "$@" >&3; echo >&3
  else
    "$@"
  fi
}
newest_ckpt(){ ls -t "$1"/checkpoints/*.ckpt 2>/dev/null | head -1; }
newest_vib(){  ls -t "$1"/*/scout_vib.ckpt  2>/dev/null | head -1; }

# ---- resolve this round's rollout inputs (chained, fallback to base) ------- #
PREV=$((NUM - 1))
if [ "$PREV" -ge 1 ] && [ -n "$(newest_ckpt "$TDP/DP-$A-exp$PREV")" ]; then
  DPROLL=$TDP/DP-$A-exp$PREV
else
  DPROLL=$TDP/DP-base
fi
DPCKPT=$(newest_ckpt "$DPROLL")
if [ -z "$DPCKPT" ] && [ "$DRY_RUN" != 1 ]; then
  log "FATAL: no DP ckpt under $DPROLL"; exit 1
fi

VIBARGS=()
if [ "$A" = SCOUT ]; then
  if [ "$PREV" -ge 1 ] && [ -n "$(newest_vib "$TDYN/dyn-$A-exp$PREV")" ]; then
    VIBDIR=$TDYN/dyn-$A-exp$PREV
  else
    VIBDIR=$TDYN/dyn-base
  fi
  VIBCKPT=$(newest_vib "$VIBDIR")
  if [ -z "$VIBCKPT" ]; then
    log "no VIB ckpt under $VIBDIR -- training dyn-base first (vib_${TASK}_image.yaml)"
    if [ "$DRY_RUN" != 1 ]; then
      mkdir -p "$TDYN/dyn-base"
      RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.train_vib \
        --config configs/vib_${TASK}_image.yaml \
        > "$DBLOG" 2>&1 || { log "dyn-base FAILED"; exit 1; }
    fi
    VIBCKPT=$(newest_vib "$VIBDIR")   # may stay empty in DRY_RUN
  fi
  [ -n "$VIBCKPT" ] && VIBARGS=(--vib-ckpt "$VIBCKPT")
fi

T0=$(date +%s)
log "=== ROUND $TASK a=$A exp=$NUM START (GPU$GPU; rollout DP=$DPROLL) ==="

# ---- [1/3] rollout: baseline 100-init + (SCOUT) guided explore on fails --- #
GUIDE=off; [ "$A" = SCOUT ] && GUIDE=dyn
log "[1/3] rollout guide=$GUIDE dp=${DPCKPT:-<dry>} vib=${VIBCKPT:-none} -> $RDIR"
RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_${TASK}.yaml --task "$TASK" --exp-num "$NUM" \
  --base-dp-ckpt "${DPCKPT:-$DPROLL/checkpoints/<newest>.ckpt}" \
  --core-hdf5 "$CORE" \
  --guide "$GUIDE" ${VIBARGS[@]+"${VIBARGS[@]}"} \
  --output-dir "$RDIR" \
  --output-success success.hdf5 \
  --output-all all.hdf5 \
  --wandb-name "$A-$TASK-rollout-exp$NUM" \
  > "$RLOG" 2>&1
RC=$?; T1=$(date +%s)
log "[1/3] rollout rc=$RC in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"
[ $RC -ne 0 ] && { log "ROLLOUT FAILED - see $RDIR/rollout.stdout"; exit 1; }

# ---- [2/3] DP retrain on success.hdf5 (core + exploration successes) ------ #
if [ -f "$RDIR/success.hdf5" ]; then
  log "[2/3] DP retrain: 600ep no mid-eval ckpt_every=100 ds=$RDIR/success.hdf5 -> $OUTDP"
  RUN env CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
    --config-path configs --config-name base_dp_${TASK}_image \
    task.dataset_path="$RDIR/success.hdf5" \
    task.train_filter_key=scout_aug \
    training.num_epochs=600 \
    training.resume=False \
    training.rollout_every=0 \
    training.sample_every=100 \
    training.checkpoint_every=100 \
    training.device=cuda:0 \
    dataloader.num_workers=8 \
    dataloader.persistent_workers=true \
    logging.name=DP-${TASK}-${A}-exp${NUM} \
    logging.project=scout-base-dp \
    hydra.run.dir="$OUTDP" \
    > "$DPLOG" 2>&1
  RC=$?; T2=$(date +%s)
  log "[2/3] DP retrain rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
  [ $RC -ne 0 ] && { log "DP RETRAIN FAILED - see $OUTDP.train.log"; exit 1; }
else
  log "[2/3] no success.hdf5 (0 exploration successes) -- DP retrain SKIPPED"
  T2=$T1
fi

# ---- [3/3] dyn retrain on all.hdf5 (core + every traj; E_s from new DP) --- #
CFG=$OUTDYN.config.yaml
NEWDP=$(newest_ckpt "$OUTDP")
if [ "$DRY_RUN" = 1 ]; then
  log "[3/3] dyn retrain (dry): ds=$RDIR/all.hdf5 es_base=${NEWDP:-<newest ckpt of $OUTDP>} -> $OUTDYN"
else
  $PY - "$CFG" "$RDIR" "$OUTDYN" "$OUTDP" "$A" "$NUM" <<'PYEOF'
import sys, yaml, glob, os
cfg_path, rdir, outdyn, outdp, a_tag, num = sys.argv[1:7]
with open("configs/vib_can_image.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["zarr_path"] = f"{rdir}/all.hdf5"
cfg["dataset"]["feature_cache"] = True
ck = sorted(glob.glob(os.path.join(outdp, "checkpoints", "*.ckpt")),
            key=os.path.getmtime)
if ck:   # E_s from the NEW DP keeps the next round's (DP, E_s) pair matched
    cfg["model"]["E_s"]["base_dp_ckpt"] = ck[-1]
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = f"dyn-can-{a_tag}-exp{num}"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
  log "[3/3] dyn retrain: ds=$RDIR/all.hdf5 es_base=${NEWDP:-base-config} -> $OUTDYN"
fi
RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.train_vib --config "$CFG" \
  > "$DYNLOG" 2>&1
RC=$?; T3=$(date +%s)
log "[3/3] dyn retrain rc=$RC in $(( (T3-T2)/60 ))m$(( (T3-T2)%60 ))s"
[ $RC -ne 0 ] && { log "DYN RETRAIN FAILED - see $OUTDYN.train.log"; exit 1; }

log "=== ROUND $TASK a=$A exp=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
