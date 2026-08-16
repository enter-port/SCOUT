#!/bin/bash
# SCOUT round driver (rollout -> retrain DP -> retrain dyn), strictly serial,
# wall-clock stamped. Task-generic: `round.sh <task> <DP|SCOUT> <exp-num>`
# (can and square are wired today; a new task needs configs/eval_<task>.yaml,
# configs/{base_dp,vib}_<task>_image.yaml and data/<task>/rollout/<task>_core.hdf5).
#
# Canonical data layout / naming (2026-08-14):
#   data/<task>/train/DP/DP-base              frozen base DP (E0; formerly DP-<task>-base)
#   data/<task>/train/DP/DP-{a}-exp{num}      retrained DP,   a in {DP, SCOUT}, num>=1
#   data/<task>/train/dyn/dyn-base            Step-1 VIB dynamics (full-data train)
#   data/<task>/train/dyn/dyn-{a}-exp{num}    retrained dyn
#   data/<task>/rollout/{a}-exp{num}/         success.hdf5 + all.hdf5 + log/*.json
#     success.hdf5  = core + successful EXPLORATION trajs    -> DP retrain
#     all.hdf5      = core + every traj of the round
#     all_accum.hdf5 = core + every traj of rounds 1..num    -> dyn retrain
#       (dyn data ACCUMULATES across rounds -- user rule 2026-08-15; the DP
#        retrains on round-N successes only)
#
# Round chaining: exp{num} rolls out with DP-{a}-exp{num-1} (fallback DP-base)
# and, for a=SCOUT, is guided by dyn-{a}-exp{num-1} (fallback dyn-base). The
# retrained dyn uses E_s from the NEW DP-{a}-exp{num} (the VIB ckpt also pins
# E_s since the ModuleDict fix), keeping the next round's (DP, E_s) pair
# matched. If dyn-base is missing it is trained first
# (configs/vib_<task>.yaml, ~15 min with the feature cache).
#
# Round seed convention (2026-08-16, SOE protocol): EVERY round rolls out
# with the SAME fixed seed 42 (init scenes 42..141), so success rates are
# directly comparable round over round. (Replaced the earlier per-round
# derivation exp1=42 / exp2=422 / exp3=4222 ... which gave each round fresh
# scenes and made cross-round numbers incomparable.)
#
# Disk note: retrains write training.checkpoint_every=200 (~6 ckpts = ~28 GB
# per round) because the shared CPFS runs near quota; change if space allows.
#
# Usage:
#   bash soe_scripts/round.sh <task> <DP|SCOUT> <exp-num>
# Env:
#   GPU=0        CUDA device for all three stages
#   DRY_RUN=1    print the commands instead of executing (no dirs created)
#   SKIP_ROLLOUT=1  crash recovery: if $RDIR/all.hdf5 already exists, skip
#                   stage [1/3] entirely and replay only retrain [2/3]+[3/3]
set -u

TASK=${1:?usage: round.sh <task> <DP|SCOUT> <exp-num>}
A=${2:?usage: round.sh <task> <DP|SCOUT> <exp-num>}
NUM=${3:?usage: round.sh <task> <DP|SCOUT> <exp-num>}
case "$TASK" in
  can|square) ;;
  *) echo "task must be can or square (got: $TASK)."
     echo "to add a task: configs/eval_<task>.yaml + configs/base_dp_<task>_image.yaml"
     echo "                + configs/vib_<task>_image.yaml + data/<task>/rollout/<task>_core.hdf5"
     exit 1 ;;
esac
case "$A" in
  DP|SCOUT) ;;
  *) echo "a must be DP or SCOUT (got: $A)"; exit 1 ;;
esac
NUM=$(printf '%d' "$NUM" 2>/dev/null) || { echo "exp-num must be an integer"; exit 1; }
[ "$NUM" -ge 1 ] || { echo "exp-num must be >= 1"; exit 1; }

# SOE protocol: fixed seed every round
SEED=42

GPU=${GPU:-0}
DRY_RUN=${DRY_RUN:-0}
SKIP_ROLLOUT=${SKIP_ROLLOUT:-0}
export MUJOCO_GL=egl
# TMPDIR MUST be a LOCAL filesystem. The inherited DSW container env sets
# TMPDIR=/mnt/workspace/zimo/.tmp (CPFS network mount): AF_UNIX bind there
# fails with EOPNOTSUPP, killing torch_shm_manager and thus every
# num_workers>0 DataLoader (2x2 experiment verified 2026-08-15). It also
# routes all tempfile IO (e.g. robosuite get_xml) over the network mount.
export TMPDIR=/tmp
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout
DATA=$REPO/data
PY=/root/workspace/baojiachun/.venv/bin/python
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
RDIR=$DATA/$TASK/rollout/$A-exp$NUM
OUTDP=$TDP/DP-$A-exp$NUM
OUTDYN=$TDYN/dyn-$A-exp$NUM
if [ "$DRY_RUN" = 1 ]; then
  RLOG=/dev/null; DPLOG=/dev/null; DYNLOG=/dev/null; DBLOG=/dev/null
else
  # every artifact (train.log / config.yaml / wandb) lives INSIDE its run dir
  mkdir -p "$RDIR" "$OUTDP" "$OUTDYN" "$TDYN/dyn-base"
  RLOG=$RDIR/rollout.stdout; DPLOG=$OUTDP/train.log
  DYNLOG=$OUTDYN/train.log;  DBLOG=$TDYN/dyn-base/train.log
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

# ---- per-task prerequisites (read-only checks; also useful in DRY_RUN) ---- #
for f in configs/eval_${TASK}.yaml configs/vib_${TASK}_image.yaml \
         configs/base_dp_${TASK}_image.yaml; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
[ -f "$CORE" ] || { echo "missing core hdf5: $CORE"; exit 1; }
[ -n "$(newest_ckpt "$TDP/DP-base")" ] || { echo "no DP-base ckpt under $TDP/DP-base"; exit 1; }

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
if [ "$SKIP_ROLLOUT" = 1 ] && [ -f "$RDIR/all.hdf5" ]; then
  log "[1/3] SKIP_ROLLOUT=1: reusing existing $RDIR/all.hdf5 (rollout already done)"
  T1=$(date +%s)
else
  GUIDE=off; [ "$A" = SCOUT ] && GUIDE=dyn
  log "[1/3] rollout guide=$GUIDE seed=$SEED dp=${DPCKPT:-<dry>} vib=${VIBCKPT:-none} -> $RDIR"
  RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
    --config configs/eval_${TASK}.yaml --task "$TASK" --exp-num "$NUM" \
    --base-dp-ckpt "${DPCKPT:-$DPROLL/checkpoints/<newest>.ckpt}" \
    --core-hdf5 "$CORE" \
    --guide "$GUIDE" --seed "$SEED" ${VIBARGS[@]+"${VIBARGS[@]}"} \
    --output-dir "$RDIR" \
    --output-success "$RDIR/success.hdf5" \
    --output-all "$RDIR/all.hdf5" \
    --wandb-name "$A-$TASK-rollout-exp$NUM" \
    > "$RLOG" 2>&1
  RC=$?; T1=$(date +%s)
  log "[1/3] rollout rc=$RC in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"
  [ $RC -ne 0 ] && { log "ROLLOUT FAILED - see $RLOG"; exit 1; }
fi

# ---- [2/3] DP retrain on ACCUMULATED successes (user rule 2026-08-15: like
# dyn's all_accum, the DP trains on core + EVERY round's exploration successes
# 1..N, not just the current round's) --------------------------------------- #
if [ -f "$RDIR/success.hdf5" ]; then
  if [ "$DRY_RUN" = 1 ]; then
    log "[2/3] success-accum (dry): core+success.hdf5(rounds 1..$NUM) -> $RDIR/success_accum.hdf5"
  else
    $PY - "$RDIR" "$A" "$CORE" <<'PYEOF'
import sys, glob, os, re
rdir, a_tag, core_path = sys.argv[1:4]
sys.path.insert(0, os.getcwd())
from scout.eval.hdf5_writer import merge_accumulated_hdf5

def expnum(p):
    m = re.search(r"-exp(\d+)", p)
    return int(m.group(1)) if m else 0

rollout_root = os.path.dirname(rdir.rstrip("/"))
succs = sorted(glob.glob(os.path.join(rollout_root, f"{a_tag}-exp*", "success.hdf5")),
               key=expnum)
accum = os.path.join(rdir, "success_accum.hdf5")
info = merge_accumulated_hdf5(core_path, succs, accum)
print(f"[dp-accum] merged {info} -> {accum}")
PYEOF
    log "[2/3] DP retrain: 600ep no mid-eval ckpt_every=200 ds=$RDIR/success_accum.hdf5 (core+rounds1..$NUM successes) -> $OUTDP"
  fi
  # dataloader workers: 8 is ~3x faster (13-15 it/s vs 4.6, verified 08-15 on
  # GPU4, train.py sets sharing strategy file_system) but torch shm crashes
  # INTERMITTENTLY on this server -- so try 8 first; on failure fall back to
  # the always-safe 0 instead of losing the round (see AGENTS.md 坑5).
  RUN env CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
    --config-path configs --config-name base_dp_${TASK}_image \
    task.dataset_path="$RDIR/success_accum.hdf5" \
    task.train_filter_key=scout_aug \
    training.num_epochs=600 \
    training.resume=False \
    training.rollout_every=0 \
    training.sample_every=100 \
    training.checkpoint_every=200 \
    training.cudnn_benchmark=true \
    training.device=cuda:0 \
    dataloader.num_workers=8 dataloader.persistent_workers=true \
    logging.name=DP-${TASK}-${A}-exp${NUM} \
    logging.project=scout-base-dp \
    hydra.run.dir="$OUTDP" \
    > "$DPLOG" 2>&1
  RC=$?; T2=$(date +%s)
  log "[2/3] DP retrain (workers=8) rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
  if [ $RC -ne 0 ]; then
    log "[2/3] workers=8 failed (known intermittent torch shm) -- retry with num_workers=0"
    RUN env CUDA_VISIBLE_DEVICES=$GPU $PY train.py \
      --config-path configs --config-name base_dp_${TASK}_image \
      task.dataset_path="$RDIR/success_accum.hdf5" \
      task.train_filter_key=scout_aug \
      training.num_epochs=600 \
      training.resume=False \
      training.rollout_every=0 \
      training.sample_every=100 \
      training.checkpoint_every=200 \
    training.cudnn_benchmark=true \
      training.device=cuda:0 \
      dataloader.num_workers=0 \
      logging.name=DP-${TASK}-${A}-exp${NUM} \
      logging.project=scout-base-dp \
      hydra.run.dir="$OUTDP" \
      > "$DPLOG" 2>&1
    RC=$?; T2=$(date +%s)
    log "[2/3] DP retrain (workers=0 fallback) rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
  fi
  [ $RC -ne 0 ] && { log "DP RETRAIN FAILED - see $DPLOG"; exit 1; }
else
  log "[2/3] no success.hdf5 (0 exploration successes) -- DP retrain SKIPPED"
  T2=$T1
fi

# ---- [3/3] dyn retrain on ACCUMULATED data (core + every traj of rounds
# 1..N; user rule 2026-08-15 updated: BOTH DP and dyn accumulate every
# round -- DP on exploration SUCCESSES, dyn on EVERY trajectory incl.
# failures; E_s from new DP) ---------------------------------------------- #
CFG=$OUTDYN/config.yaml
NEWDP=$(newest_ckpt "$OUTDP")
if [ "$DRY_RUN" = 1 ]; then
  log "[3/3] dyn retrain (dry): accum=core+all.hdf5(rounds 1..$NUM) es_base=${NEWDP:-<newest ckpt of $OUTDP>} -> $OUTDYN"
else
  $PY - "$CFG" "$RDIR" "$OUTDYN" "$OUTDP" "$A" "$NUM" "$TASK" "$CORE" <<'PYEOF'
import sys, yaml, glob, os, re
cfg_path, rdir, outdyn, outdp, a_tag, num, task, core_path = sys.argv[1:9]
sys.path.insert(0, os.getcwd())
from scout.eval.hdf5_writer import merge_accumulated_hdf5

def expnum(p):
    m = re.search(r"-exp(\d+)", p)
    return int(m.group(1)) if m else 0

rollout_root = os.path.dirname(rdir.rstrip("/"))
alls = sorted(glob.glob(os.path.join(rollout_root, f"{a_tag}-exp*", "all.hdf5")),
              key=expnum)
accum = os.path.join(rdir, "all_accum.hdf5")
info = merge_accumulated_hdf5(core_path, alls, accum)
print(f"[dyn-accum] merged {info} -> {accum}")

with open(f"configs/vib_{task}_image.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["zarr_path"] = accum
cfg["dataset"]["feature_cache"] = True
ck = sorted(glob.glob(os.path.join(outdp, "checkpoints", "*.ckpt")),
            key=os.path.getmtime)
if ck:   # E_s from the NEW DP keeps the next round's (DP, E_s) pair matched
    cfg["model"]["E_s"]["base_dp_ckpt"] = ck[-1]
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = f"dyn-{task}-{a_tag}-exp{num}"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
  log "[3/3] dyn retrain: ds=$RDIR/all_accum.hdf5 (core+rounds1..$NUM) es_base=${NEWDP:-base-config} -> $OUTDYN"
fi
RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.train_vib --config "$CFG" \
  > "$DYNLOG" 2>&1
RC=$?; T3=$(date +%s)
log "[3/3] dyn retrain rc=$RC in $(( (T3-T2)/60 ))m$(( (T3-T2)%60 ))s"
[ $RC -ne 0 ] && { log "DYN RETRAIN FAILED - see $DYNLOG"; exit 1; }

log "=== ROUND $TASK a=$A exp=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
