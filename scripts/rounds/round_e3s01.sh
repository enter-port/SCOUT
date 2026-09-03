#!/bin/bash
# SCOUT01 experiment2 round driver -- SPLIT eval/explore protocol (user 2026-08-17).
# Strictly serial per round: rollout -> retrain DP -> retrain dyn.
#   `round_e2.sh <task> <DP|SCOUT01> <round-num>`  (env: GPU=<cuda id>)
#
# Differences from round.sh (experiment1, legacy protocol):
#   * eval and explore are DECOUPLED:
#       - eval   : FIXED scenes every round, seed 42 -> scenes 42..141 (100),
#                  1 try each, pure measurement (record_obs=False; no data).
#       - explore: FRESH scenes every round -- round i uses base seed
#                  i*1000+42 -> scenes +0..+499 (500, try_times=1 each).
#                  successes -> success.hdf5 (DP retrain), ALL trajs ->
#                  all.hdf5 (dyn retrain). Data accumulates across rounds
#                  (success_accum / all_accum) exactly like experiment1.
#   * wandb: ONE run PER ROUND, shared by all three stages:
#       project <TASK>-experiment3 (CAN-experiment3 / SQUARE-experiment3),
#       run name <A>-round<i> (SCOUT01-round1, DP-round3, ...). The rollout
#       process creates the run (id saved to the rollout json); DP / dyn
#       retrains RESUME it via WANDB_RUN_ID + WANDB_RESUME=must and log
#       under their own section keys (DP/* with DP/epoch x-axis, dyn/* with
#       dyn/epoch x-axis; eval/* + explore/* come from the rollout stage).
#       dyn-base prereq is its own run: SCOUT01-round0.
#   * DP retrain epochs are ADAPTIVE: the explore pool is ~15-30x the old
#       core-only convention per round, so 600 fixed epochs would blow up the
#       wall clock at ~90x by round 6. epochs = clamp(12000/n_demos, 100, 600)
#       keeps the gradient-step budget near the 600ep-on-20-demos convention.
set -u

TASK=${1:?usage: round_e2.sh <task> <DP|SCOUT01> <round-num>}
A=${2:?usage: round_e2.sh <task> <DP|SCOUT01> <round-num>}
NUM=${3:?usage: round_e2.sh <task> <DP|SCOUT01> <round-num>}
case "$TASK" in
  can)    TASKUP=CAN ;;
  square) TASKUP=SQUARE ;;
  *) echo "task must be can or square (got: $TASK)"; exit 1 ;;
esac
case "$A" in
  DP|SCOUT01) ;;
  *) echo "a must be DP or SCOUT01 (got: $A)"; exit 1 ;;
esac
NUM=$(printf '%d' "$NUM" 2>/dev/null) || { echo "round-num must be an integer"; exit 1; }
[ "$NUM" -ge 1 ] || { echo "round-num must be >= 1"; exit 1; }
# 2026-08-18 user: optional 4th arg MODE=full|eval-only (default full).
# eval-only: run ONLY the seed-fixed eval phase (success_rate measurement),
# skip explore/accum/retrains -- e.g. the final round of a chain.
MODE=${4:-full}
case "$MODE" in
  full|eval-only) ;;
  *) echo "mode must be full or eval-only (got: $MODE)"; exit 1 ;;
esac
EVALONLY=()
[ "$MODE" = "eval-only" ] && EVALONLY=(--eval-only)

SEED=42                 # eval phase: fixed scene set every round (42..141)
ESEED=$SEED                  # e3: explore REUSES the fixed eval scene set (seed 42, same 100 scenes)
NEXPLORE=100             # 2026-08-18 user: 100 scenes per round (was 500)
ETRIES=1

GPU=${GPU:-0}
export MUJOCO_GL=egl
export TMPDIR=/tmp      # MUST be local (CPFS TMPDIR kills torch_shm_manager)
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout
DATA=$REPO/data/experiment3
PY=/root/workspace/baojiachun/.venv/bin/python
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
RDIR=$DATA/$TASK/rollout/$A-exp$NUM
OUTDP=$TDP/DP-$A-exp$NUM
OUTDYN=$TDYN/dyn-$A-exp$NUM
mkdir -p "$RDIR" "$OUTDP" "$OUTDYN" "$TDYN/dyn-base"
RLOG=$RDIR/rollout.stdout; DPLOG=$OUTDP/train.log
DYNLOG=$OUTDYN/train.log;  DBLOG=$TDYN/dyn-base/train.log
LOG=$DATA/$TASK/round.log
WPROJ=${TASKUP}-experiment3
WNAME=${A}-round${NUM}
cd "$REPO" || exit 1
exec 3>&1

log(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
RUN(){
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY_RUN:' >&3; printf ' %q' "$@" >&3; echo >&3
  else
    "$@"
  fi
}
newest_ckpt(){ ls -t "$1"/checkpoints/*.ckpt 2>/dev/null | head -1; }
newest_vib(){  ls -t "$1"/*/scout_vib.ckpt  2>/dev/null | head -1; }

# ---- prerequisites ------------------------------------------------------- #
for f in configs/eval_${TASK}_e3s01.yaml configs/vib_${TASK}_image_e3.yaml \
         configs/base_dp_${TASK}_image.yaml; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
[ -f "$CORE" ] || { echo "missing core hdf5: $CORE"; exit 1; }
[ -n "$(newest_ckpt "$TDP/DP-base")" ] || { echo "no DP-base ckpt under $TDP/DP-base"; exit 1; }

# ---- resolve this round's rollout inputs (chained, fallback to base) ----- #
PREV=$((NUM - 1))
DPROLL=$TDP/DP-base
for e in $(seq "$PREV" -1 1); do
  if [ -n "$(newest_ckpt "$TDP/DP-$A-exp$e")" ]; then DPROLL=$TDP/DP-$A-exp$e; break; fi
done
DPCKPT=$(newest_ckpt "$DPROLL")
[ -n "$DPCKPT" ] || { log "FATAL: no DP ckpt under $DPROLL"; exit 1; }

VIBARGS=()
if [ "$A" = SCOUT01 ]; then
  if [ "$PREV" -ge 1 ] && [ -n "$(newest_vib "$TDYN/dyn-$A-exp$PREV")" ]; then
    VIBDIR=$TDYN/dyn-$A-exp$PREV
  else
    VIBDIR=$TDYN/dyn-base
  fi
  VIBCKPT=$(newest_vib "$VIBDIR")
  if [ -z "$VIBCKPT" ]; then
    log "no VIB ckpt under $VIBDIR -- training dyn-base first (SCOUT01-round0, vib_${TASK}_image_e3.yaml)"
    RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.train_vib \
      --config configs/vib_${TASK}_image_e3.yaml \
      > "$DBLOG" 2>&1 || { log "dyn-base FAILED"; exit 1; }
    VIBCKPT=$(newest_vib "$VIBDIR")
  fi
  [ -n "$VIBCKPT" ] && VIBARGS=(--vib-ckpt "$VIBCKPT")
fi

T0=$(date +%s)
log "=== ROUND(e2) $TASK a=$A round=$NUM mode=$MODE START (GPU$GPU; rollout DP=$DPROLL; eval_seed=$SEED explore_seed=$ESEED) ==="

# ---- [1/3] rollout: SPLIT protocol (eval fixed scenes + explore fresh) --- #
# SKIP_ROLLOUT=1: crash recovery -- if all.hdf5 already exists (rollout done,
# e.g. a later stage failed), replay only [2/3]+[3/3]; the shared wandb run
# id is re-read from the existing rollout json.
GUIDE=off; [ "$A" = SCOUT01 ] && GUIDE=dyn
if [ "${SKIP_ROLLOUT:-0}" = 1 ] && [ -f "$RDIR/all.hdf5" ]; then
  log "[1/3] SKIP_ROLLOUT=1: reusing existing $RDIR/all.hdf5 (rollout already done)"
  T1=$(date +%s)
else
log "[1/3] rollout guide=$GUIDE eval=$SEED(100) explore=$ESEED($NEXPLORE x$ETRIES) dp=$DPCKPT vib=${VIBCKPT:-none} -> $RDIR"
RUN env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_${TASK}_e3s01.yaml --task "$TASK" --exp-num "$NUM" \
  --base-dp-ckpt "$DPCKPT" \
  --core-hdf5 "$CORE" \
  --guide "$GUIDE" --seed "$SEED" \
  --eval-seed "$SEED" --explore-seed "$ESEED" \
  --n-explore "$NEXPLORE" --explore-try-times "$ETRIES" \
  ${EVALONLY[@]+"${EVALONLY[@]}"} \
  ${VIBARGS[@]+"${VIBARGS[@]}"} \
  --output-dir "$RDIR" \
  --output-success "$RDIR/success.hdf5" \
  --output-all "$RDIR/all.hdf5" \
  --wandb-name "$WNAME" \
  --wandb-project "$WPROJ" \
  > "$RLOG" 2>&1
RC=$?; T1=$(date +%s)
log "[1/3] rollout rc=$RC in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"
if [ $RC -ne 0 ]; then
  # eval-only tolerance: EGL teardown at process exit can crash AFTER the
  # round json is fully written (observed 2026-08-19 square r6: EGL_NOT_
  # INITIALIZED at eglMakeCurrent, json complete). If the measurement
  # landed, the round is done -- don't let a teardown crash kill the chain.
  if [ "$MODE" = "eval-only" ] && $PY - "$RDIR/log" <<'PYEOF'
import sys, json, glob, os
ok = False
for p in glob.glob(os.path.join(sys.argv[1], "*.json")):
    try:
        d = json.load(open(p))
        ok = ok or (d.get("protocol") == "eval_only")
    except Exception:
        pass
sys.exit(0 if ok else 1)
PYEOF
  then
    log "[1/3] WARN: eval-only rollout rc=$RC but round json complete -- treating as success (EGL teardown crash)"
  else
    log "ROLLOUT FAILED - see $RLOG"; exit 1
  fi
fi
fi   # end SKIP_ROLLOUT else-branch

# the rollout json carries the shared wandb run id for the retrains
RID=$($PY - "$RDIR/log" <<'PYEOF'
import sys, json, glob, os
rid = ""
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json")),
                key=os.path.getmtime):
    try:
        d = json.load(open(p))
        if d.get("wandb_run_id"):
            rid = d["wandb_run_id"]
    except Exception:
        pass
print(rid)
PYEOF
)
if [ -n "$RID" ]; then
  log "[wandb] shared round-run $WPROJ/$WNAME id=$RID (DP+dyn will resume it)"
else
  log "[wandb] WARN: no wandb_run_id in rollout json -- retrains will start their own runs"
fi

# ---- eval-only round: measurement done, skip accum + both retrains ------- #
if [ "$MODE" = "eval-only" ]; then
  log "[2/3]+[3/3] SKIPPED (MODE=eval-only: success-rate measurement only)"
  T3=$(date +%s)
  log "=== ROUND(e2) $TASK a=$A round=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
  exit 0
fi

# ---- [2/3] DP retrain on ACCUMULATED successes (core + rounds 1..N) ----- #
if [ ! -f "$RDIR/success.hdf5" ]; then
  log "[2/3] 0 exploration successes this round -- retrain on the SAME accumulated data (anti-deadlock)"
fi
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
# adaptive epochs: keep gradient-step budget near the 600ep-on-20-demos
# convention now that the explore pool is much larger (see header).
# NOTE: f["data"].keys() yields DIRECT children ("demo_0", "total") -- no
# "/" inside, so the old k.split("/")[1] fallback raised IndexError (fixed
# 2026-08-17 after it emptied $EP -> num_epochs="" -> str//int TypeError).
EP=$($PY - "$RDIR/success_accum.hdf5" <<'PYEOF'
import sys, h5py
with h5py.File(sys.argv[1], "r") as f:
    n = f["data"].attrs.get("n_episodes")
    if n is None:
        n = sum(1 for k in f["data"].keys() if str(k).startswith("demo"))
print(max(100, min(600, round(12000 / max(int(n), 1)))))
PYEOF
)
if ! [[ "$EP" =~ ^[0-9]+$ ]]; then
  log "[2/3] WARN: adaptive-epoch probe failed (EP='$EP') -- defaulting to 100"
  EP=100
fi
CKE=$(( EP < 200 ? EP : 200 ))
log "[2/3] DP retrain: ${EP}ep (adaptive, clamp 100..600) ckpt_every=$CKE ds=$RDIR/success_accum.hdf5 -> $OUTDP (wandb resume $WPROJ/$WNAME)"
RUN env CUDA_VISIBLE_DEVICES=$GPU WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY train.py \
  --config-path configs --config-name base_dp_${TASK}_image \
  task.dataset_path="$RDIR/success_accum.hdf5" \
  task.train_filter_key=scout_aug \
  training.num_epochs=$EP \
  training.resume=False \
  training.rollout_every=0 \
  training.sample_every=100 \
  training.checkpoint_every=$CKE \
  training.cudnn_benchmark=true \
  training.device=cuda:0 \
  dataloader.num_workers=8 dataloader.persistent_workers=true \
  +logging.metric_prefix=DP/ \
  logging.name=$WNAME \
  logging.project=$WPROJ \
  hydra.run.dir="$OUTDP" \
  > "$DPLOG" 2>&1
RC=$?; T2=$(date +%s)
log "[2/3] DP retrain (workers=8) rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
if [ $RC -ne 0 ]; then
  log "[2/3] workers=8 failed (known intermittent torch shm) -- retry with num_workers=0"
  RUN env CUDA_VISIBLE_DEVICES=$GPU WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY train.py \
    --config-path configs --config-name base_dp_${TASK}_image \
    task.dataset_path="$RDIR/success_accum.hdf5" \
    task.train_filter_key=scout_aug \
    training.num_epochs=$EP \
    training.resume=False \
    training.rollout_every=0 \
    training.sample_every=100 \
    training.checkpoint_every=$CKE \
    training.cudnn_benchmark=true \
    training.device=cuda:0 \
    dataloader.num_workers=0 \
    +logging.metric_prefix=DP/ \
    logging.name=$WNAME \
    logging.project=$WPROJ \
    hydra.run.dir="$OUTDP" \
    > "$DPLOG" 2>&1
  RC=$?; T2=$(date +%s)
  log "[2/3] DP retrain (workers=0 fallback) rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
fi
[ $RC -ne 0 ] && { log "DP RETRAIN FAILED - see $DPLOG"; exit 1; }

# ---- [3/3] dyn retrain (SCOUT01 only) on ACCUMULATED data ------------------ #
if [ "$A" != "DP" ]; then
CFG=$OUTDYN/config.yaml
NEWDP=$(newest_ckpt "$OUTDP")
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

with open(f"configs/vib_{task}_image_e3.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["zarr_path"] = accum
cfg["dataset"]["feature_cache"] = True
ck = sorted(glob.glob(os.path.join(outdp, "checkpoints", "*.ckpt")),
            key=os.path.getmtime)
if ck:
    cfg["model"]["E_s"]["base_dp_ckpt"] = ck[-1]
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = f"{a_tag}-round{num}"
cfg["wandb"]["project"] = f"{task.upper()}-experiment3"
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
PYEOF
log "[3/3] dyn retrain: ds=$RDIR/all_accum.hdf5 es_base=${NEWDP:-base-config} -> $OUTDYN (wandb resume)"
RUN env CUDA_VISIBLE_DEVICES=$GPU WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY -m scout.train_vib \
  --config "$CFG" \
  > "$DYNLOG" 2>&1
RC=$?; T3=$(date +%s)
log "[3/3] dyn retrain rc=$RC in $(( (T3-T2)/60 ))m$(( (T3-T2)%60 ))s"
[ $RC -ne 0 ] && { log "DYN RETRAIN FAILED - see $DYNLOG"; exit 1; }
else
  log "[3/3] dyn retrain SKIPPED for a=DP (baseline never consumes the VIB)"
  T3=$T2
fi

log "=== ROUND(e2) $TASK a=$A round=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
