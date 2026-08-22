#!/bin/bash
# round.sh (v3, 2026-08-21) -- seeded reproducible self-improvement round driver.
# Evolves round_e2.sh (SPLIT eval/explore protocol) with the user's 2026-08-21
# requirements:
#
#   T1  ONE seed (env TSEED) controls ALL training randomness:
#       DP trainings get training.seed=$TSEED, dyn trainings get cfg.seed=$TSEED.
#       (Eval scenes stay seed 42..141; explore scenes stay i*1000+42.)
#   T2  CUDA determinism: every training stage runs with
#       +training.cudnn_deterministic=true (cudnn.benchmark off, deterministic
#       conv algos) and CUBLAS_WORKSPACE_CONFIG=:4096:8 (deterministic GEMM).
#   T4  dyn failure_weight comes from configs/vib_${TASK}_exp1.yaml (=1).
#   T5  ROUND 0 IS BUILT IN (arm BASE, num 0; idempotent per component):
#         [0/3] seeded split: 20 demos of the OFFICIAL robomimic 200
#               (data/robomimic/<task>/ph/image_v141_abs.hdf5) via
#               soe_scripts/split_core.py rng(default_rng(TSEED)) -> core;
#         [1/3] base DP: 600ep, seed=$TSEED, deterministic, on the core;
#         [2/3] dyn-base: vib_${TASK}_exp1.yaml on the core, E_s from the new
#               DP-base.  Round 0 is SHARED by the SCOUT and DP arms of the
#               same seed (run it once; the other arm's round 0 skips).
#   T6  DYN_FREEZE_AFTER (default 3): rounds > it SKIP the dyn retrain and the
#       rollout VIB walk-backs to the last trained dyn (dyn-$A-exp$FREEZE).
#
# Usage:  round.sh <task> <BASE|SCOUT|DP> <num>
#           arm BASE + num 0   -> round 0 (split + base DP + dyn-base)
#           arm SCOUT/DP + num>=1 -> one chain round (rollout -> DP retrain
#                                    -> dyn retrain (SCOUT, until freeze))
#         optional 4th arg MODE=full|eval-only (default full).
# Env:    GPU=<id> TSEED=<int> DATA_ROOT=<abs dir> (required)
#         DYN_FREEZE_AFTER=3 NEXPLORE=100
# Layout: $DATA_ROOT/<task>/{rollout/,train/DP/,train/dyn/} (as experiment2).
set -u

TASK=${1:?usage: round.sh <task> <BASE|SCOUT|DP> <num>}
A=${2:?usage: round.sh <task> <BASE|SCOUT|DP> <num>}
NUM=${3:?usage: round.sh <task> <BASE|SCOUT|DP> <num>}
case "$TASK" in
  can)      TASKUP=CAN ;;
  square)   TASKUP=SQUARE ;;
  transport) TASKUP=TRANSPORT ;;
  *) echo "task must be can, square or transport (got: $TASK)"; exit 1 ;;
esac
case "$A" in
  BASE) [ "$NUM" = 0 ] || { echo "arm BASE only valid with num 0"; exit 1; } ;;
  DP|SCOUT) [ "$NUM" -ge 1 ] 2>/dev/null || { echo "arm SCOUT/DP needs num>=1"; exit 1; } ;;
  *) echo "a must be BASE, DP or SCOUT (got: $A)"; exit 1 ;;
esac
MODE=${4:-full}
case "$MODE" in
  full|eval-only) ;;
  *) echo "mode must be full or eval-only"; exit 1 ;;
esac
EVALONLY=()
[ "$MODE" = "eval-only" ] && EVALONLY=(--eval-only)

GPU=${GPU:?set GPU=<cuda id>}
TSEED=${TSEED:?set TSEED=<training seed -- controls split/init/shuffle/crop>}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT=<experiment dir, e.g. data/2026_8_21/CAN-exp1-233>}
DYN_FREEZE_AFTER=${DYN_FREEZE_AFTER:-3}
SEED=42                       # eval phase: FIXED scene set every round (42..141)
NEXPLORE=${NEXPLORE:-100}
ETRIES=1
NENV=${NENV:-25}              # rollout concurrency (config also says 25)
NENV_RETRY=${NENV_RETRY:-12}  # render-corruption retry: halve the EGL surface
if [ "$A" = BASE ]; then ESEED=0; else ESEED=$((NUM * 1000 + 42)); fi

export MUJOCO_GL=egl
export TMPDIR=/tmp            # MUST be local (CPFS TMPDIR kills torch_shm_manager)
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # T2: deterministic cuBLAS GEMM
# spread offscreen rendering one GPU per chain (see rollout.py's env factory;
# 2026-08-22: everything-on-EGL-device-0 overloaded it into frame corruption)
export SCOUT_RENDER_GPU=$GPU
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
DATA=$DATA_ROOT
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
LOG=$DATA/$TASK/round.log
WPROJ=${TASKUP}-2026-8-21
mkdir -p "$TDP" "$TDYN" "$(dirname "$CORE")"
cd "$REPO" || exit 1
exec 3>&1

# DRY_RUN must not append to round.log (a fake TOTAL line would trip
# wait_launch_dp.sh into launching the DP arm prematurely -- happened once
# 2026-08-22; smoke output goes to stdout/chain-log only).
log(){
  echo "[$(date '+%F %T')] $*"
  [ "${DRY_RUN:-0}" = 1 ] || echo "[$(date '+%F %T')] $*" >> "$LOG"
}
RUN(){
  if [ "${DRY_RUN:-0}" = 1 ]; then
    printf 'DRY_RUN:' >&3; printf ' %q' "$@" >&3; echo >&3
  else
    "$@"
  fi
}
newest_ckpt(){ ls -t "$1"/checkpoints/*.ckpt 2>/dev/null | head -1; }
newest_vib(){  ls -t "$1"/*/scout_vib.ckpt  2>/dev/null | head -1; }

# DP/dyn hydra/yaml overrides shared by EVERY training stage (T1+T2).
DPOPTS=(training.seed="$TSEED" training.resume=False training.rollout_every=0
        training.sample_every=100 training.cudnn_benchmark=false
        +training.cudnn_deterministic=true training.device=cuda:0)

# ============================================================================ #
# ROUND 0 (arm BASE): seeded split + base DP + dyn-base (T5)
# ============================================================================ #
if [ "$A" = BASE ]; then
  OFFICIAL=$REPO/data/robomimic/$TASK/ph/image_v141_abs.hdf5
  [ -f "configs/vib_${TASK}_exp1.yaml" ] || { echo "missing configs/vib_${TASK}_exp1.yaml"; exit 1; }
  [ -f "configs/base_dp_${TASK}_image.yaml" ] || { echo "missing base_dp config"; exit 1; }
  [ -f "$OFFICIAL" ] || { echo "missing official 200-demo dataset: $OFFICIAL"; exit 1; }

  T0=$(date +%s)
  log "=== ROUND0 $TASK seed=$TSEED START (GPU$GPU; official=$OFFICIAL) ==="

  # [0/3] seeded 20-of-200 core split
  if [ -f "$CORE" ]; then
    log "[0/3] core exists -- skip split ($CORE)"
  else
    log "[0/3] split: 20 of 200 demos, rng seed $TSEED -> $CORE"
    RUN "$PY" soe_scripts/split_core.py "$OFFICIAL" "$CORE" 20 "$TSEED" \
      || { log "SPLIT FAILED"; exit 1; }
  fi

  # [1/3] base DP: 600ep on the core (seeded + deterministic)
  if [ -n "$(newest_ckpt "$TDP/DP-base")" ]; then
    log "[1/3] DP-base ckpt exists -- skip"
  else
    mkdir -p "$TDP/DP-base"
    log "[1/3] base DP: 600ep seed=$TSEED deterministic ds=$CORE -> $TDP/DP-base"
    RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY train.py \
      --config-path configs --config-name base_dp_${TASK}_image \
      task.dataset_path="$CORE" \
      "${DPOPTS[@]}" \
      training.num_epochs=600 \
      training.checkpoint_every=100 \
      dataloader.num_workers=8 dataloader.persistent_workers=true \
      +logging.metric_prefix=DP/ \
      logging.name=DP-s${TSEED}-round0 \
      logging.project=$WPROJ \
      hydra.run.dir="$TDP/DP-base" \
      > "$TDP/DP-base/train.log" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      log "[1/3] workers=8 failed -- retry num_workers=0"
      RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY train.py \
        --config-path configs --config-name base_dp_${TASK}_image \
        task.dataset_path="$CORE" \
        "${DPOPTS[@]}" \
        training.num_epochs=600 \
        training.checkpoint_every=100 \
        dataloader.num_workers=0 \
        +logging.metric_prefix=DP/ \
        logging.name=DP-s${TSEED}-round0 \
        logging.project=$WPROJ \
        hydra.run.dir="$TDP/DP-base" \
        > "$TDP/DP-base/train.log" 2>&1
      RC=$?
    fi
    [ $RC -ne 0 ] && { log "BASE DP FAILED - see $TDP/DP-base/train.log"; exit 1; }
  fi

  # [2/3] dyn-base on the core (E_s from the fresh DP-base)
  if [ -n "$(newest_vib "$TDYN/dyn-base")" ]; then
    log "[2/3] dyn-base ckpt exists -- skip"
  else
    mkdir -p "$TDYN/dyn-base"
    DPB=$(newest_ckpt "$TDP/DP-base")
    if [ -z "$DPB" ]; then
      if [ "${DRY_RUN:-0}" = 1 ]; then DPB="<DP-base-ckpt>"; else
        log "FATAL: no DP-base ckpt"; exit 1
      fi
    fi
    CFG=$TDYN/dyn-base/config.yaml
    $PY - "$CFG" "$CORE" "$TDYN/dyn-base" "$DPB" "$TSEED" "$WPROJ" "$TASK" <<'PYEOF'
import sys, yaml
cfg_path, zarr, outdyn, dpb, tseed, wproj, task = sys.argv[1:8]
with open(f"configs/vib_{task}_exp1.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["zarr_path"] = zarr
cfg["dataset"]["feature_cache"] = True
cfg["model"]["E_s"]["base_dp_ckpt"] = dpb
cfg["seed"] = int(tseed)
cfg["cudnn_deterministic"] = True
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = f"SCOUT-s{tseed}-round0"
cfg["wandb"]["project"] = wproj
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[dyn-base-cfg] seed={tseed} ds={zarr} es={dpb} -> {outdyn}")
PYEOF
    log "[2/3] dyn-base: seed=$TSEED deterministic ds=$CORE es_base=${DPB##*/} -> $TDYN/dyn-base"
    RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 $PY -m scout.train_vib \
      --config "$CFG" \
      > "$TDYN/dyn-base/train.log" 2>&1 \
      || { log "DYN-BASE FAILED - see $TDYN/dyn-base/train.log"; exit 1; }
  fi

  T3=$(date +%s)
  log "=== ROUND0 $TASK seed=$TSEED TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
  exit 0
fi

# ============================================================================ #
# ROUNDS 1..N (arm SCOUT / DP)
# ============================================================================ #
RDIR=$DATA/$TASK/rollout/$A-exp$NUM
OUTDP=$TDP/DP-$A-exp$NUM
OUTDYN=$TDYN/dyn-$A-exp$NUM
mkdir -p "$RDIR" "$OUTDP" "$OUTDYN"
RLOG=$RDIR/rollout.stdout; DPLOG=$OUTDP/train.log; DYNLOG=$OUTDYN/train.log
WNAME=${A}-s${TSEED}-round${NUM}

for f in configs/eval_${TASK}_exp1.yaml configs/vib_${TASK}_exp1.yaml \
         configs/base_dp_${TASK}_image.yaml; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
[ -n "$(newest_ckpt "$TDP/DP-base")" ] || { echo "no DP-base ckpt (run: round.sh $TASK BASE 0)"; exit 1; }
[ -n "$(newest_vib "$TDYN/dyn-base")" ] || [ "$A" = DP ] \
  || { echo "no dyn-base ckpt (run: round.sh $TASK BASE 0)"; exit 1; }

# ---- resolve this round's rollout inputs (walk-back, fallback to base) ---- #
PREV=$((NUM - 1))
DPROLL=$TDP/DP-base
for e in $(seq "$PREV" -1 1); do
  if [ -n "$(newest_ckpt "$TDP/DP-$A-exp$e")" ]; then DPROLL=$TDP/DP-$A-exp$e; break; fi
done
DPCKPT=$(newest_ckpt "$DPROLL")
[ -n "$DPCKPT" ] || { log "FATAL: no DP ckpt under $DPROLL"; exit 1; }

VIBARGS=()
if [ "$A" = SCOUT ]; then
  # walk-back to the nearest TRAINED dyn (freeze-aware: rounds past
  # DYN_FREEZE_AFTER find dyn-exp$FREEZE here, not dyn-base)
  VIBDIR=$TDYN/dyn-base
  for e in $(seq "$PREV" -1 1); do
    if [ -n "$(newest_vib "$TDYN/dyn-$A-exp$e")" ]; then VIBDIR=$TDYN/dyn-$A-exp$e; break; fi
  done
  VIBCKPT=$(newest_vib "$VIBDIR")
  [ -n "$VIBCKPT" ] || { log "FATAL: no VIB ckpt under $VIBDIR"; exit 1; }
  VIBARGS=(--vib-ckpt "$VIBCKPT")
fi

T0=$(date +%s)
log "=== ROUND $TASK a=$A seed=$TSEED round=$NUM mode=$MODE START (GPU$GPU; rollout DP=$DPROLL vib=${VIBDIR:-none}; eval_seed=$SEED explore_seed=$ESEED) ==="

# ---- [1/3] rollout: SPLIT protocol (eval fixed scenes + explore fresh) ---- #
# Render-corruption guard (2026-08-22): after the rollout, vis_validate the
# explore images in all.hdf5; on CORRUPT, delete the outputs and retry ONCE
# at NENV_RETRY (fewer concurrent envs = smaller EGL surface). Two strikes ->
# FATAL (chain stops loudly instead of training on blind-policy data).
GUIDE=off; [ "$A" = SCOUT ] && GUIDE=dyn
NE=$NENV
ROLLOUT_OK=0
for ATTEMPT in 1 2; do
  if [ "${SKIP_ROLLOUT:-0}" = 1 ] && [ "$ATTEMPT" = 1 ] && [ -f "$RDIR/all.hdf5" ]; then
    log "[1/3] SKIP_ROLLOUT=1: reusing existing $RDIR/all.hdf5 (validating it)"
  else
    rm -f "$RDIR/all.hdf5" "$RDIR/success.hdf5"; rm -rf "$RDIR/log"
    log "[1/3] rollout guide=$GUIDE try$ATTEMPT n_envs=$NE eval=$SEED(100) explore=$ESEED($NEXPLORE x$ETRIES) dp=$DPCKPT vib=${VIBCKPT:-none} -> $RDIR"
    RUN env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout \
      --config configs/eval_${TASK}_exp1.yaml --task "$TASK" --exp-num "$NUM" \
      --base-dp-ckpt "$DPCKPT" \
      --core-hdf5 "$CORE" \
      --guide "$GUIDE" --seed "$SEED" \
      --eval-seed "$SEED" --explore-seed "$ESEED" \
      --n-explore "$NEXPLORE" --explore-try-times "$ETRIES" \
      --n-envs "$NE" \
      ${EVALONLY[@]+"${EVALONLY[@]}"} \
      ${VIBARGS[@]+"${VIBARGS[@]}"} \
      --output-dir "$RDIR" \
      --output-success "$RDIR/success.hdf5" \
      --output-all "$RDIR/all.hdf5" \
      --wandb-name "$WNAME" \
      --wandb-project "$WPROJ" \
      > "$RLOG" 2>&1
    RC=$?
    if [ $RC -ne 0 ]; then
      log "[1/3] rollout rc=$RC (attempt $ATTEMPT, n_envs=$NE) -- see $RLOG"
      [ "$MODE" = "eval-only" ] && { log "ROLLOUT FAILED (eval-only, no retry data)"; exit 1; }
      NE=$NENV_RETRY
      continue
    fi
  fi
  T1=$(date +%s)
  if [ "$MODE" = "eval-only" ] || [ ! -f "$RDIR/all.hdf5" ]; then
    ROLLOUT_OK=1; break     # eval-only writes no all.hdf5 -- nothing to check
  fi
  if "$PY" soe_scripts/vis_validate.py "$RDIR/all.hdf5" 20 > "$RDIR/validate.log" 2>&1; then
    log "[1/3] rollout images HEALTHY (attempt $ATTEMPT, n_envs=$NE)"
    ROLLOUT_OK=1; break
  fi
  log "[1/3] RENDER CORRUPT (attempt $ATTEMPT, n_envs=$NE): $(grep -m1 CORRUPT "$RDIR/validate.log")"
  NE=$NENV_RETRY
done
[ $ROLLOUT_OK -eq 1 ] || { log "ROLLOUT FAILED - render corruption persisted after retry - see $RDIR/validate.log"; exit 1; }

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
  log "=== ROUND $TASK a=$A seed=$TSEED round=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
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
log "[2/3] DP retrain: ${EP}ep (adaptive, clamp 100..600) ckpt_every=$CKE seed=$TSEED ds=$RDIR/success_accum.hdf5 -> $OUTDP"
RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY train.py \
  --config-path configs --config-name base_dp_${TASK}_image \
  task.dataset_path="$RDIR/success_accum.hdf5" \
  task.train_filter_key=scout_aug \
  "${DPOPTS[@]}" \
  training.num_epochs=$EP \
  training.checkpoint_every=$CKE \
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
  RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY train.py \
    --config-path configs --config-name base_dp_${TASK}_image \
    task.dataset_path="$RDIR/success_accum.hdf5" \
    task.train_filter_key=scout_aug \
    "${DPOPTS[@]}" \
    training.num_epochs=$EP \
    training.checkpoint_every=$CKE \
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

# ---- [3/3] dyn retrain (SCOUT only; frozen past DYN_FREEZE_AFTER) -------- #
if [ "$A" != "DP" ]; then
if [ "$NUM" -gt "$DYN_FREEZE_AFTER" ]; then
  log "[3/3] dyn retrain SKIPPED (round=$NUM > DYN_FREEZE_AFTER=$DYN_FREEZE_AFTER -- frozen at last trained dyn; rollout already used it via walk-back)"
  T3=$T2
else
CFG=$OUTDYN/config.yaml
NEWDP=$(newest_ckpt "$OUTDP")
$PY - "$CFG" "$RDIR" "$OUTDYN" "$OUTDP" "$A" "$NUM" "$TSEED" "$WPROJ" "$TASK" "$CORE" <<'PYEOF'
import sys, yaml, glob, os, re
cfg_path, rdir, outdyn, outdp, a_tag, num, tseed, wproj, task, core_path = sys.argv[1:11]
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

with open(f"configs/vib_{task}_exp1.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["zarr_path"] = accum
cfg["dataset"]["feature_cache"] = True
ck = sorted(glob.glob(os.path.join(outdp, "checkpoints", "*.ckpt")),
            key=os.path.getmtime)
if ck:
    cfg["model"]["E_s"]["base_dp_ckpt"] = ck[-1]
cfg["seed"] = int(tseed)
cfg["cudnn_deterministic"] = True
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = f"{a_tag}-s{tseed}-round{num}"
cfg["wandb"]["project"] = wproj
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[dyn-cfg] seed={tseed} ds={accum} es={ck[-1] if ck else None} -> {outdyn}")
PYEOF
log "[3/3] dyn retrain: seed=$TSEED ds=$RDIR/all_accum.hdf5 es_base=${NEWDP:-base-config} -> $OUTDYN"
RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY -m scout.train_vib \
  --config "$CFG" \
  > "$DYNLOG" 2>&1
RC=$?; T3=$(date +%s)
log "[3/3] dyn retrain rc=$RC in $(( (T3-T2)/60 ))m$(( (T3-T2)%60 ))s"
[ $RC -ne 0 ] && { log "DYN RETRAIN FAILED - see $DYNLOG"; exit 1; }
fi
else
  log "[3/3] dyn retrain SKIPPED for a=DP (baseline never consumes the VIB)"
  T3=$T2
fi

log "=== ROUND $TASK a=$A seed=$TSEED round=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
