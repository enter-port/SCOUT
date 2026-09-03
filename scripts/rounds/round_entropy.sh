#!/bin/bash
# round_entropy.sh (2026-08-24) -- FORMAL entropy experiment round driver.
# COPY of round.sh (v3) for the user's formal protocol; round.sh stays
# untouched. Differences from round.sh:
#   * runs in the scout-entropy worktree (entropy cost code lives there);
#   * SCOUT arm explores with the ENTROPY COST (方案三, --guide atypical,
#     cap $ATT_CAP=2.5; scale 3.0 / gst 100 from configs/eval_<task>_entropy.yaml)
#     -- the DP arm stays --guide off;
#   * XMODE=soe by default, ETRIES=10 (10 explore retries on failed eval inits);
#   * DYN_FREEZE_AFTER=6 by default (dyn retrained EVERY round, user 2026-08-24);
#   * wandb minimal: ONLY eval/success_rate + explore/pass@10 + DP/loss,
#     DP/epoch + dyn/KL-loss, dyn/mse-loss, dyn/epoch (user: 其他都不要).
# Inherited from v3 unchanged: TSEED controls all training randomness
# (split/init/shuffle/crop + DP training.seed + dyn cfg.seed), CUDA
# determinism (cudnn_deterministic + CUBLAS_WORKSPACE_CONFIG), idempotent
# round 0 (seeded 20-of-200 split + base DP 600ep + dyn-base) SHARED by the
# seed's two arms, render-corruption guard with n_envs-halving retry.
#
# Usage:  round_entropy.sh <task> <BASE|SCOUT|DP> <num>
# Env:    GPU=<id> TSEED=<int> DATA_ROOT=<abs dir> (required)
#         ATT_CAP=2.5 DYN_FREEZE_AFTER=6 WPROJ=<wandb project>
# Layout: $DATA_ROOT/<task>/{rollout/,train/DP/,train/dyn/}.
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
DYN_FREEZE_AFTER=${DYN_FREEZE_AFTER:-6}   # formal: dyn EVERY round (user 2026-08-24)
ATT_CAP=${ATT_CAP:-2.5}         # entropy cost: KL-bonus cap kappa (calibrated)
SEED=42                       # eval phase: FIXED scene set every round (42..141)
NEXPLORE=${NEXPLORE:-100}
# XMODE (user 2026-08-23): explore protocol -- both settings coexist and are
# switchable by this one parameter.
#   fresh (default, v3 split): explore = NEXPLORE FRESH scenes (seed
#       NUM*1000+42) x ETRIES each; ALL explore trajs -> dyn data.
#   soe (SOE protocol): explore = retry ONLY the failed eval inits (the SAME
#       scenes/initial states as eval) x ETRIES each (default 5); DP data =
#       successful retries; dyn data = per failed init {successful retries
#       if any, else FIRST retry}. Fixed budgets: DP_EPOCHS_SOE=300,
#       DYN_EPOCHS_SOE=100 (fresh mode keeps adaptive DP / config dyn).
XMODE=${XMODE:-soe}
case "$XMODE" in fresh|soe) ;; *) echo "XMODE must be fresh or soe (got: $XMODE)"; exit 1 ;; esac
if [ "$XMODE" = soe ]; then ETRIES=${ETRIES:-10}; else ETRIES=${ETRIES:-1}; fi
NENV=50                        # 2026-08-26 user order: every arm (SCOUT+DP)
                               # runs env=50 (old default 12; render gate gone)
if [ "$A" = BASE ]; then ESEED=0
elif [ "$XMODE" = soe ]; then ESEED=$SEED        # explore targets the eval scene set's failed inits
else ESEED=$((NUM * 1000 + 42)); fi
if [ "$XMODE" = soe ]; then
  EXPLORE_ARGS=(--explore-mode rescue --explore-try-times "$ETRIES")
  EDESC="failed-of-eval(x$ETRIES)"
else
  EXPLORE_ARGS=(--explore-seed "$ESEED" --n-explore "$NEXPLORE" --explore-try-times "$ETRIES")
  EDESC="$ESEED($NEXPLORE x$ETRIES)"
fi

export MUJOCO_GL=egl
export TMPDIR=/tmp            # MUST be local (CPFS TMPDIR kills torch_shm_manager)
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # T2: deterministic cuBLAS GEMM
# spread offscreen rendering one GPU per chain (see rollout.py's env factory;
# 2026-08-22: everything-on-EGL-device-0 overloaded it into frame corruption)
export SCOUT_RENDER_GPU=$GPU
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout-entropy
PY=/root/workspace/baojiachun/.venv/bin/python
DATA=$DATA_ROOT
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
LOG=$DATA/$TASK/round.log
# wandb project: one project PER SEED (user 2026-08-24), e.g.
# WPROJ=CAN-8-24-entropy-s233; caller passes it via env.
WPROJ=${WPROJ:-${TASKUP}-8-24-entropy}
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
  OFFICIAL=/root/workspace/baojiachun/scout/data/robomimic/$TASK/ph/image_v141_abs.hdf5
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
      +logging.metric_prefix=DP/ +logging.wandb_minimal=true \
      logging.name=DP-BASE-round0 \
      logging.project=\'"$WPROJ"\' \
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
        +logging.metric_prefix=DP/ +logging.wandb_minimal=true \
        logging.name=DP-BASE-round0 \
        logging.project=\'"$WPROJ"\' \
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
cfg.setdefault("wandb", {})["name"] = "dyn-BASE-round0"
cfg["wandb"]["project"] = wproj
cfg["wandb"]["minimal"] = True
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

for f in configs/eval_${TASK}_entropy.yaml configs/vib_${TASK}_exp1.yaml \
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
# Render gate REMOVED (2026-08-26 user order): vis_validate's thresholds were
# calibrated on can (agentview tstd healthy 3.4-14.4 / noise 27-32); square's
# healthy demos measure 17.6-27.4 (both seeds' offline-rendered core data) and
# straddle the >20 noise line, so its CORRUPT verdicts on square were false
# positives. Single-shot rollout now; rc!=0 -> stop (no retry, no validation).
GUIDE=off; GEXTRA=()
[ "$A" = SCOUT ] && { GUIDE=atypical; GEXTRA=(--atypical-cap "$ATT_CAP"); }
NE=$NENV
if [ "${SKIP_ROLLOUT:-0}" = 1 ] && [ -f "$RDIR/all.hdf5" ]; then
  log "[1/3] SKIP_ROLLOUT=1: reusing existing $RDIR/all.hdf5"
else
  rm -f "$RDIR/all.hdf5" "$RDIR/success.hdf5"; rm -rf "$RDIR/log"
  log "[1/3] rollout guide=$GUIDE n_envs=$NE xmode=$XMODE eval=$SEED(100) explore=$EDESC dp=$DPCKPT vib=${VIBCKPT:-none} -> $RDIR"
  RUN env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout \
    --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num "$NUM" \
    --base-dp-ckpt "$DPCKPT" \
    --core-hdf5 "$CORE" \
    --guide "$GUIDE" --seed "$SEED" \
    --eval-seed "$SEED" \
    ${EXPLORE_ARGS[@]+"${EXPLORE_ARGS[@]}"} \
    --n-envs "$NE" \
    ${EVALONLY[@]+"${EVALONLY[@]}"} \
    ${VIBARGS[@]+"${VIBARGS[@]}"} \
    ${GEXTRA[@]+"${GEXTRA[@]}"} \
    --wandb-minimal \
    --output-dir "$RDIR" \
    --output-success "$RDIR/success.hdf5" \
    --output-all "$RDIR/all.hdf5" \
    --wandb-name "$WNAME" \
    --wandb-project "$WPROJ" \
    > "$RLOG" 2>&1
  RC=$?
  [ $RC -ne 0 ] && { log "[1/3] rollout rc=$RC (n_envs=$NE) -- see $RLOG"; exit 1; }
fi
T1=$(date +%s)

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
if [ "$XMODE" = soe ]; then
  EP=${DP_EPOCHS_SOE:-300}
  CKE=150                   # 300ep -> ckpts 149/299 (final epoch is saved)
  log "[2/3] DP retrain: ${EP}ep (soe fixed budget) ckpt_every=$CKE seed=$TSEED ds=$RDIR/success_accum.hdf5 -> $OUTDP"
else
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
fi
RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY train.py \
  --config-path configs --config-name base_dp_${TASK}_image \
  task.dataset_path="$RDIR/success_accum.hdf5" \
  task.train_filter_key=scout_aug \
  "${DPOPTS[@]}" \
  training.num_epochs=$EP \
  training.checkpoint_every=$CKE \
  dataloader.num_workers=8 dataloader.persistent_workers=true \
  +logging.metric_prefix=DP/ +logging.wandb_minimal=true \
  logging.name=$WNAME \
  logging.project=\'"$WPROJ"\' \
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
    +logging.metric_prefix=DP/ +logging.wandb_minimal=true \
    logging.name=$WNAME \
    logging.project=\'"$WPROJ"\' \
    hydra.run.dir="$OUTDP" \
    > "$DPLOG" 2>&1
  RC=$?; T2=$(date +%s)
  log "[2/3] DP retrain (workers=0 fallback) rc=$RC in $(( (T2-T1)/60 ))m$(( (T2-T1)%60 ))s"
fi
[ $RC -ne 0 ] && { log "DP RETRAIN FAILED - see $DPLOG"; exit 1; }

# disk hygiene (2026-08-29, POST_PRUNE=1): this round's success_accum was
# rebuilt fresh from core + per-round success.hdf5 (all sources kept) and no
# later round reads it. Remove it (+ its zarr cache) once retraining is done.
if [ "${POST_PRUNE:-0}" = "1" ] && [ -f "$RDIR/success_accum.hdf5" ]; then
  rm -f "$RDIR"/success_accum.hdf5 "$RDIR"/success_accum.hdf5.zarr.zip
  log "[prune] removed $RDIR/success_accum.hdf5(+zarr cache); sources success.hdf5 kept"
fi

# ---- [3/3] dyn retrain (SCOUT only; frozen past DYN_FREEZE_AFTER) -------- #
if [ "$A" != "DP" ]; then
if [ "$NUM" -gt "$DYN_FREEZE_AFTER" ]; then
  log "[3/3] dyn retrain SKIPPED (round=$NUM > DYN_FREEZE_AFTER=$DYN_FREEZE_AFTER -- frozen at last trained dyn; rollout already used it via walk-back)"
  T3=$T2
else
CFG=$OUTDYN/config.yaml
NEWDP=$(newest_ckpt "$OUTDP")
DYN_EPOCHS=0
[ "$XMODE" = soe ] && DYN_EPOCHS=${DYN_EPOCHS_SOE:-100}
$PY - "$CFG" "$RDIR" "$OUTDYN" "$OUTDP" "$A" "$NUM" "$TSEED" "$WPROJ" "$TASK" "$CORE" "$DYN_EPOCHS" <<'PYEOF'
import sys, yaml, glob, os, re
cfg_path, rdir, outdyn, outdp, a_tag, num, tseed, wproj, task, core_path, dyn_ep = sys.argv[1:12]
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
if int(dyn_ep) > 0:
    cfg["num_epochs"] = int(dyn_ep)   # soe fixed budget (default 100)
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = f"{a_tag}-s{tseed}-round{num}"
cfg["wandb"]["project"] = wproj
cfg["wandb"]["minimal"] = True
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[dyn-cfg] seed={tseed} ds={accum} es={ck[-1] if ck else None} -> {outdyn}")
PYEOF
log "[3/3] dyn retrain: seed=$TSEED ep=${DYN_EPOCHS:-cfg} ds=$RDIR/all_accum.hdf5 es_base=${NEWDP:-base-config} -> $OUTDYN"
RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY -m scout.train_vib \
  --config "$CFG" \
  > "$DYNLOG" 2>&1
RC=$?; T3=$(date +%s)
log "[3/3] dyn retrain rc=$RC in $(( (T3-T2)/60 ))m$(( (T3-T2)%60 ))s"
[ $RC -ne 0 ] && { log "DYN RETRAIN FAILED - see $DYNLOG"; exit 1; }
# disk hygiene (2026-08-29, POST_PRUNE=1): all_accum is rebuilt each round
# from the per-round all.hdf5 chain (sources kept); remove + featbank caches.
if [ "${POST_PRUNE:-0}" = "1" ] && [ -f "$RDIR/all_accum.hdf5" ]; then
  rm -f "$RDIR"/all_accum.hdf5 "$RDIR"/all_accum.hdf5.featbank.* "$RDIR"/all_accum.hdf5.zarr.zip
  log "[prune] removed $RDIR/all_accum.hdf5(+featbank caches); sources all.hdf5 kept"
fi
fi
else
  log "[3/3] dyn retrain SKIPPED for a=DP (baseline never consumes the VIB)"
  T3=$T2
fi

log "=== ROUND $TASK a=$A seed=$TSEED round=$NUM TOTAL: $(( (T3-T0)/60 ))m$(( (T3-T0)%60 ))s ==="
