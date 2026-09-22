#!/bin/bash
# round_thm3.sh (2026-09-23) -- THREADING-THM3 round driver. COPY of
# scripts/threading/round_thm2.sh with these deltas ONLY (user plan
# 2026-09-23, autonomous session -- "重启threading6臂,用新的explore success
# 记录方式+和AB实验中一样的训练轮数"):
#   * WPROJ default THREADING-THM3-s$TSEED;
#   * explore phase B runs with --stop-on-first-success (NEW recording:
#     an init's first success closes it -- ONE successful traj per solved
#     scene; pass@k unaffected);
#   * DP retrain budget = the B2 recipe: 600ep b64 (DP_EPOCHS_SOE default
#     600, DP_BS default 64, ckpt_every 300);
#   * dyn retrain budget = the A3 recipe: DYN_EPOCHS_SOE default 300
#     (beta stays 1.0e-5 on every dyn training, as in p2);
#   * ATY arm: per-round R-anchored eta recalibration (rcalib, unchanged
#     from p2) FOLLOWED by per-round kappa v2 C-anchored recalibration
#     (user order 2026-09-23 "如果重标定更好，则重启...用重标定 kappa"): C =
#     step-averaged uncapped KL, kappa' = kappa * C_target / C_mean (inverted
#     ratio, commit a6d004d method), C_target measured on the BASE pair at
#     BASE_ETA (the base's own rcalib eta from the p2 campaign), kappa0 =
#     previous round's kappa (kcal json -> rcalib json -> ATT_CAP fallback).
#     Calib jsons: calib_rcalib_r$NUM.json (eta) + calib_kcal_r$NUM.json
#     (kappa) at $RDIR level. kappa failure = round FATAL (same as eta).
# Everything else inherited VERBATIM from round_thm2.sh: three arms (DP /
# ATY / ORBIT), ETRIES=5, SHARD_P=8 x25env, two-phase sharded rescue, TSEED
# determinism, walk-back, anti-deadlock retrain, heartbeat reporter, wandb
# backfill, flush-every, BETA on every dyn training.
#
# Dose knobs are DEFAULTS ONLY -- for ATY the per-round calib OVERWRITES
# ATY_SCALE (eta) and ATT_CAP (kappa); DP arm is guide=off.
#
# Usage:  round_thm3.sh <threading> <BASE|ATY|ORBIT|DP> <num> [full|eval-only|eval-pk]
# Env:    GPU=<id> TSEED=<int> DATA_ROOT=<abs dir> (required)
#         WPROJ=<wandb project> ATT_CAP=2.5 ATY_SCALE=1.0 (fallbacks only)
#         ETA0=1.0 (rcalib eta anchor for round 1)
#         BASE_ETA=<float> (kappa v2 C-anchor on the base pair -- per-seed
#         p2 rcalib value; REQUIRED for ATY arms)
#         BETA=1.0e-5 ORB_* DYN_FREEZE_AFTER=6 CORE_N=20 DP_BS=64
#         DP_EPOCHS_SOE=600 DYN_EPOCHS_SOE=300
#         SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 ETRIES=5 FLUSH_EVERY=100
# Layout: $DATA_ROOT/threading/{rollout/,train/DP/,train/dyn/}.
set -u

TASK=${1:?usage: round_thm3.sh <task> <BASE|ATY|ORBIT|DP> <num> [mode]}
A=${2:?usage: round_thm3.sh <task> <BASE|ATY|ORBIT|DP> <num> [mode]}
NUM=${3:?usage: round_thm3.sh <task> <BASE|ATY|ORBIT|DP> <num> [mode]}
case "$TASK" in
  can)      TASKUP=CAN ;;
  square)   TASKUP=SQUARE ;;
  transport) TASKUP=TRANSPORT ;;
  tool_hang) TASKUP=TOOLHANG ;;
  threading) TASKUP=THREADING ;;
  *) echo "task must be can, square, transport, tool_hang or threading (got: $TASK)"; exit 1 ;;
esac
case "$A" in
  BASE) [ "$NUM" = 0 ] || { echo "arm BASE only valid with num 0"; exit 1; } ;;
  DP|ATY|ORBIT) [ "$NUM" -ge 1 ] 2>/dev/null || { echo "arm ATY/ORBIT/DP needs num>=1"; exit 1; } ;;
  *) echo "a must be BASE, ATY, ORBIT or DP (got: $A)"; exit 1 ;;
esac
MODE=${4:-full}
case "$MODE" in
  full|eval-only|eval-pk) ;;
  *) echo "mode must be full, eval-only or eval-pk"; exit 1 ;;
esac

GPU=${GPU:?set GPU=<cuda id>}
TSEED=${TSEED:?set TSEED=<training seed -- controls split/init/shuffle/crop>}
DATA_ROOT=${DATA_ROOT:?set DATA_ROOT=<experiment dir>}
DYN_FREEZE_AFTER=${DYN_FREEZE_AFTER:-6}   # 6-round chain: dyn EVERY full round
ATT_CAP=${ATT_CAP:-2.5}         # kappa anchor (constant; rcalib carries it over)
ATY_SCALE=${ATY_SCALE:-1.0}     # fallback only -- rcalib OVERWRITES per round (R->0.01)
ETA0=${ETA0:-1.0}               # rcalib eta anchor for ROUND 1 (user 2026-09-22)
BETA=${BETA:-1.0e-5}            # VIB beta for EVERY dyn training (user 2026-09-22)
ORB_LAM=${ORB_LAM:-0.5}         # orbit: Newton feedback gain lambda
ORB_DELTA=${ORB_DELTA:-0.25}    # orbit: phase-switch buffer delta
ORB_SIGMA=${ORB_SIGMA:-0.05}    # orbit: tangential noise std base (th95)
ORB_SIGMA_DECAY=${ORB_SIGMA_DECAY:-0.5}  # orbit: per-round sigma decay (v3: sigma*0.5^(r-1))
ORB_ANNEAL=${ORB_ANNEAL:-2}     # orbit: tangent-noise annealing exponent p (v3)
CORE_N=${CORE_N:-20}            # round0 split: 20 of 1000 mg demos (user 2026-09-16)
DP_BS=${DP_BS:-64}              # [2/3] DP retrain batch = B2 recipe (THM3: b64, same as BASE)
B256ARGS=(dataloader.batch_size="$DP_BS" val_dataloader.batch_size="$DP_BS")
SHARD_P=${SHARD_P:-8}           # explore shard workers per arm (th95)
SHARD_ENVS=${SHARD_ENVS:-25}    # envs per shard worker
EVALNENV=${EVALNENV:-25}        # eval-phase (monolithic) n_envs
FLUSH_EVERY=${FLUSH_EVERY:-100} # TrajSpool staging flush (OOM fix 6f3d844)
SEED=42                       # eval phase: FIXED scene set every round (42..141)
XMODE=${XMODE:-soe}
case "$XMODE" in soe) ;; *) echo "XMODE must be soe (got: $XMODE)"; exit 1 ;; esac
ETRIES=${ETRIES:-5}

export MUJOCO_GL=egl
export TMPDIR=/tmp            # MUST be local (CPFS TMPDIR kills torch_shm_manager)
export CUBLAS_WORKSPACE_CONFIG=:4096:8   # T2: deterministic cuBLAS GEMM
export SCOUT_RENDER_GPU=$GPU
export PYTHONUNBUFFERED=1
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
DATA=$DATA_ROOT
TDP=$DATA/$TASK/train/DP
TDYN=$DATA/$TASK/train/dyn
CORE=$DATA/$TASK/rollout/${TASK}_core.hdf5
LOG=$DATA/$TASK/round.log
WPROJ=${WPROJ:-THREADING-THM3-s${TSEED}}
mkdir -p "$TDP" "$TDYN" "$(dirname "$CORE")"
cd "$REPO" || exit 1
exec 3>&1

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

DPOPTS=(training.seed="$TSEED" training.resume=False training.rollout_every=0
        training.sample_every=100 training.cudnn_benchmark=false
        +training.cudnn_deterministic=true training.device=cuda:0)

# ============================================================================ #
# ROUND 0 (arm BASE): seeded split + base DP + dyn-base (T5)
# ============================================================================ #
if [ "$A" = BASE ]; then
  OFFICIAL=/root/workspace/baojiachun/scout/data/robomimic/threading/mg/threading_d0.hdf5
  [ -f "configs/vib_${TASK}_exp1.yaml" ] || { echo "missing configs/vib_${TASK}_exp1.yaml"; exit 1; }
  [ -f "configs/base_dp_${TASK}_image.yaml" ] || { echo "missing base_dp config"; exit 1; }
  [ -f "$OFFICIAL" ] || { echo "missing official mg-generated dataset: $OFFICIAL"; exit 1; }

  T0=$(date +%s)
  log "=== ROUND0 $TASK seed=$TSEED START (GPU$GPU; official=$OFFICIAL; beta=$BETA) ==="

  if [ -f "$CORE" ]; then
    log "[0/3] core exists -- skip split ($CORE; abs conversion done by thread_prep_data.sh)"
  else
    # P2-3 (review 2026-09-16): split_core.py alone yields a core WITHOUT
    # abs_actions -> DP training would fail much later with a confusing
    # loader error. The canonical path is thread_prep_data.sh (split + abs
    # conversion); never bootstrap a raw split here.
    log "[0/3] FATAL: $CORE missing -- run scripts/threading/thread_prep_data.sh <seed> first (split + abs conversion)"
    exit 1
  fi

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
    $PY - "$CFG" "$CORE" "$TDYN/dyn-base" "$DPB" "$TSEED" "$WPROJ" "$TASK" "$BETA" <<'PYEOF'
import sys, yaml
cfg_path, zarr, outdyn, dpb, tseed, wproj, task, beta = sys.argv[1:9]
with open(f"configs/vib_{task}_exp1.yaml") as f:
    cfg = yaml.safe_load(f)
cfg["dataset"]["zarr_path"] = zarr
cfg["dataset"]["feature_cache"] = True
cfg["model"]["E_s"]["base_dp_ckpt"] = dpb
cfg["seed"] = int(tseed)
cfg["cudnn_deterministic"] = True
cfg["beta"] = float(beta)
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = "dyn-BASE-round0"
cfg["wandb"]["project"] = wproj
cfg["wandb"]["minimal"] = True
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[dyn-base-cfg] seed={tseed} ds={zarr} es={dpb} beta={beta} -> {outdyn}")
PYEOF
    log "[2/3] dyn-base: seed=$TSEED deterministic ds=$CORE es_base=${DPB##*/} beta=$BETA -> $TDYN/dyn-base"
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
# ROUNDS 1..N (arm ATY / ORBIT / DP)
# ============================================================================ #
RDIR=$DATA/$TASK/rollout/$A-exp$NUM
OUTDP=$TDP/DP-$A-exp$NUM
OUTDYN=$TDYN/dyn-$A-exp$NUM
mkdir -p "$RDIR" "$OUTDP" "$OUTDYN"
RLOG=$RDIR/rollout.stdout; DPLOG=$OUTDP/train.log; DYNLOG=$OUTDYN/train.log
WNAME_BASE=${WNAME_BASE:-$A}
WNAME=${WNAME_BASE}-round${NUM}

for f in configs/eval_${TASK}_entropy.yaml configs/vib_${TASK}_exp1.yaml \
         configs/base_dp_${TASK}_image.yaml scripts/infra/shard_rollout.sh \
         scripts/coffee/cfk_rcalib.py; do
  [ -f "$f" ] || { echo "missing $f"; exit 1; }
done
[ -n "$(newest_ckpt "$TDP/DP-base")" ] || { echo "no DP-base ckpt (run: round_thm3.sh $TASK BASE 0)"; exit 1; }
[ -n "$(newest_vib "$TDYN/dyn-base")" ] || [ "$A" = DP ] \
  || { echo "no dyn-base ckpt (run: round_thm3.sh $TASK BASE 0)"; exit 1; }

# ---- resolve this round's rollout inputs (walk-back, fallback to base) ---- #
PREV=$((NUM - 1))
DPROLL=$TDP/DP-base
for e in $(seq "$PREV" -1 1); do
  if [ -n "$(newest_ckpt "$TDP/DP-$A-exp$e")" ]; then DPROLL=$TDP/DP-$A-exp$e; break; fi
done
DPCKPT=$(newest_ckpt "$DPROLL")
[ -n "$DPCKPT" ] || { log "FATAL: no DP ckpt under $DPROLL"; exit 1; }

VIBARGS=()
VIBDIR=""
if [ "$A" != "DP" ]; then
  VIBDIR=$TDYN/dyn-base
  for e in $(seq "$PREV" -1 1); do
    if [ -n "$(newest_vib "$TDYN/dyn-$A-exp$e")" ]; then VIBDIR=$TDYN/dyn-$A-exp$e; break; fi
  done
  VIBCKPT=$(newest_vib "$VIBDIR")
  [ -n "$VIBCKPT" ] || { log "FATAL: no VIB ckpt under $VIBDIR"; exit 1; }
  VIBARGS=(--vib-ckpt "$VIBCKPT")
fi

# ---- [thm2 delta] ATY arm: R-anchored eta re-calibration (per round) ------ #
# Runs BEFORE GEXTRA so the re-anchored dose is what phase A/B actually
# carry (the eval phase is unguided regardless; the dose only bites in
# phase B rescue). Artifacts at $RDIR level (calib_rcalib_r$NUM.json +
# calib.stdout) survive the phase-A restart rm of $RDIR/log.
# eta_prev/kappa_prev: round 1 -> ETA0/ATT_CAP; round N>1 -> the previous
# ATY round's calib json. Deterministic: frozen core batch, same x_T draw
# (torch.manual_seed(0)) across all eta evaluations. ATY_SCALE is
# OVERWRITTEN with the converged eta; kappa stays at kappa_prev.
if [ "$A" = ATY ]; then
  ETA_PREV=$ETA0
  KAP_PREV=$ATT_CAP
  if [ "$NUM" -gt 1 ]; then
    PREVJ=$(dirname "$RDIR")/ATY-exp$((NUM-1))/calib_rcalib_r$((NUM-1)).json
    PREVK=$(dirname "$RDIR")/ATY-exp$((NUM-1))/calib_kcal_r$((NUM-1)).json
    if [ -f "$PREVJ" ]; then
      ETA_PREV=$($PY -c "import json;print('%.6g'%json.load(open('$PREVJ'))['eta'])")
      # [thm3] kappa_prev: prefer the previous round's KAPPA calib json -- the
      # rcalib json only carries kappa through unchanged
      if [ -f "$PREVK" ]; then
        KAP_PREV=$($PY -c "import json;print('%.6g'%json.load(open('$PREVK'))['kappa'])")
      else
        KAP_PREV=$($PY -c "import json;print('%.6g'%json.load(open('$PREVJ'))['kappa'])")
      fi
    else
      log "[calib-rcalib] round=$NUM WARN no prev calib json ($PREVJ) -- falling back to ETA0/ATT_CAP"
    fi
  fi
  if [ "${DRY_RUN:-0}" = 1 ]; then
    log "[calib-rcalib] DRY_RUN: planned rcalib (dp=$DPCKPT vib=$VIBCKPT eta_prev=$ETA_PREV kappa_prev=$KAP_PREV); ATY_SCALE/ATT_CAP stay $ATY_SCALE/$ATT_CAP"
  else
    CALIBJ=$RDIR/calib_rcalib_r${NUM}.json
    env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/coffee/cfk_rcalib.py \
      --eval-config configs/eval_${TASK}_entropy.yaml \
      --dp-ckpt "$DPCKPT" \
      --vib-ckpt "$VIBCKPT" \
      --core-hdf5 "$CORE" \
      --eta-prev "$ETA_PREV" --kappa-prev "$KAP_PREV" \
      --out "$CALIBJ" > "$RDIR/calib.stdout" 2>&1
    RC=$?
    [ $RC -ne 0 ] && { log "[calib-rcalib] FATAL rc=$RC -- see $RDIR/calib.stdout"; exit 1; }
    ATY_SCALE=$($PY -c "import json;print('%.6g'%json.load(open('$CALIBJ'))['eta'])")
    ATT_CAP=$($PY -c "import json;print('%.6g'%json.load(open('$CALIBJ'))['kappa'])")
    DOSEV=$($PY -c "import json;d=json.load(open('$CALIBJ'));print('R_prev=%.4g R_final=%.4g converged=%s n_upd=%d'%(d['R_prev'],d['R_mean'],d['converged'],d['n_updates']))")
    log "[calib-rcalib] round=$NUM $DOSEV -> ATY_SCALE=$ATY_SCALE (json: $CALIBJ)"

    # ---- [thm3 delta] kappa v2 C-anchored recalibration ----------------- #
    # AFTER the eta rcalib so C is measured at the eta this round carries.
    # C = step-averaged uncapped KL on the frozen core batch; kappa' =
    # kappa * C_target/C_mean (inverted ratio); C_target on the BASE pair at
    # BASE_ETA (per-seed p2 anchor); kappa0 = kappa_prev. Non-arrival within
    # max-repeat takes the LAST value (rcalib convention).
    [ -n "${BASE_ETA:-}" ] || { log "[calib-kcal] FATAL BASE_ETA not set (per-seed p2 anchor)"; exit 1; }
    KCALJ=$RDIR/calib_kcal_r${NUM}.json
    env CUDA_VISIBLE_DEVICES=$GPU $PY scripts/threading/thm2_c_calib.py \
      --eval-config configs/eval_${TASK}_entropy.yaml \
      --core-hdf5 "$CORE" \
      --base-dp-ckpt "$(newest_ckpt "$TDP/DP-base")" \
      --base-vib-ckpt "$(newest_vib "$TDYN/dyn-base")" \
      --base-eta "$BASE_ETA" \
      --round-dp-ckpt "$DPCKPT" --round-vib-ckpt "$VIBCKPT" \
      --round-eta "$ATY_SCALE" \
      --kappa0 "$KAP_PREV" \
      --out "$KCALJ" > "$RDIR/kcal.stdout" 2>&1
    RC=$?
    [ $RC -ne 0 ] && { log "[calib-kcal] FATAL rc=$RC -- see $RDIR/kcal.stdout"; exit 1; }
    ATT_CAP=$($PY -c "import json;print('%.6g'%json.load(open('$KCALJ'))['kappa'])")
    log "[calib-kcal] round=$NUM -> ATT_CAP=$ATT_CAP (json: $KCALJ)"
  fi
fi

T0=$(date +%s)
DOSELOG=""
[ "$A" = ATY ] && DOSELOG=" aty_scale=$ATY_SCALE cap=$ATT_CAP"
log "=== ROUND $TASK a=$A seed=$TSEED round=$NUM mode=$MODE START (GPU$GPU; rollout DP=$DPROLL vib=${VIBDIR:-none}; eval_seed=$SEED$DOSELOG) ==="

# ---- [1/3] rollout: TWO-PHASE SHARDED rescue (eval monolithic + explore P workers)
GUIDE=off; GEXTRA=()
case "$A" in
  ATY)   GUIDE=atypical
         GEXTRA=(--atypical-cap "$ATT_CAP" --guidance-scale "$ATY_SCALE") ;;
  ORBIT) GUIDE=orbit
         GEXTRA=(--atypical-cap "$ATT_CAP" --guidance-scale "$ATY_SCALE"
                 --orbit-lam "$ORB_LAM" --orbit-delta "$ORB_DELTA"
                 --orbit-sigma "$ORB_SIGMA" --orbit-sigma-decay "$ORB_SIGMA_DECAY"
                 --orbit-round "$NUM" --orbit-fb-clamp soft
                 --orbit-noise-anneal "$ORB_ANNEAL") ;;
esac
EXPLORE_JSON=$RDIR/log/${TASK}_${A}_explore_exp${NUM}.json

if [ "${SKIP_ROLLOUT:-0}" = 1 ] && [ -f "$RDIR/all.hdf5" ]; then
  log "[1/3] SKIP_ROLLOUT=1: reusing existing $RDIR/all.hdf5"
elif [ "$MODE" = "eval-only" ]; then
  log "[1/3] eval-only rollout guide=$GUIDE n_envs=$EVALNENV eval=$SEED(100) dp=$DPCKPT vib=${VIBCKPT:-none} -> $RDIR"
  RUN env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout \
    --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num "$NUM" \
    --base-dp-ckpt "$DPCKPT" \
    --core-hdf5 "$CORE" \
    --guide "$GUIDE" --seed "$SEED" \
    --eval-seed "$SEED" \
    --eval-only \
    --n-envs "$EVALNENV" \
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
  [ $RC -ne 0 ] && { log "[1/3] eval-only rollout rc=$RC -- see $RLOG"; exit 1; }
else
  # -- phase A: eval + freeze the failed set (monolithic, carries the wandb run)
  # run_rollout.py:437 hardcodes the eval-json tag: "SCOUT" if guided else "DP"
  # -- renamed arms (ATY/ORBIT) must fall back to the SCOUT-named file or the
  # resume path never fires and a crashed phase B would rerun eval + rm the
  # wandb_run_id carrier.
  EVALJSON=$RDIR/log/${TASK}_${A}_rollout_exp${NUM}.json
  [ -f "$EVALJSON" ] || EVALJSON=$RDIR/log/${TASK}_SCOUT_rollout_exp${NUM}.json
  if [ -f "$RDIR/failed.json" ] \
     && [ -f "$EVALJSON" ] \
     && [ ! -f "$RDIR/all.hdf5" ]; then
    log "[1/3a] resume: failed.json + eval json intact from a crashed phase B -- skip eval, reuse frozen failed set"
  else
  rm -f "$RDIR/all.hdf5" "$RDIR/success.hdf5" "$RDIR/failed.json" "$RDIR"/success.hdf5.spool "$RDIR"/all.hdf5.spool; rm -rf "$RDIR/log"
  log "[1/3a] eval phase guide=$GUIDE n_envs=$EVALNENV eval=$SEED(100) dp=$DPCKPT -> failed.json"
  RUN env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU $PY -m scout.eval.run_rollout \
    --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num "$NUM" \
    --base-dp-ckpt "$DPCKPT" \
    --core-hdf5 "$CORE" \
    --guide "$GUIDE" --seed "$SEED" \
    --eval-seed "$SEED" \
    --explore-mode rescue --eval-only --save-failed-set "$RDIR/failed.json" \
    --n-envs "$EVALNENV" \
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
  [ $RC -ne 0 ] && { log "[1/3a] eval rollout rc=$RC -- see $RLOG"; exit 1; }
  [ -f "$RDIR/failed.json" ] || [ "${DRY_RUN:-0}" = 1 ] || { log "[1/3a] FATAL: failed.json not written"; exit 1; }
  fi

  # -- phase B: sharded rescue explore (P workers x SHARD_ENVS envs, one GPU)
  log "[1/3b] explore phase: $SHARD_P workers x n_envs=$SHARD_ENVS guide=$GUIDE failed-of-eval(x$ETRIES) flush_every=$FLUSH_EVERY -> merged $EXPLORE_JSON"
  HB_PID=""
  if [ "${DRY_RUN:-0}" != 1 ]; then
    RID_HB=$($PY - "$RDIR/log" <<'PYEOF'
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
    if [ -n "$RID_HB" ]; then
      pkill -f "shard_heartbeat.py.*--match $RDIR " 2>/dev/null && sleep 2
      nohup $PY scripts/infra/shard_heartbeat.py --project "$WPROJ" --run-id "$RID_HB" \
        --shard-glob "$RDIR/log/shard*.stdout" --match "$RDIR" \
        --stop-file "$RDIR/all.hdf5" --log-file "$RDIR/heartbeat.log" \
        >> "$RDIR/heartbeat.stdout" 2>&1 3>&- &
      HB_PID=$!
      log "[1/3-hb] heartbeat pid=$HB_PID -> $RDIR/heartbeat.log + wandb explore_hb/* (run $RID_HB)"
    else
      log "[1/3-hb] WARN: no wandb_run_id found in $RDIR/log -- reporter skipped"
    fi
  fi
  RUN env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU PYTHON=$PY CLEANUP_SHARDS=1 \
    bash scripts/infra/shard_rollout.sh "$SHARD_P" \
    "$EXPLORE_JSON" \
    "$RDIR/success.hdf5" \
    "$RDIR/all.hdf5" \
    "$CORE" \
    -- \
    --config configs/eval_${TASK}_entropy.yaml --task "$TASK" --exp-num "$NUM" \
    --base-dp-ckpt "$DPCKPT" \
    --core-hdf5 "$CORE" \
    --guide "$GUIDE" --seed "$SEED" \
    --eval-seed "$SEED" \
    --explore-mode rescue --explore-try-times "$ETRIES" --stop-on-first-success \
    --failed-set-json "$RDIR/failed.json" \
    --n-envs "$SHARD_ENVS" \
    --flush-every "$FLUSH_EVERY" \
    ${VIBARGS[@]+"${VIBARGS[@]}"} \
    ${GEXTRA[@]+"${GEXTRA[@]}"} \
    --no-wandb \
    --output-dir "$RDIR" \
    --output-success "$RDIR/success.hdf5" \
    --output-all "$RDIR/all.hdf5"
  RC=$?
  if [ -n "$HB_PID" ]; then
    kill "$HB_PID" 2>/dev/null; wait "$HB_PID" 2>/dev/null
    log "[1/3-hb] heartbeat stopped (tail: $(tail -1 "$RDIR/heartbeat.log" 2>/dev/null))"
  fi
  [ $RC -ne 0 ] && { log "[1/3b] sharded explore rc=$RC -- see $RDIR/shard*.stdout"; exit 1; }
  [ -f "$RDIR/all.hdf5" ] || [ "${DRY_RUN:-0}" = 1 ] || { log "[1/3b] FATAL: merged all.hdf5 missing"; exit 1; }

  # -- wandb backfill: explore/pass@{ETRIES} into the phase-A run (label from
  #    the json's explore_try_times; json key pass_at_5 is legacy naming)
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
    log "[wandb] phase-A run $WPROJ/$WNAME id=$RID (DP+dyn will resume it; backfilling explore keys)"
    $PY - "$RID" "$WPROJ" "$EXPLORE_JSON" <<'PYEOF' || log "[wandb] WARN: explore backfill failed (non-fatal)"
import sys, os, json
rid, proj, jpath = sys.argv[1:4]
d = json.load(open(jpath))
import wandb
etry = d.get("explore_try_times")   # json key pass_at_5 is legacy; value = pass@ETRIES
pk = f"explore/pass@{etry}" if etry else "explore/pass@5"
m = {pk: d.get("pass_at_5"),
     "explore/rescued": d.get("exploration_rescued")}
m = {k: v for k, v in m.items() if isinstance(v, (int, float))}
run = wandb.init(id=rid, project=proj, resume="must")
wandb.log(m)
wandb.finish()
print(f"[wandb-backfill] {m}")
PYEOF
  else
    log "[wandb] WARN: no wandb_run_id in rollout jsons -- retrains will start their own runs"
  fi
fi
T1=$(date +%s)

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

# ---- eval-only / eval-pk round: measurement done, skip accum + retrains --- #
# eval-pk (2026-09-08 口径定案): final round must carry SR AND pass@K. It
# already ran the FULL two-phase rescue rollout above (phase A = eval
# first-try SR + frozen failed set; phase B = sharded rescue x$ETRIES ->
# merged json pass_at_5 = (bs+rescued)/n), so only the retrains are skipped
# -- unlike eval-only, which rolls 1 try/scene and emits SR alone.
if [ "$MODE" != "full" ]; then
  log "[2/3]+[3/3] SKIPPED (MODE=$MODE: measurement only)"
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
  EP=${DP_EPOCHS_SOE:-600}
  CKE=300                   # [thm3] B2 recipe: 600ep b64 -> ckpts 299/599
  log "[2/3] DP retrain: ${EP}ep bs=$DP_BS (soe fixed budget) ckpt_every=$CKE seed=$TSEED ds=$RDIR/success_accum.hdf5 -> $OUTDP"
else
  EP=600; CKE=300
  log "[2/3] DP retrain: ${EP}ep bs=$DP_BS ckpt_every=$CKE seed=$TSEED ds=$RDIR/success_accum.hdf5 -> $OUTDP"
fi
RUN env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_RUN_ID="$RID" WANDB_RESUME=must $PY train.py \
  --config-path configs --config-name base_dp_${TASK}_image \
  task.dataset_path="$RDIR/success_accum.hdf5" \
  task.train_filter_key=scout_aug \
  "${DPOPTS[@]}" \
  training.num_epochs=$EP \
  training.checkpoint_every=$CKE \
  "${B256ARGS[@]}" \
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
    "${B256ARGS[@]}" \
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

# ---- [3/3] dyn retrain (ATY/ORBIT only; frozen past DYN_FREEZE_AFTER) ----- #
if [ "$A" != "DP" ]; then
if [ "$NUM" -gt "$DYN_FREEZE_AFTER" ]; then
  log "[3/3] dyn retrain SKIPPED (round=$NUM > DYN_FREEZE_AFTER=$DYN_FREEZE_AFTER -- frozen at last trained dyn; rollout already used it via walk-back)"
  T3=$T2
else
CFG=$OUTDYN/config.yaml
NEWDP=$(newest_ckpt "$OUTDP")
DYN_EPOCHS=0
[ "$XMODE" = soe ] && DYN_EPOCHS=${DYN_EPOCHS_SOE:-300}   # [thm3] A3 recipe
$PY - "$CFG" "$RDIR" "$OUTDYN" "$OUTDP" "$A" "$NUM" "$TSEED" "$WPROJ" "$TASK" "$CORE" "$DYN_EPOCHS" "$WNAME" "$BETA" <<'PYEOF'
import sys, yaml, glob, os, re
cfg_path, rdir, outdyn, outdp, a_tag, num, tseed, wproj, task, core_path, dyn_ep, wname, beta = sys.argv[1:14]
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
cfg["beta"] = float(beta)
if int(dyn_ep) > 0:
    cfg["num_epochs"] = int(dyn_ep)   # soe fixed budget (default 100)
cfg["save_dir"] = outdyn
cfg.setdefault("wandb", {})["name"] = wname
cfg["wandb"]["project"] = wproj
cfg["wandb"]["minimal"] = True
with open(cfg_path, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[dyn-cfg] seed={tseed} ds={accum} es={ck[-1] if ck else None} beta={beta} -> {outdyn} name={wname}")
PYEOF
log "[3/3] dyn retrain: seed=$TSEED ep=${DYN_EPOCHS:-cfg} beta=$BETA ds=$RDIR/all_accum.hdf5 es_base=${NEWDP:-base-config} -> $OUTDYN"
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
