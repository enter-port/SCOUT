#!/bin/bash
# thm3_ab.sh -- stage-1 arms of the THM3 undertraining study (user plan
# 2026-09-23, autonomous). Five arms, one GPU each (port-1024 machine):
#   A1  : dyn 600ep  E_s=DP-base(599)     data=core20
#         (budget curve: 100ep=2.591, 300ep=9.553 already measured)
#   A2L : dyn 300ep  E_s=DP-ATY-exp1(299) data=dedup73 all_accum
#   A2H : dyn 600ep  same                 (dedup73@100ep=2.143 already measured)
#   B1  : DP retrain 900ep b256 on dedup43 success_accum -> E_s -> core-only
#         dyn 100ep -> probe    (step-count axis: ~48k steps vs base's 42k)
#   B2  : DP retrain 600ep b64  on dedup43 success_accum -> E_s -> core-only
#         dyn 100ep -> probe    (recipe axis: exactly the base DP recipe)
# Probe = cfk_grad_probe_arm.py fixed core batch (|g|@0.25sigma + far-field
# KL), tag thm3_<ARM>. Run thm3_ab_prep.sh ONCE before the B arms.
# usage: bash scripts/threading/thm3_ab.sh <A1|A2L|A2H|B1|B2> <GPU>
set -u
ARM=${1:?arm}
GPU=${2:?gpu}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_22_threading_mg_p2
S=$CAMP/THREADING-MG-p2-s233/threading
EXP=$CAMP/PROBE_AB
cd "$ROOT" || exit 1
CORE=$S/rollout/threading_core.hdf5
ESBASE=$S/train/DP/DP-base/checkpoints/599.ckpt
ESATY1=$S/train/DP/DP-ATY-exp1/checkpoints/299.ckpt
DEDUP73=$CAMP/PROBE_DEDUP/all_accum_dedup_r1.hdf5
ACCUM43=$EXP/succ_accum_dedup_r1.hdf5
newest_ckpt(){ ls -t "$1"/checkpoints/*.ckpt 2>/dev/null | head -1; }
newest_vib(){  ls -t "$1"/*/scout_vib.ckpt  2>/dev/null | head -1; }

mk_vib_cfg(){  # args: cfg_out, data, es_ckpt, epochs, save_dir
$PY - "$@" <<'PYEOF'
import sys, yaml
out, ds, es, ep, sd = sys.argv[1:6]
cfg = yaml.safe_load(open("configs/vib_threading_exp1.yaml"))
cfg["dataset"]["zarr_path"] = ds
cfg["dataset"]["feature_cache"] = True
cfg["model"]["E_s"]["base_dp_ckpt"] = es
cfg["seed"] = 233
cfg["cudnn_deterministic"] = True
cfg["num_epochs"] = int(ep)
cfg["beta"] = 1.0e-5
cfg["save_dir"] = sd
cfg["use_wandb"] = False
cfg["wandb"] = {"name": "dyn-" + sd.rstrip("/").split("/")[-1],
                "project": "offline", "minimal": True}
yaml.safe_dump(cfg, open(out, "w"), sort_keys=False)
print(f"[cfg] ep={ep} beta=1e-5 seed=233 ds={ds} es={es}")
PYEOF
}

case "$ARM" in
# ---------------- dyn-budget arms (A) ---------------- #
A1|A2L|A2H)
  case "$ARM" in
    A1)  EPO=600; ES=$ESBASE; DS=$CORE;    OUT=$EXP/train_A1 ;;
    A2L) EPO=300; ES=$ESATY1; DS=$DEDUP73; OUT=$EXP/train_A2L ;;
    A2H) EPO=600; ES=$ESATY1; DS=$DEDUP73; OUT=$EXP/train_A2H ;;
  esac
  CFG=$EXP/vib_${ARM}.yaml
  mk_vib_cfg "$CFG" "$DS" "$ES" "$EPO" "$OUT"
  TMO=14400; [ "$EPO" = 300 ] && TMO=7200
  echo "[$ARM] dyn train ${EPO}ep start $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout $TMO \
    $PY -m scout.train_vib --config "$CFG" > $EXP/train_${ARM}.log 2>&1
  RC=$?
  echo "[$ARM] dyn train rc=$RC $(date '+%F %T')"
  [ $RC -ne 0 ] && { tail -8 $EXP/train_${ARM}.log; exit 1; }
  VIB=$(newest_vib "$OUT")
  ;;
# ---------------- DP-budget arms (B) ---------------- #
B1|B2)
  case "$ARM" in
    B1) EPO=900; DPBS=256; OUT=$EXP/train_DP_B1 ;;
    B2) EPO=600; DPBS=64;  OUT=$EXP/train_DP_B2 ;;
  esac
  [ -f "$ACCUM43" ] || { echo "[$ARM] FATAL missing $ACCUM43 (run thm3_ab_prep.sh first)"; exit 1; }
  echo "[$ARM] DP retrain ${EPO}ep b${DPBS} start $(date '+%F %T')"
  # wandb.init runs unconditionally in the LPB workspace and even a disabled
  # run calls login in this wandb version -> source the server key; the
  # WANDB_MODE=disabled keeps the run off the cloud entirely.
  set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
  env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 WANDB_MODE=disabled \
    timeout 43200 $PY train.py \
    --config-path configs --config-name base_dp_threading_image \
    task.dataset_path="$ACCUM43" task.train_filter_key=scout_aug \
    training.seed=233 training.resume=False training.rollout_every=0 \
    training.sample_every=100 training.cudnn_benchmark=false \
    +training.cudnn_deterministic=true training.device=cuda:0 \
    training.num_epochs=$EPO training.checkpoint_every=300 \
    dataloader.batch_size=$DPBS val_dataloader.batch_size=$DPBS \
    dataloader.num_workers=8 dataloader.persistent_workers=true \
    hydra.run.dir="$OUT" > $EXP/train_${ARM}_dp.log 2>&1
  RC=$?
  echo "[$ARM] DP retrain rc=$RC $(date '+%F %T')"
  [ $RC -ne 0 ] && { tail -8 $EXP/train_${ARM}_dp.log; exit 1; }
  ES=$(newest_ckpt "$OUT")
  [ -n "$ES" ] || { echo "[$ARM] FATAL no DP ckpt under $OUT"; exit 1; }
  echo "[$ARM] E_s ckpt = $ES"
  # core-only dyn 100ep on the new E_s (same protocol as the 0.398/2.591 arms)
  OUTD=$EXP/train_dyn_${ARM}
  CFG=$EXP/vib_${ARM}.yaml
  mk_vib_cfg "$CFG" "$CORE" "$ES" 100 "$OUTD"
  echo "[$ARM] core-only dyn 100ep start $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 timeout 7200 \
    $PY -m scout.train_vib --config "$CFG" > $EXP/train_${ARM}_dyn.log 2>&1
  RC=$?
  echo "[$ARM] dyn train rc=$RC $(date '+%F %T')"
  [ $RC -ne 0 ] && { tail -8 $EXP/train_${ARM}_dyn.log; exit 1; }
  VIB=$(newest_vib "$OUTD")
  ;;
*) echo "unknown arm $ARM"; exit 1 ;;
esac

[ -n "$VIB" ] || { echo "[$ARM] FATAL no vib ckpt"; exit 1; }
echo "$VIB" > $EXP/vib_path_${ARM}.txt

# ---------------- probe (fixed core batch) ---------------- #
echo "[$ARM] probe start $(date '+%F %T')"
env CUDA_VISIBLE_DEVICES=$GPU timeout 900 $PY scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root "$S" --task threading --core-name threading_core.hdf5 \
  --vib-config configs/vib_threading_exp1.yaml \
  --eval-config configs/eval_threading_entropy.yaml \
  --data-hdf5 "$CORE" --vib-ckpt "$VIB" --dp-ckpt "$ES" --tag thm3_$ARM
RC=$?
[ $RC -ne 0 ] && { echo "[$ARM] PROBE FAILED rc=$RC"; exit 1; }
echo "[$ARM] SUMMARY (dyn train summary.yaml):"
cat "$(dirname "$VIB")/summary.yaml" 2>/dev/null | head -20
echo "[$ARM] ALL DONE $(date '+%F %T')"
