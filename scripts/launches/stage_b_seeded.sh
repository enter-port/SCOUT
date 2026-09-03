#!/bin/bash
# stage-B retrain on a PREBUILT train.hdf5 (no merge, no BEST dir) -- fixes the
# A3 assembly bug where seed variants re-merged from a single source and
# clobbered the prebuilt accumulated file.
# usage: stage_b_seeded.sh <train_hdf5_abs_or_rel> <tag> <gpu> <seed>
set -u
TRAIN=${1:?prebuilt train.hdf5 path}
TAG=${2:?tag}
GPU=${3:?gpu id}
SEED=${4:?training seed}
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
cd /root/workspace/baojiachun/scout-entropy
PY=/root/workspace/baojiachun/.venv/bin/python
OUT=data/entropy_e2e/$TAG
mkdir -p "$OUT"
[ -f "$TRAIN" ] || { echo "[stage_b] missing $TRAIN"; exit 1; }
[ -f "$OUT/DONE" ] && { echo "[stage_b] $TAG already done"; exit 0; }
$PY - "$TRAIN" "$OUT/train.hdf5" <<'PYEOF'
import sys, shutil
src, dst = sys.argv[1:3]
shutil.copyfile(src, dst)
import h5py
with h5py.File(dst, "r") as f:
    print("[stage_b] prebuilt", dst, len([k for k in f["data"] if k.startswith("demo")]), "demos")
PYEOF
echo "[stage_b] train 300ep GPU$GPU seed=$SEED -> $OUT/dp"
env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY train.py --config-path configs --config-name base_dp_can_image \
  task.dataset_path="$PWD/$OUT/train.hdf5" \
  task.train_filter_key=scout_aug \
  training.num_epochs=300 training.checkpoint_every=150 training.seed=$SEED \
  dataloader.num_workers=8 dataloader.persistent_workers=true \
  +logging.metric_prefix=DP/ logging.project=CAN-entropy-stageB logging.name=$TAG \
  hydra.run.dir="$PWD/$OUT/dp" \
  > "$OUT/train.log" 2>&1
CK="$OUT/dp/checkpoints/299.ckpt"
[ -f "$CK" ] || { echo "[stage_b] TRAIN FAILED - no $CK"; exit 1; }
echo "[stage_b] eval-only SR with $CK"
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout --config configs/eval_att_a1.yaml --task can \
  --base-dp-ckpt "$PWD/$CK" --core-hdf5 /root/workspace/baojiachun/scout/data/2026_8_21/CAN-exp1-233-ee/can/rollout/can_core.hdf5 \
  --guide off --eval-only --seed 42 --eval-seed 42 \
  --n-init-states 100 --n-envs 12 --no-wandb \
  --output-dir "$OUT/eval" > "$OUT/eval.log" 2>&1
grep -aE "eval: success" "$OUT/eval.log" | tail -1
touch "$OUT/DONE"
echo "[stage_b] $TAG COMPLETE"
