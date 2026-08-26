#!/bin/bash
# entropy-dev Stage B (SOE method): core + best-rescue success.hdf5 -> 300ep DP -> eval-only SR.
# usage: stage_b.sh <best_run_dir> <tag> <gpu>   (best_run_dir relative to worktree root)
set -u
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
BEST=${1:?best run dir, e.g. data/entropy_e2e/full_a2k}
TAG=${2:?tag, e.g. sb_a2k}
GPU=${3:?gpu id}
cd /root/workspace/baojiachun/scout-entropy
PY=/root/workspace/baojiachun/.venv/bin/python
CORE=/root/workspace/baojiachun/scout/data/2026_8_21/CAN-exp1-233-ee/can/rollout/can_core.hdf5
OUT=data/entropy_e2e/$TAG
mkdir -p "$OUT"
[ -f "$OUT/DONE" ] && { echo "[stage_b] $TAG already done"; exit 0; }
[ -f "$BEST/success.hdf5" ] || { echo "[stage_b] missing $BEST/success.hdf5"; exit 1; }

echo "[stage_b] [1/3] merge core + $BEST/success.hdf5 -> $OUT/train.hdf5"
$PY - "$CORE" "$BEST/success.hdf5" "$OUT/train.hdf5" <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from scout.eval.hdf5_writer import merge_accumulated_hdf5
core, succ, out = sys.argv[1:4]
print("[stage_b] merge:", merge_accumulated_hdf5(core, [succ], out))
PYEOF

echo "[stage_b] [2/3] train 300ep DP on GPU$GPU -> $OUT/dp"
env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY train.py --config-path configs --config-name base_dp_can_image \
  task.dataset_path="$PWD/$OUT/train.hdf5" \
  task.train_filter_key=scout_aug \
  training.num_epochs=300 training.checkpoint_every=150 training.seed=233 \
  dataloader.num_workers=8 dataloader.persistent_workers=true \
  +logging.metric_prefix=DP/ logging.project=CAN-entropy-stageB logging.name=$TAG \
  hydra.run.dir="$PWD/$OUT/dp" \
  > "$OUT/train.log" 2>&1
RC=$?
if [ $RC -ne 0 ] || [ ! -f "$OUT/dp/checkpoints/299.ckpt" ]; then
  echo "[stage_b] workers=8 failed (rc=$RC) -- retry num_workers=0"
  env CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 TMPDIR=/tmp PYTHONUNBUFFERED=1 \
    $PY train.py --config-path configs --config-name base_dp_can_image \
    task.dataset_path="$PWD/$OUT/train.hdf5" \
    task.train_filter_key=scout_aug \
    training.num_epochs=300 training.checkpoint_every=150 training.seed=233 \
    dataloader.num_workers=0 \
    +logging.metric_prefix=DP/ logging.project=CAN-entropy-stageB logging.name=$TAG \
    hydra.run.dir="$PWD/$OUT/dp" \
    >> "$OUT/train.log" 2>&1
fi
CK="$OUT/dp/checkpoints/299.ckpt"
[ -f "$CK" ] || { echo "[stage_b] TRAIN FAILED - no $CK (see $OUT/train.log)"; exit 1; }

echo "[stage_b] [3/3] eval-only SR with new DP ($CK)"
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout --config configs/eval_att_a1.yaml --task can \
  --base-dp-ckpt "$PWD/$CK" --core-hdf5 "$CORE" \
  --guide off --eval-only --seed 42 --eval-seed 42 \
  --n-init-states 100 --n-envs 12 --no-wandb \
  --output-dir "$OUT/eval" > "$OUT/eval.log" 2>&1
grep -aE "eval: success" "$OUT/eval.log" | tail -1
touch "$OUT/DONE"
echo "[stage_b] $TAG COMPLETE"
