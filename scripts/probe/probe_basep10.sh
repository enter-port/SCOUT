#!/usr/bin/env bash
# probe_basep10.sh -- user order 2026-08-27: rerun s233 can round-1 pass@10
# (base DP 599.ckpt + base dyn, atypical cost s3.0/k2.5/gst100, rescue x10,
# eval seed42 x100 scenes) at n_envs=50 on GPU2. Purpose: run-to-run variance
# of the current cost's pass@10 vs the recorded r1 value 0.76 (r1 ran env12).
# Read-only wrt the chain; writes only to rollout/PROBE-base-pass10/.
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=2
export SCOUT_RENDER_GPU=2
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout-entropy
D=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
RDIR=$D/rollout/PROBE-base-pass10
mkdir -p "$RDIR"
$PY -m scout.eval.run_rollout \
  --config configs/eval_can_entropy.yaml --task can --exp-num 99 \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --vib-ckpt "$D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide atypical --seed 42 --eval-seed 42 \
  --explore-mode rescue --explore-try-times 10 \
  --n-envs 50 \
  --atypical-cap 2.5 \
  --no-wandb \
  --output-dir "$RDIR" \
  --output-success "$RDIR/success.hdf5" \
  --output-all "$RDIR/all.hdf5" \
  > "$RDIR/rollout.stdout" 2>&1
RC=$?
echo "PROBE_RC=$RC  $(date)"
exit $RC
