#!/usr/bin/env bash
# probe_shellA2.sh -- 方案A rerun (user order 2026-08-27): identical to
# probe_shellA.sh except guidance_scale 3.0 -> 0.115 (configs/eval_can_shell.yaml)
# to match 方案三's calibrated mean_inject ~1.0 (first run measured ~26x too hot).
# Base DP 599 + base dyn, kappa 2.5, gst 100, rescue x10, eval seed42 x100,
# env 50, GPU5.  Writes only to rollout/PROBE-shellA-pass10-eta0p115/.
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=5
export SCOUT_RENDER_GPU=5
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout-entropy
D=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
RDIR=$D/rollout/PROBE-shellA-pass10-eta0p115
mkdir -p "$RDIR"
$PY -m scout.eval.run_rollout \
  --config configs/eval_can_shell.yaml --task can --exp-num 97 \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --vib-ckpt "$D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide shell --shell-kappa 2.5 --seed 42 --eval-seed 42 \
  --explore-mode rescue --explore-try-times 10 \
  --n-envs 50 \
  --no-wandb \
  --output-dir "$RDIR" \
  --output-success "$RDIR/success.hdf5" \
  --output-all "$RDIR/all.hdf5" \
  > "$RDIR/rollout.stdout" 2>&1
RC=$?
echo "PROBE_RC=$RC  $(date)"
exit $RC
