#!/usr/bin/env bash
# probe_shellA4.sh -- 方案A full pass@10 at eta 0.7 (user tuning mandate
# 2026-08-27): dose trend 0.115->10, 0.35->13 rescued (monotone up); 0.7 is
# the next untested point (minis: 0.35~0.7 plateau, 1.4 degrades).  kappa 2.5,
# everything else identical to probe_shellA2/3.  GPU2.  Writes only to
# rollout/PROBE-shellA-pass10-eta0p7/.
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=2
export SCOUT_RENDER_GPU=2
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout-entropy
D=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
RDIR=$D/rollout/PROBE-shellA-pass10-eta0p7
mkdir -p "$RDIR"
$PY -m scout.eval.run_rollout \
  --config configs/eval_can_shell_e07.yaml --task can --exp-num 95 \
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
