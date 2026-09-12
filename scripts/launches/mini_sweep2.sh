#!/usr/bin/env bash
# mini_sweep2.sh -- 方案A kappa sweep at fixed eta 0.35 (user order 2026-08-27):
# kappa = target-shell radius in nats (sqrt(2*kappa)*sigma0 displacement).
# 10 fixed scenes (seed42 first 10), rescue x10, env10, GPU5.  Tags k1p25/k5.
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=5
export SCOUT_RENDER_GPU=5
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout-entropy
D=data/2026_8_21_entropy/CAN-entropy-s233/can
for KP in 1.25:mini_k1p25 5.0:mini_k5; do
  K=${KP%%:*}; T=${KP##*:}
  RDIR=$D/rollout/MINI-shell-eta0p35-$T
  mkdir -p "$RDIR"
  $PY -m scout.eval.run_rollout \
    --config configs/eval_can_shell_eta035.yaml --task can --exp-num 0 \
    --base-dp-ckpt $D/train/DP/DP-base/checkpoints/599.ckpt \
    --vib-ckpt $D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt \
    --core-hdf5 $D/rollout/can_core.hdf5 \
    --guide shell --shell-kappa $K --seed 42 --eval-seed 42 \
    --explore-mode rescue --explore-try-times 10 \
    --n-init-states 10 --n-envs 10 \
    --no-wandb \
    --output-dir $RDIR --output-success $RDIR/success.hdf5 --output-all $RDIR/all.hdf5 \
    > $RDIR/rollout.stdout 2>&1
  echo "MINI $T rc=$? $(date)"
done
echo "MINI_SWEEP2_DONE $(date)"
