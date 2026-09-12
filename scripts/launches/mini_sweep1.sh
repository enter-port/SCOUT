#!/usr/bin/env bash
# mini_sweep1.sh -- 方案A 10-env dose screen (user order 2026-08-27):
# 10 fixed scenes (seed42 first 10), rescue x10, env10, kappa 2.5, base
# assets.  Serial on GPU5 next to the running full probes (node load ~14,
# env10 is light).  Tags: eta0p35 (anchor, same cfg as full A3), eta0p7,
# eta1p4.  Writes only to rollout/MINI-shell-<tag>/.
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=5
export SCOUT_RENDER_GPU=5
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout-entropy
D=data/2026_8_21_entropy/CAN-entropy-s233/can
for CFG in eval_can_shell_eta035:eta0p35 eval_can_shell_e07:eta0p7 eval_can_shell_e14:eta1p4; do
  C=${CFG%%:*}; T=${CFG##*:}
  RDIR=$D/rollout/MINI-shell-$T
  mkdir -p "$RDIR"
  $PY -m scout.eval.run_rollout \
    --config configs/$C.yaml --task can --exp-num 0 \
    --base-dp-ckpt $D/train/DP/DP-base/checkpoints/599.ckpt \
    --vib-ckpt $D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt \
    --core-hdf5 $D/rollout/can_core.hdf5 \
    --guide shell --shell-kappa 2.5 --seed 42 --eval-seed 42 \
    --explore-mode rescue --explore-try-times 10 \
    --n-init-states 10 --n-envs 10 \
    --no-wandb \
    --output-dir $RDIR --output-success $RDIR/success.hdf5 --output-all $RDIR/all.hdf5 \
    > $RDIR/rollout.stdout 2>&1
  echo "MINI $T rc=$? $(date)"
done
echo "MINI_SWEEP1_DONE $(date)"
