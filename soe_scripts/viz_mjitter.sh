#!/usr/bin/env bash
# viz_mjitter.sh -- first-chunk visualization: mjitter b=0.5 vs entropy cost
# (user order 2026-08-28), campaign base assets, seed 42, 20 draws/arm, GPU6.
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=6
export SCOUT_RENDER_GPU=6
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout-rand
D=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
$PY -m scout.eval.visualize_first_chunk \
  --config configs/eval_can_entropy.yaml --task can \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --vib-ckpt "$D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --seed 42 --n 20 --n-envs 10 \
  --guide rand_mjitter --rand-kwargs "rand_b=0.5" \
  --out-dir data/rand/vis_first_chunk/base_mjitter_b05 \
  > data/rand/vis_first_chunk_mjitter.log 2>&1
echo "VIZ mjitter rc=$?"
$PY -m scout.eval.visualize_first_chunk \
  --config configs/eval_can_entropy.yaml --task can \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --vib-ckpt "$D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --seed 42 --n 20 --n-envs 10 \
  --guide atypical --atypical-cap 2.5 \
  --out-dir data/rand/vis_first_chunk/base_entropy \
  > data/rand/vis_first_chunk_entropy.log 2>&1
echo "VIZ entropy rc=$?"
