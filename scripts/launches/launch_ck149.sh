#!/bin/bash
# eval-only with sb_n5t10 epoch-149 ckpt (best-by-intraining-score candidate) vs 299.ckpt SR=0.70
cd /root/workspace/baojiachun/scout-entropy
mkdir -p data/entropy_e2e/sb_n5t10_ck149/eval
exec env CUDA_VISIBLE_DEVICES=4 SCOUT_RENDER_GPU=4 MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1   /root/workspace/baojiachun/.venv/bin/python -m scout.eval.run_rollout   --config configs/eval_att_a1.yaml --task can   --base-dp-ckpt /root/workspace/baojiachun/scout-entropy/data/entropy_e2e/sb_n5t10/dp/checkpoints/149.ckpt   --core-hdf5 /root/workspace/baojiachun/scout/data/2026_8_21/CAN-exp1-233-ee/can/rollout/can_core.hdf5   --guide off --eval-only --seed 42 --eval-seed 42   --n-init-states 100 --n-envs 12 --no-wandb   --output-dir data/entropy_e2e/sb_n5t10_ck149/eval   > data/entropy_e2e/sb_n5t10_ck149/stdout.log 2>&1
