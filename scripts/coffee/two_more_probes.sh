#!/bin/bash
set -u
cd /root/workspace/baojiachun/scout
P=/root/workspace/baojiachun/.venv_mg/bin/python
S3=data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s2333/coffee/train
S5=data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s23333/coffee/train
D3=data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s2333/coffee/rollout/ATY-exp1/all_accum.hdf5
D5=data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s23333/coffee/rollout/ATY-exp1/all_accum.hdf5
V3=$(ls -t $S3/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)
P3=$(ls -t $S3/DP/DP-ATY-exp5/checkpoints/*.ckpt | head -1)
V5=$(ls -t $S5/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)
P5=$(ls -t $S5/DP/DP-ATY-exp5/checkpoints/*.ckpt | head -1)
echo "resolved V3=$V3"
echo "resolved V5=$V5"
CUDA_VISIBLE_DEVICES=1 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s2333 \
  --data-hdf5 "$D3" --vib-ckpt "$V3" --dp-ckpt "$P3" --tag s2333_roll_aty5
CUDA_VISIBLE_DEVICES=2 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s23333 \
  --data-hdf5 "$D5" --vib-ckpt "$V5" --dp-ckpt "$P5" --tag s23333_roll_aty5
echo ALLDONE
