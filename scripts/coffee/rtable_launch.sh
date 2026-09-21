#!/bin/bash
# rtable_launch.sh -- remaining probes+norms for the eta*|grad|/|a| table.
set -u
cd /root/workspace/baojiachun/scout
P=/root/workspace/baojiachun/.venv_mg/bin/python
L=data/2026_9_20_coffee_mg_p1/TELEMETRY_LOGS
CN=data/2026_8_21_entropy/CAN-entropy-s233/can
SQ=data/2026_8_26_entropy/SQUARE-entropy-s233/square
TP=data/2026_9_15_transport_p1/TRANSPORT-s233/transport
CA=data/robomimic/can/ph/image_v141_abs_core20.hdf5
SA=data/robomimic/square/ph/image_v141_abs_core20.hdf5
TA=$TP/rollout/ATY-exp1/all_accum.hdf5
C6=$(ls -t $CN/train/dyn/dyn-SCOUT-exp6/*/scout_vib.ckpt | head -1)
CD6=$(ls -t $CN/train/DP/DP-SCOUT-exp6/checkpoints/*.ckpt | head -1)
SB=$(ls -t $SQ/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
S6=$(ls -t $SQ/train/dyn/dyn-SCOUT-exp6/*/scout_vib.ckpt | head -1)
SDB=$(ls -t $SQ/train/DP/DP-base/checkpoints/*.ckpt | head -1)
SD6=$(ls -t $SQ/train/DP/DP-SCOUT-exp6/checkpoints/*.ckpt | head -1)
TB=$(ls -t $TP/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
T5=$(ls -t $TP/train/dyn/dyn-ATY-exp5/*/scout_vib.ckpt | head -1)
TDB=$(ls -t $TP/train/DP/DP-base/checkpoints/*.ckpt | head -1)
TD5=$(ls -t $TP/train/DP/DP-ATY-exp5/checkpoints/*.ckpt | head -1)
echo "resolved: C6=$C6"
echo "resolved: SB=$SB S6=$S6"
echo "resolved: TB=$TB T5=$T5"
nohup env CUDA_VISIBLE_DEVICES=1 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_8_21_entropy/CAN-entropy-s233 --task can --core-name x.hdf5 \
  --vib-config configs/vib_can_exp1.yaml --eval-config configs/eval_can_entropy.yaml \
  --data-hdf5 "$CA" --vib-ckpt "$C6" --dp-ckpt "$CD6" --tag can_e6 > $L/can_e6.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=2 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_8_26_entropy/SQUARE-entropy-s233 --task square --core-name x.hdf5 \
  --vib-config configs/vib_square_exp1.yaml --eval-config configs/eval_square_entropy.yaml \
  --data-hdf5 "$SA" --vib-ckpt "$SB" --dp-ckpt "$SDB" --tag sq_base > $L/sq_base.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=3 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_8_26_entropy/SQUARE-entropy-s233 --task square --core-name x.hdf5 \
  --vib-config configs/vib_square_exp1.yaml --eval-config configs/eval_square_entropy.yaml \
  --data-hdf5 "$SA" --vib-ckpt "$S6" --dp-ckpt "$SD6" --tag sq_e6 > $L/sq_e6.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=4 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_9_15_transport_p1/TRANSPORT-s233 --task transport --core-name transport_core.hdf5 \
  --vib-config configs/vib_transport_exp1.yaml --eval-config configs/eval_transport_entropy.yaml \
  --data-hdf5 "$TA" --vib-ckpt "$TB" --dp-ckpt "$TDB" --tag tp_base > $L/tp_base.log 2>&1 &
nohup env CUDA_VISIBLE_DEVICES=5 timeout 280 $P scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_9_15_transport_p1/TRANSPORT-s233 --task transport --core-name transport_core.hdf5 \
  --vib-config configs/vib_transport_exp1.yaml --eval-config configs/eval_transport_entropy.yaml \
  --data-hdf5 "$TA" --vib-ckpt "$T5" --dp-ckpt "$TD5" --tag tp_e5 > $L/tp_e5.log 2>&1 &
nohup bash -c "
$P scripts/coffee/cfk_anorm.py --vib-config configs/vib_can_exp1.yaml --data-hdf5 $CA --task can
$P scripts/coffee/cfk_anorm.py --vib-config configs/vib_square_exp1.yaml --data-hdf5 $SA --task square
$P scripts/coffee/cfk_anorm.py --vib-config configs/vib_transport_exp1.yaml --data-hdf5 $TA --task transport
$P scripts/coffee/cfk_anorm.py --vib-config configs/vib_threading_exp1.yaml --data-hdf5 data/2026_9_17_threading_p3/THREADING-P3-s233/threading/rollout/ATY-exp1/all.hdf5 --task threading
$P scripts/coffee/cfk_anorm.py --vib-config configs/vib_tool_hang_exp1.yaml --data-hdf5 data/2026_9_5_toolhang/TOOLHANG-s233/tool_hang/rollout/ATY-exp1/all_accum.hdf5 --task toolhang
$P scripts/coffee/cfk_anorm.py --vib-config configs/vib_coffee_exp1.yaml --data-hdf5 data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s233/coffee/rollout/ATY-exp1/all_accum.hdf5 --task coffee
" > $L/anorms.log 2>&1 &
sleep 4
pgrep -fc 'cfk_grad_probe_arm|cfk_anorm'
