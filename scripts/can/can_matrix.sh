#!/bin/bash
# CAN exploit matrix -- square champion config frozen (visual slice,
# eta=250, gst=100, kNN=1, binary OOD gate, thr = p75 calibrated per
# round bank) applied to every SCOUT-chain ckpt exp1..6 of ONE seed.
# Bank per round = that ckpt's training data (core + success_1..N,
# rebuilt by can_matrix_prep.py). Protocol identical to the square
# campaign: single try per scene, scenes seed 42..141 (n=100),
# skip-eval explore pipeline, n_envs 25, no wandb.
# After round-1 calibration the driver pauses until $OUTROOT/GO exists
# (one-time inspection gate), then runs everything idempotently.
# Usage: can_matrix.sh SEED GPU   (SEED in 233/2333/23333)
set -euo pipefail
SEED="${1:?seed}" GPU="${2:?gpu}"
REPO=/root/workspace/baojiachun/scout-exploit
PY=/root/workspace/baojiachun/.venv/bin/python
BASE=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s$SEED/can
OUTROOT=$REPO/data/exploit_can_matrix/s$SEED
CORE=$BASE/rollout/can_core.hdf5

case "$SEED" in
  233)   T=(x 20260825-074749 20260825-122009 20260825-185709 20260825-230040 20260826-030203 20260826-074559) ;;
  2333)  T=(x 20260825-084315 20260825-182721 20260825-221728 20260826-024512 20260826-082627 20260826-150122) ;;
  23333) T=(x 20260825-071516 20260825-102750 20260825-133347 20260825-181256 20260825-215517 20260826-015139) ;;
  *) echo "unknown seed $SEED"; exit 1 ;;
esac

export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=$GPU
export SCOUT_RENDER_GPU=$GPU
mkdir -p "$OUTROOT"

echo "[$(date '+%F %T')] [prep] s$SEED rebuilding accum banks"
"$PY" soe_scripts/can_matrix_prep.py "$SEED"

cd "$REPO"
for N in 1 2 3 4 5 6; do
  OUT=$OUTROOT/r$N
  if [ -f "$OUT/log/can_SCOUT_rollout_exp1.json" ]; then
    echo "[$(date '+%F %T')] s$SEED r$N done, skip"; continue
  fi
  DP=$BASE/train/DP/DP-SCOUT-exp$N/checkpoints/299.ckpt
  VIB=$BASE/train/dyn/dyn-SCOUT-exp$N/${T[$N]}/scout_vib.ckpt
  BANK=$OUTROOT/bank_accum_r$N.hdf5
  for f in "$DP" "$VIB" "$BANK" "$CORE"; do
    [ -f "$f" ] || { echo "FATAL missing $f"; exit 1; }
  done
  if [ ! -f "$OUTROOT/thr_r$N.txt" ]; then
    echo "[$(date '+%F %T')] [calib] s$SEED r$N"
    "$PY" soe_scripts/calib_can.py "$SEED" "$N" "$VIB"
  fi
  THR=$(cat "$OUTROOT/thr_r$N.txt")
  [[ "$THR" =~ ^[0-9]+\.[0-9]{2}$ ]] || { echo "FATAL bad thr '$THR'"; exit 1; }
  if [ "$N" = "1" ] && [ ! -f "$OUTROOT/GO" ]; then
    echo "[$(date '+%F %T')] [gate] waiting for $OUTROOT/GO (inspection gate)"
    while [ ! -f "$OUTROOT/GO" ]; do sleep 60; done
    echo "[$(date '+%F %T')] [gate] GO found, proceeding"
  fi
  if [ ! -f "$OUTROOT/eta_r$N.txt" ]; then
    echo "FATAL missing $OUTROOT/eta_r$N.txt (run can_dose_probe.py first)"; exit 1
  fi
  "$PY" - "$SEED" "$N" "$THR" <<'PYEOF'
import sys
from omegaconf import OmegaConf
seed, n, thr = sys.argv[1], sys.argv[2], sys.argv[3]
eta = float(open(f"data/exploit_can_matrix/s{seed}/eta_r{n}.txt").read().strip())
cfg = OmegaConf.load("configs/eval_can_entropy.yaml")
cfg.exploration.guidance_scale = eta
cfg.exploration.guidance_start_timestep = 100
OmegaConf.save(cfg, f"configs/_tmp_can_s{seed}_r{n}.yaml")
print(f"[cfg] s{seed}_r{n}: eta={eta} gst=100 thr={thr}")
PYEOF
  echo "[$(date '+%F %T')] [run] s$SEED r$N thr=$THR eta=$(cat "$OUTROOT/eta_r$N.txt")"
  "$PY" -m scout.eval.run_rollout \
    --config "configs/_tmp_can_s${SEED}_r${N}.yaml" \
    --task can \
    --base-dp-ckpt "$DP" \
    --vib-ckpt "$VIB" \
    --core-hdf5 "$CORE" \
    --guide exploit --bank-hdf5 "$BANK" --exploit-latent visual \
    --exploit-ood-threshold "$THR" \
    --skip-eval --explore-seed 42 --n-explore 100 \
    --n-envs 25 \
    --seed 42 --no-wandb \
    --output-dir "$OUT" \
    --output-success "$OUT/success.hdf5" \
    --output-all "$OUT/all.hdf5" \
    --exp-num 1
  echo "[$(date '+%F %T')] s$SEED r$N DONE thr=$THR"
done
echo "[$(date '+%F %T')] [matrix] s$SEED ALL DONE"
