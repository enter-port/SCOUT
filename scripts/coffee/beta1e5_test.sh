#!/bin/bash
# beta1e5_test.sh -- threading s233: retrain the round-1 dyn with beta=1e-5
# (chain used 3e-5) on the SAME data (core + ATY-exp1/all.hdf5 rebuild) and
# SAME E_s (DP-ATY-exp1), then probe |dKL/da| on the same fixed anchor.
# Question: does the action-dependence still collapse?  (base 2.05 / 3e-5 0.41)
# All outputs under data/2026_9_17_threading_p3/BETA1E5_TEST_s233/ -- touches
# no chain file.
set -u
cd /root/workspace/baojiachun/scout
P=/root/workspace/baojiachun/.venv_mg/bin/python
GPU=1
S=data/2026_9_17_threading_p3/THREADING-P3-s233/threading
EXP=data/2026_9_17_threading_p3/BETA1E5_TEST_s233
mkdir -p "$EXP/train"
CORE=$S/rollout/threading_core.hdf5
ALL1=$S/rollout/ATY-exp1/all.hdf5
DP=$S/train/DP/DP-ATY-exp1/checkpoints/299.ckpt
for f in "$CORE" "$ALL1" "$DP"; do
  [ -f "$f" ] || { echo "[b1e5] FATAL missing $f"; exit 1; }
done

# 1) rebuild round-1 all_accum exactly as the chain did (core + r1 all.hdf5)
ACCUM=$EXP/all_accum_r1.hdf5
$P - "$CORE" "$ALL1" "$ACCUM" <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from scout.eval.hdf5_writer import merge_accumulated_hdf5
info = merge_accumulated_hdf5(sys.argv[1], [sys.argv[2]], sys.argv[3])
print("[b1e5] accum:", info)
PYEOF

# 2) write the training config: chain overlays + beta 1e-5 + wandb off
CFG=$EXP/vib_threading_b1e5.yaml
$P - "$CFG" "$ACCUM" "$DP" "$EXP/train" <<'PYEOF'
import sys, yaml
cfg = yaml.safe_load(open("configs/vib_threading_exp1.yaml"))
cfg["dataset"]["zarr_path"] = sys.argv[2]
cfg["dataset"]["feature_cache"] = True
cfg["model"]["E_s"]["base_dp_ckpt"] = sys.argv[3]
cfg["seed"] = 233
cfg["cudnn_deterministic"] = True
cfg["num_epochs"] = 100
cfg["beta"] = 1.0e-5
cfg["save_dir"] = sys.argv[4]
cfg["use_wandb"] = False
cfg["wandb"] = {"name": "dyn-BETA1E5-test", "project": "offline", "minimal": True}
yaml.safe_dump(cfg, open(sys.argv[1], "w"), sort_keys=False)
print("[b1e5] cfg written: beta=1e-5 seed=233 ep=100 ds=%s es=%s" % (sys.argv[2], sys.argv[3]))
PYEOF

# 3) train (chain flags: CUBLAS determinism)
echo "[b1e5] training starts $(date '+%F %T')"
CUDA_VISIBLE_DEVICES=$GPU CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  timeout 3000 $P -m scout.train_vib --config "$CFG" \
  > "$EXP/train/train.log" 2>&1
RC=$?
echo "[b1e5] train rc=$RC $(date '+%F %T')"
if [ $RC -ne 0 ]; then tail -20 "$EXP/train/train.log"; exit 1; fi
VIB=$(ls -t "$EXP"/train/*/scout_vib.ckpt | head -1)
echo "[b1e5] vib ckpt: $VIB"

# 4) probe on the SAME anchor as th_base/th_aty1/th_aty5
CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU timeout 280 $P \
  scripts/coffee/cfk_grad_probe_arm.py \
  --seed-data-root data/2026_9_17_threading_p3/THREADING-P3-s233 \
  --task threading --core-name threading_core.hdf5 \
  --vib-config configs/vib_threading_exp1.yaml \
  --eval-config configs/eval_threading_entropy.yaml \
  --data-hdf5 "$ALL1" --vib-ckpt "$VIB" --dp-ckpt "$DP" --tag th_b1e5
echo "[b1e5] DONE $(date '+%F %T')"
