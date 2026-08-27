#!/usr/bin/env bash
# rand_screen.sh <tag> <guide> <gpu> [extra run_rollout args...]
#   20-scene parameter screen (user protocol 2026-08-27: env 42-62, i.e. the
#   first 20 seed-42 scenes) -- rescue x10, env20, base DP+dyn, --no-wandb.
#   SCALE env var sets exploration.guidance_scale of the scratch config
#   (default 3.0).  Extra args appended verbatim (e.g. --rand-kwargs
#   "rand_w_lo=0.5,rand_w_hi=2").  Output: scout-rand/data/rand/<tag>/.
#   Run on EITHER host: ssh -p 1022 (GPU 2/5 free) or -p 1024 (all 8).
set -u
TAG=$1; GUIDE=$2; GPU=$3; shift 3
W=/root/workspace/baojiachun/scout-rand
D=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
SCALE=${SCALE:-3.0}
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=$GPU
export SCOUT_RENDER_GPU=$GPU
PY=/root/workspace/baojiachun/.venv/bin/python
cd $W
CFG=$($PY - <<EOF
import re, pathlib
base = pathlib.Path("configs/eval_can_entropy.yaml").read_text()
t = re.sub(r"(guidance_scale: )[0-9.]+", r"\g<1>$SCALE", base, count=1)
p = pathlib.Path("configs/rand_scratch/$TAG.yaml")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(t)
print(p)
EOF
)
RDIR=$W/data/rand/$TAG
mkdir -p "$RDIR"
$PY -m scout.eval.run_rollout \
  --config "$CFG" --task can --exp-num 0 \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --vib-ckpt "$D/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt" \
  --core-hdf5 "$D/rollout/can_core.hdf5" \
  --guide "$GUIDE" --seed 42 --eval-seed 42 \
  --explore-mode rescue --explore-try-times 10 \
  --n-init-states 20 --n-envs 20 \
  --no-wandb \
  --output-dir "$RDIR" --output-success "$RDIR/success.hdf5" \
  --output-all "$RDIR/all.hdf5" \
  "$@" > "$RDIR/rollout.stdout" 2>&1
echo "RAND $TAG rc=$? $(date)"
