#!/usr/bin/env bash
# rand_screen_sq.sh <tag> <guide> <gpu> [extra run_rollout args...]
#   SQUARE replication of the 20-scene rand screen (LESSONS.md 提案 2, external
#   validity of the can "randomization has no upside" verdict).  Same protocol
#   as rand_screen.sh: first 20 seed-42 scenes, rescue x10, env20, base DP+dyn
#   from the square entropy campaign chain, --no-wandb.  SCALE env var sets
#   exploration.guidance_scale of the scratch config (default 3.0, the square
#   formal value per configs/eval_square_entropy.yaml).  Extra args appended
#   verbatim.  Output: scout-rand/data/rand/<tag>/.
#   Assets: scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square
#   (base DP 599.ckpt, dyn-base 20260826-112119, square_core.hdf5).
set -u
TAG=$1; GUIDE=$2; GPU=$3; shift 3
W=/root/workspace/baojiachun/scout-rand
D=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square
SCALE=${SCALE:-3.0}
export MUJOCO_GL=egl
export TMPDIR=/tmp
export CUDA_VISIBLE_DEVICES=$GPU
export SCOUT_RENDER_GPU=$GPU
PY=/root/workspace/baojiachun/.venv/bin/python
cd $W
CFG=$($PY - <<EOF
import re, pathlib
base = pathlib.Path("configs/eval_square_entropy.yaml").read_text()
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
  --config "$CFG" --task square --exp-num 0 \
  --base-dp-ckpt "$D/train/DP/DP-base/checkpoints/599.ckpt" \
  --vib-ckpt "$D/train/dyn/dyn-base/20260826-112119/scout_vib.ckpt" \
  --core-hdf5 "$D/rollout/square_core.hdf5" \
  --guide "$GUIDE" --seed 42 --eval-seed 42 \
  --atypical-cap 2.5 \
  --explore-mode rescue --explore-try-times 10 \
  --n-init-states 20 --n-envs 20 \
  --no-wandb \
  --output-dir "$RDIR" --output-success "$RDIR/success.hdf5" \
  --output-all "$RDIR/all.hdf5" \
  "$@" > "$RDIR/rollout.stdout" 2>&1
echo "RAND-SQ $TAG rc=$? $(date)"
