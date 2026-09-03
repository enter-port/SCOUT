#!/bin/bash
# th_calib_probe.sh -- tool_hang orbit dose-response probe (clone of
# sq_calib_probe.sh, 2026-09-01). Runs AFTER the s233 round0 trio exists and
# BEFORE round 1: mini rescue (10 scenes, try 2) with orbit guidance at a
# given guidance_scale; reports eval SR / rescued / mean_inject so the scale
# landing in the square orbit band (mean_inject ~1.05-1.20 at scale 3.0) can
# be picked, then patched into configs/eval_tool_hang_entropy.yaml.
#
# usage: th_calib_probe.sh <tag> <gpu> <scale>
set -u
TAG=${1:?}; GPU=${2:?}; SCALE=${3:?}
cd /root/workspace/baojiachun/scout-orbit || exit 1
PY=/root/workspace/baojiachun/.venv/bin/python
TH=/root/workspace/baojiachun/scout-orbit/data/2026_9_1_toolhang/TOOLHANG-s233/tool_hang
DP=$(ls -t "$TH"/train/DP/DP-base/checkpoints/*.ckpt 2>/dev/null | head -1)
VIB=$(ls -t "$TH"/train/dyn/dyn-base/*/scout_vib.ckpt 2>/dev/null | head -1)
CORE=$TH/rollout/tool_hang_core.hdf5
[ -n "$DP" ] && [ -n "$VIB" ] && [ -f "$CORE" ] || { echo "FATAL: round0 trio missing (dp=$DP vib=$VIB core=$CORE)"; exit 1; }
OUT=data/toolhang_calib/$TAG
mkdir -p "$OUT"
# run_rollout has NO --guidance-scale flag (review 2026-09-01 P0-1): the dose
# lives in the eval config's exploration.guidance_scale, so generate a
# per-scale config copy and pass that.
CFG_OUT=$OUT/eval_tool_hang_gs${SCALE}.yaml
$PY - configs/eval_tool_hang_entropy.yaml "$CFG_OUT" "$SCALE" <<'PYEOF'
import sys, yaml
src, dst, scale = sys.argv[1:4]
cfg = yaml.safe_load(open(src))
cfg["exploration"]["guidance_scale"] = float(scale)
yaml.safe_dump(cfg, open(dst, "w"), sort_keys=False)
print(f"[calib-cfg] exploration.guidance_scale={scale} -> {dst}")
PYEOF
env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1 \
  $PY -m scout.eval.run_rollout --config "$CFG_OUT" --task tool_hang --exp-num 1 \
  --base-dp-ckpt "$DP" --core-hdf5 "$CORE" --vib-ckpt "$VIB" \
  --guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 \
  --explore-mode rescue --explore-try-times 2 \
  --n-init-states 10 --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
  --output-dir "$OUT" > "$OUT/stdout.log" 2>&1
RC=$?
echo "== th_calib_probe tag=$TAG gpu=$GPU scale=$SCALE rc=$RC =="
grep -aE "eval: success_rate|rescue: success_rate|pass@|rescued" "$OUT/stdout.log" | tail -4
grep -ao "mean_inject=[0-9.]*" "$OUT/stdout.log" | tail -3
exit $RC
