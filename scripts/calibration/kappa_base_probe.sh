#!/usr/bin/env bash
# Paired base-checkpoint kappa probe on eval scenes 42..141.
# Usage: bash scripts/calibration/kappa_base_probe.sh TASK GPU DP_CKPT VIB_CKPT CORE_HDF5 ETA_PREV
set -euo pipefail

TASK=${1:?task}
GPU=${2:?gpu}
DP_CKPT=${3:?DP checkpoint}
VIB_CKPT=${4:?VIB checkpoint}
CORE=${5:?core HDF5}
ETA_PREV=${6:?initial eta}

case "$TASK" in
  can|square|coffee|threading|tool_hang) ;;
  *) echo "unsupported task: $TASK" >&2; exit 2 ;;
esac
case "$GPU" in
  0|1|2|3|4|5|6) ;;
  *) echo "GPU $GPU is invalid or prohibited" >&2; exit 2 ;;
esac

ROOT=${SCOUT_ROOT:-/mnt/workspace/baojiachun/scout}
PY=${SCOUT_PYTHON:-/mnt/workspace/baojiachun/.venv_mg/bin/python}
OUT="$ROOT/experiments/2026_09_24_kappa_calibration/base/$TASK"
cd "$ROOT"
for path in "$DP_CKPT" "$VIB_CKPT" "$CORE" "$PY"; do
  test -f "$path" || { echo "missing: $path" >&2; exit 1; }
done

check_gpu() {
  local used
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n "$((GPU + 1))p" | tr -d ' ')
  test -n "$used" && test "$used" -lt 128 || {
    echo "GPU $GPU is occupied (${used:-unknown} MiB); stopping before the next job" >&2
    exit 1
  }
}

run_env() {
  env CUDA_VISIBLE_DEVICES="$GPU" SCOUT_RENDER_GPU="$GPU" MUJOCO_GL=egl \
    TMPDIR=/tmp PYTHONUNBUFFERED=1 PYTHON="$PY" "$@"
}

check_gpu
mkdir -p "$OUT"
if test ! -f "$OUT/eta.json"; then
  run_env "$PY" scripts/coffee/cfk_rcalib.py \
    --eval-config "configs/eval_${TASK}_entropy.yaml" \
    --dp-ckpt "$DP_CKPT" --vib-ckpt "$VIB_CKPT" --core-hdf5 "$CORE" \
    --eta-prev "$ETA_PREV" --kappa-prev 2.5 --out "$OUT/eta.json" \
    > "$OUT/eta.stdout" 2>&1
fi
ETA=$("$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["converged"], d; print(d["eta"])' "$OUT/eta.json")
echo "[$TASK] eta=$ETA (R target 0.01)"

if test ! -f "$OUT/failed.json"; then
  check_gpu
  mkdir -p "$OUT/eval"
  run_env "$PY" -m scout.eval.run_rollout \
    --config "configs/eval_${TASK}_entropy.yaml" --task "$TASK" --exp-num 0 \
    --base-dp-ckpt "$DP_CKPT" --core-hdf5 "$CORE" \
    --guide off --seed 42 --eval-seed 42 --n-init-states 100 \
    --explore-mode rescue --eval-only --save-failed-set "$OUT/failed.json" \
    --n-envs 25 --no-wandb --output-dir "$OUT/eval" \
    --output-json "$OUT/eval/summary.json" > "$OUT/eval/stdout" 2>&1
fi

for ARM in dp k1 k2.5 k5; do
  ADIR="$OUT/$ARM"
  test -f "$ADIR/summary.json" && continue
  test ! -d "$ADIR" || { echo "incomplete output: $ADIR" >&2; exit 1; }
  check_gpu
  mkdir -p "$ADIR"
  GUIDE=(--guide off)
  if test "$ARM" != dp; then
    KAP=${ARM#k}
    GUIDE=(--guide atypical --vib-ckpt "$VIB_CKPT" \
      --guidance-scale "$ETA" --atypical-cap "$KAP")
    run_env "$PY" scripts/threading/thm2_cap_probe.py \
      --eval-config "configs/eval_${TASK}_entropy.yaml" \
      --dp-ckpt "$DP_CKPT" --vib-ckpt "$VIB_CKPT" --core-hdf5 "$CORE" \
      --eta "$ETA" --kappa "$KAP" --out "$ADIR/kl_probe.json" \
      > "$ADIR/kl_probe.stdout" 2>&1
  fi
  echo "[$TASK] arm=$ARM GPU=$GPU start $(date -Is)"
  run_env bash scripts/infra/shard_rollout.sh 4 \
    "$ADIR/summary.json" "$ADIR/success.hdf5" "$ADIR/all.hdf5" "$CORE" -- \
    --config "configs/eval_${TASK}_entropy.yaml" --task "$TASK" --exp-num 0 \
    --base-dp-ckpt "$DP_CKPT" --core-hdf5 "$CORE" \
    "${GUIDE[@]}" --seed 42 --eval-seed 42 --n-init-states 100 \
    --explore-mode rescue --failed-set-json "$OUT/failed.json" \
    --explore-try-times 5 --n-envs 25 --no-wandb \
    --output-dir "$ADIR" --output-success "$ADIR/success.hdf5" \
    --output-all "$ADIR/all.hdf5" > "$ADIR/rollout.stdout" 2>&1
  echo "[$TASK] arm=$ARM done $(date -Is)"
done
