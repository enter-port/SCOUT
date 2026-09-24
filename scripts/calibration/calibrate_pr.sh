#!/usr/bin/env bash
# Source from a round driver after DPCKPT/VIBCKPT/CORE/RDIR are resolved.
# Returns ATY_SCALE and ATT_CAP together; never follows PR with C calibration.
scout_calibrate_pr() {
  local output="$RDIR/calib_pr_r${NUM}.json" uuid pair
  if [ "${DRY_RUN:-0}" = 1 ]; then
    log "[calib-pr] DRY_RUN task=$TASK dp=$DPCKPT dyn=$VIBCKPT core=$CORE P=${CALIB_P:-6} R=${CALIB_R:-0.01} -> $output"
    return 0
  fi
  uuid=$(nvidia-smi -i "$GPU" --query-gpu=uuid --format=csv,noheader) || return 1
  "$PY" -m scout.calib.joint_pr \
    --task "$TASK" --eval-config "configs/eval_${TASK}_entropy.yaml" \
    --dp-ckpt "$DPCKPT" --vib-ckpt "$VIBCKPT" --core-hdf5 "$CORE" \
    --potential-cap "${CALIB_P:-6}" --target-r "${CALIB_R:-0.01}" \
    --initial-kappa "${CALIB_KAPPA0:-2.5}" --band "${CALIB_BAND:-0.1}" \
    --gpu "$GPU" --gpu-uuid "$uuid" --out "$output" \
    > "$RDIR/calib_pr.stdout" 2>&1 || {
      log "[calib-pr] FAILED; inspect $RDIR/calib_pr.stdout and $output"
      return 1
    }
  pair=$("$PY" -c 'import json,math,sys; d=json.load(open(sys.argv[1])); p=float(sys.argv[2]); r=float(sys.argv[3]); b=float(sys.argv[4]); assert d["R_converged"] and all(math.isfinite(d[k]) and d[k]>0 for k in ("eta","kappa","R_mean")); assert abs(d["eta"]*d["kappa"]/p-1)<=b+1e-12 and abs(d["R_mean"]/r-1)<=b+1e-12; print(d["eta"], d["kappa"])' \
    "$output" "${CALIB_P:-6}" "${CALIB_R:-0.01}" "${CALIB_BAND:-0.1}") || return 1
  read -r ATY_SCALE ATT_CAP <<< "$pair"
  log "[calib-pr] eta=$ATY_SCALE kappa=$ATT_CAP -> $output"
}
