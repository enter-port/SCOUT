#!/bin/bash
cd /root/workspace/baojiachun/scout-orbit || exit 1
mkdir -p data/toolhang_calib
for S in 1.0 3.0 5.0; do
  TAG="th_s${S%.*}"
  bash soe_scripts/th_calib_probe.sh "$TAG" 1 "$S"
done > data/toolhang_calib/SWEEP_RESULT.txt 2>&1
