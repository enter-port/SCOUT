#!/bin/bash
# Prep (mkdir + core copy + base-trio symlinks) for the s2333/s23333 square
# orbit chains -- byte-identical to the prep block of /tmp/sq_orb_chain_2333.sh
# and /tmp/sq_orb_chain_23333.sh (idempotent; chains re-run it at startup).
set -eu
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
ROOT=/root/workspace/baojiachun/scout-rand
prep() {
  s=$1; ts=$2
  T=$ROOT/data/2026_9_1_orbchain/ORBIT-s$s/square
  mkdir -p "$T/rollout" "$T/train/DP/DP-base/checkpoints" "$T/train/dyn/dyn-base"
  [ -f "$T/rollout/square_core.hdf5" ] || cp "$R26/core_rebuild/square_core_s$s.hdf5" "$T/rollout/square_core.hdf5"
  ln -sf "$R26/SQUARE-entropy-s$s/square/train/DP/DP-base/checkpoints/599.ckpt" \
        "$T/train/DP/DP-base/checkpoints/599.ckpt"
  ln -sfn "$R26/SQUARE-entropy-s$s/square/train/dyn/dyn-base/$ts" \
         "$T/train/dyn/dyn-base/$ts"
  echo "prep s$s done"
}
prep 2333 20260826-112147
prep 23333 20260829-025739
