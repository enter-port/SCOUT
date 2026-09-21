#!/bin/bash
# tp15_prep_base.sh -- one-time prep for the TRANSPORT-9-15-p1 (pass@1)
# campaign (2026-09-15). Copies the round0 three-piece (core hdf5 + DP-base +
# dyn-base) of each seed from the COMPLETED 2026_9_13_transport tree into the
# NEW campaign root data/2026_9_15_transport_p1/TRANSPORT-s<SEED>/transport,
# and writes the ROUND0 TOTAL gate line that tp15_start_arms.sh greps.
# Old tree is read-only here (user order: nothing deleted/modified).
#
# usage: bash scripts/transport/tp15_prep_base.sh   (run once; CPFS shared,
#        1022 works for all three seeds)
set -euo pipefail
ROOT=/root/workspace/baojiachun/scout
SRC=$ROOT/data/2026_9_13_transport
DST=$ROOT/data/2026_9_15_transport_p1

# CPFS 卷满防护: df 不可信(2026-09-10 教训),先写 512MB 探针
if ! dd if=/dev/zero of=$ROOT/data/.tp15_probe bs=1M count=512 oflag=direct 2>/dev/null; then
  dd if=/dev/zero of=$ROOT/data/.tp15_probe bs=1M count=512
fi
rm -f $ROOT/data/.tp15_probe
echo "[prep] disk write probe OK"

for s in 233 2333 23333; do
  O=$SRC/TRANSPORT-s$s/transport
  N=$DST/TRANSPORT-s$s/transport
  [ -f "$O/rollout/transport_core.hdf5" ] || { echo "[prep] FATAL: core hdf5 missing for s$s"; exit 1; }
  ls "$O"/train/dyn/dyn-base/*/scout_vib.ckpt >/dev/null 2>&1 || { echo "[prep] FATAL: dyn-base scout_vib.ckpt missing for s$s"; exit 1; }
  ls "$O"/train/DP/DP-base/checkpoints/599.ckpt >/dev/null 2>&1 || { echo "[prep] FATAL: DP-base 599.ckpt missing for s$s"; exit 1; }
  mkdir -p "$N/rollout" "$N/train/DP" "$N/train/dyn"
  cp -L  "$O/rollout/transport_core.hdf5" "$N/rollout/transport_core.hdf5"
  cp -rL "$O/train/DP/DP-base" "$N/train/DP/DP-base"
  cp -rL "$O/train/dyn/dyn-base" "$N/train/dyn/dyn-base"
  if ! grep -q "ROUND0 transport seed=$s TOTAL" "$N/round.log" 2>/dev/null; then
    echo "[$(date '+%F %T')] === ROUND0 transport seed=$s TOTAL: 0m0s === (base copied from data/2026_9_13_transport/TRANSPORT-s$s for pass@1 campaign tp15; no retrain)" >> "$N/round.log"
  fi
  sb=$(stat -c%s "$O/rollout/transport_core.hdf5"); db=$(stat -c%s "$N/rollout/transport_core.hdf5")
  echo "[prep] s$s core $sb -> $db bytes; DP ckpts: $(ls "$N/train/DP/DP-base/checkpoints" | sort -n | tr '\n' ' '); dyn dirs: $(ls "$N/train/dyn/dyn-base" | tr '\n' ' ')"
  [ "$sb" = "$db" ] || { echo "[prep] FATAL: core byte mismatch s$s"; exit 1; }
done
echo "[prep] all three seeds prepped under $DST"
