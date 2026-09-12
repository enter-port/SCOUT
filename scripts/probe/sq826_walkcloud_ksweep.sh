#!/bin/bash
# sq826_walkcloud_ksweep.sh -- square walkcloud PARAM SWEEP (user 2026-09-10
# order: NO eta_dimless; kappa {2.5,5,7.5} @ gst100 first + gst50 @ kappa2.5;
# each run 45-min backstop; parallel; never interrupt anything running on
# 1022/1024).
#
# 4 configs in parallel, one free GPU each (1/3/5/6 -- 0/2/4 are th95_s23333,
# 7 banned ECC). Each config = sq826_walkcloud_probe.sh with ARMS=wc under a
# PER-CONFIG ROOT (avoids the shared r${ROUND}/ collision), two sequential
# windows [0:32]+[32:63] covering the full frozen 63-failed set:
#   k2.5g100  = wc baseline replication (prior readout 18/63; same set,
#                same windows, same P -- big drift = protocol alarm)
#   k5.0g100  = cap raised
#   k7.5g100  = cap raised further (tests whether the kappa cap was binding)
#   k2.5g50   = guidance only on the last 50 denoise steps (halved cloud)
# eta 3.0 / adapt tau / hist_max 0 / TRIES=5 / P=6 everywhere; aty arm NOT
# rerun -- its bar is the existing 23/63 on the same frozen set.
#
# usage: bash scripts/probe/sq826_walkcloud_ksweep.sh   (server, /tmp/scout-drift)
set -uo pipefail
cd /tmp/scout-drift || { echo "[ksweep] FATAL: scout dir missing"; exit 1; }
SH=scripts/probe/sq826_walkcloud_probe.sh
BASE=data/walkcloud_ksweep_sq
SRC=data/walkcloud_probe_sq826
[ -f "$SH" ] || { echo "[ksweep] FATAL: probe script missing"; exit 1; }
[ -f "$SRC/failed_dp.json" ] && [ -f "$SRC/square_core.hdf5" ] \
  || { echo "[ksweep] FATAL: frozen failed set / core missing under $SRC"; exit 1; }

run_cfg() { # name gpu cap gst
  local name=$1 gpu=$2 cap=$3 gst=$4
  local root=$BASE/$name
  mkdir -p "$root"
  cp -n "$SRC/failed_dp.json" "$SRC/square_core.hdf5" "$root/" 2>/dev/null || true
  for w in "1 0 32" "2 32 31"; do
    set -- $w
    echo "[ksweep] cfg=$name gpu=$gpu k=$cap gst=$gst window r$1=[$2:$3] start $(date '+%F %T')"
    ROUND=$1 OFF=$2 N=$3 P=6 TRIES=5 ARMS=wc CAP=$cap GST=$gst \
      GPU_WC=$gpu ROOT=$root bash "$SH" \
      > "$root/r$1.probe.log" 2>&1
    echo "[ksweep] cfg=$name r$1 rc=$? $(date '+%F %T')"
  done
  echo "[ksweep] CFG $name DONE (k=$cap gst=$gst) $(date '+%F %T')"
}

run_cfg k2.5g100 1 2.5 100 &
run_cfg k5.0g100 3 5.0 100 &
run_cfg k7.5g100 5 7.5 100 &
run_cfg k2.5g50 6 2.5 50 &
wait
echo "[ksweep] ALL DONE $(date '+%F %T')"
