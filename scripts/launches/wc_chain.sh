#!/bin/bash
# wc_chain.sh -- sequence the walkcloud round0 probes on port1024
# (user order 2026-09-10: can/square/toolhang round0, 45 min each arm pair).
#
# chain a (GPU 1/3): can r1 (full 43) -> toolhang r1 (N=48) -> toolhang r2 (OFF=48 N=21)
# chain b (GPU 5/6): square r1 (N=32) -> square r2 (OFF=32 N=31)
#
# Stages run sequentially (each blocks until its probe finishes -- the
# per-arm 45-min backstop bounds every stage, so a crashed stage just
# yields to the next one instead of deadlocking the chain).
#
# usage: bash scripts/launches/wc_chain.sh a|b   (from /tmp/scout-drift)
set -u
cd /tmp/scout-drift || exit 1

run() { # <round> <log> <script> [VAR=val...]
  local rnd=$1 log=$2 script=$3; shift 3
  echo "[wc-chain] $(date '+%F %T') starting round=$rnd $script $*"
  env "$@" ROUND=$rnd bash "$script" 2>&1 | tee -a "$log"
  echo "[wc-chain] $(date '+%F %T') stage round=$rnd finished"
}

if [ "${1:-}" = "a" ]; then
  mkdir -p data/walkcloud_probe_can824 data/walkcloud_probe_th94
  run 1 data/walkcloud_probe_can824/r1.probe.log scripts/probe/can824_walkcloud_probe.sh \
    N=0 P=4 GPU_WC=1 GPU_ATY=3
  run 1 data/walkcloud_probe_th94/r1.probe.log scripts/probe/th94_walkcloud_probe.sh \
    N=48 P=4 GPU_WC=1 GPU_ATY=3
  run 2 data/walkcloud_probe_th94/r2.probe.log scripts/probe/th94_walkcloud_probe.sh \
    OFF=48 N=21 P=4 GPU_WC=1 GPU_ATY=3
elif [ "${1:-}" = "b" ]; then
  mkdir -p data/walkcloud_probe_sq826
  run 1 data/walkcloud_probe_sq826/r1.probe.log scripts/probe/sq826_walkcloud_probe.sh \
    N=32 P=6 GPU_WC=5 GPU_ATY=6
  run 2 data/walkcloud_probe_sq826/r2.probe.log scripts/probe/sq826_walkcloud_probe.sh \
    OFF=32 N=31 P=6 GPU_WC=5 GPU_ATY=6
else
  echo "usage: bash scripts/launches/wc_chain.sh a|b"; exit 1
fi
echo "[wc-chain] chain $1 complete $(date '+%F %T')"
