#!/usr/bin/env bash
# rand_pshell_chain.sh -- pshell 20-scene screen, three arms SERIAL on GPU2
# (campaign re-aim 2026-08-28: main metric = retry-distribution WIDTH;
# SR is a guard rail only).  SCALE=0.35 = 方案A dose-response peak.
set -u
W=/root/workspace/baojiachun/scout-rand
cd $W
run_arm() {
  local tag=$1 kwargs=$2
  echo "[pshell_chain] START $tag kwargs=$kwargs $(date)"
  SCALE=0.35 bash $W/soe_scripts/rand_screen.sh "$tag" rand_pshell 2 \
    --rand-kwargs "$kwargs"
  echo "[pshell_chain] END   $tag rc=$? $(date)"
}
run_arm pshell_r_k25 "rand_anchor_refresh=retry,shell_kappa=2.5"
run_arm pshell_r_k5  "rand_anchor_refresh=retry,shell_kappa=5.0"
run_arm pshell_c_k25 "rand_anchor_refresh=chunk,shell_kappa=2.5"
echo "[pshell_chain] ALL DONE $(date)"
