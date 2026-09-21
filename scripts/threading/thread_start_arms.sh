#!/bin/bash
# thread_start_arms.sh -- start the THREE arms of one THREADING-MG-p1 seed
# (2026-09-16; 3 seeds x 3 arms = 9 chains, ETRIES=5 pass@5, GPU1-6 shared
# via SCOUT_RENDER_GPU rotation -- tp15 9-chain 23.1h zero-ABORT precedent).
#
# Prereq: round0 TOTAL gate line present in
#   data/2026_9_16_threading_p1/THREADING-MG-p1-s<SEED>/threading/round.log
# (written by the BASE branch of thread_chain.sh after DP+dyn-base train).
#
# usage: ATY_SCALE=<s> [ORB_SIGMA=0.05 ORB_SIGMA_DECAY=.5 ORB_LAM=.5
#          ORB_DELTA=.25 ORB_ANNEAL=2 ATT_CAP=2.5] \
#        bash scripts/threading/thread_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>
set -u
SEED=${1:?usage: thread_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>}
G_DP=${2:?set DP GPU}; G_ATY=${3:?set ATY GPU}; G_ORB=${4:?set ORBIT GPU}
ATT_CAP=${ATT_CAP:-2.5}
# Single SHARED dose (user 2026-09-17: 三个臂同一组参数, seed-233 grid verdict).
# ATY and ORBIT both run --guidance-scale $ATY_SCALE; the DP arm ignores it.
ATY_SCALE=${ATY_SCALE:?set ATY_SCALE=<grid verdict: combined pass@5 argmax on s233>}
ORB_LAM=${ORB_LAM:-0.5}
ORB_DELTA=${ORB_DELTA:-0.25}
ORB_SIGMA=${ORB_SIGMA:-0.05}
ORB_SIGMA_DECAY=${ORB_SIGMA_DECAY:-0.5}
ORB_ANNEAL=${ORB_ANNEAL:-2}
unset DATA_ROOT XMODE WNAME_BASE DP_EPOCHS_SOE DYN_EPOCHS_SOE

ROOT=/root/workspace/baojiachun/scout
DR=$ROOT/data/2026_9_16_threading_p1/THREADING-MG-p1-s$SEED
cd "$ROOT" || exit 1

# round0 must be complete before arms start
grep -q "ROUND0 threading seed=$SEED TOTAL" "$DR/threading/round.log" 2>/dev/null \
  || { echo "[arms] FATAL: round0 TOTAL gate line missing for s$SEED"; exit 1; }

for spec in "thmg_dp_s$SEED:$G_DP:DP" "thmg_aty_s$SEED:$G_ATY:ATY" "thmg_orbit_s$SEED:$G_ORB:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[arms] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=$SEED GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 CORE_N=20 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=$ATT_CAP ATY_SCALE=$ATY_SCALE ORB_LAM=$ORB_LAM ORB_DELTA=$ORB_DELTA ORB_SIGMA=$ORB_SIGMA ORB_SIGMA_DECAY=$ORB_SIGMA_DECAY ORB_ANNEAL=$ORB_ANNEAL bash scripts/threading/thread_chain.sh"
  echo "[arms] spawned $S @GPU$G ($A, ETRIES=5 pass@5, shared ATY_SCALE=$ATY_SCALE, ORB_SIGMA=$ORB_SIGMA)"
done
echo "[arms] seed s$SEED arms spawned $(date '+%F %T'); consoles: $DR/chain_{DP,ATY,ORBIT}.console.log"
