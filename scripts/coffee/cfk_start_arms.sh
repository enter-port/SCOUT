#!/bin/bash
# cfk_start_arms.sh -- start the THREE arms of one COFFEE-MG-p1 seed
# (2026-09-20; 3 seeds x 3 arms = 9 chains, ETRIES=5 pass@5, one chain per
# GPU -- THREADING-P3/COFFEEPREP-p1 9-chain precedent).
#
# Prereq: round0 TOTAL gate line present in
#   data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s<SEED>/coffee/round.log
# (written by the BASE branch of cfk_chain.sh after DP+dyn-base train).
#
# usage: ATY_SCALE=<s> [ORB_SIGMA=0.05 ORB_SIGMA_DECAY=.5 ORB_LAM=.5
#          ORB_DELTA=.25 ORB_ANNEAL=2 ATT_CAP=2.5] \
#        bash scripts/coffee/cfk_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>
set -u
SEED=${1:?usage: cfk_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>}
G_DP=${2:?set DP GPU}; G_ATY=${3:?set ATY GPU}; G_ORB=${4:?set ORBIT GPU}
ATT_CAP=${ATT_CAP:-2.5}
# Single SHARED dose (user 2026-09-20, verbatim: "eta 取两引导组 pass@5 之和
# 最大,如果有并列你来选 不要汇报不要停" -- ties -> higher min(ATY,ORBIT)
# pass@5, then LOWER eta; delegated to the agent).
# ATY and ORBIT both run --guidance-scale $ATY_SCALE; the DP arm ignores it.
ATY_SCALE=${ATY_SCALE:?set ATY_SCALE=<grid verdict: max(aty+orb) pass@5 on s233>}
ORB_LAM=${ORB_LAM:-0.5}
ORB_DELTA=${ORB_DELTA:-0.25}
ORB_SIGMA=${ORB_SIGMA:-0.05}
ORB_SIGMA_DECAY=${ORB_SIGMA_DECAY:-0.5}
ORB_ANNEAL=${ORB_ANNEAL:-2}
unset DATA_ROOT XMODE WNAME_BASE DP_EPOCHS_SOE DYN_EPOCHS_SOE

ROOT=/root/workspace/baojiachun/scout
DR=$ROOT/data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s$SEED
cd "$ROOT" || exit 1

# round0 must be complete before arms start
grep -q "ROUND0 coffee seed=$SEED TOTAL" "$DR/coffee/round.log" 2>/dev/null \
  || { echo "[arms] FATAL: round0 TOTAL gate line missing for s$SEED"; exit 1; }

for spec in "cfk_dp_s$SEED:$G_DP:DP" "cfk_aty_s$SEED:$G_ATY:ATY" "cfk_orbit_s$SEED:$G_ORB:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[arms] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=$SEED GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 CORE_N=20 SHARD_P=5 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=$ATT_CAP ATY_SCALE=$ATY_SCALE ORB_LAM=$ORB_LAM ORB_DELTA=$ORB_DELTA ORB_SIGMA=$ORB_SIGMA ORB_SIGMA_DECAY=$ORB_SIGMA_DECAY ORB_ANNEAL=$ORB_ANNEAL bash scripts/coffee/cfk_chain.sh"
  echo "[arms] spawned $S @GPU$G ($A, ETRIES=5 pass@5, shared ATY_SCALE=$ATY_SCALE, ORB_SIGMA=$ORB_SIGMA)"
done
echo "[arms] seed s$SEED arms spawned $(date '+%F %T'); consoles: $DR/chain_{DP,ATY,ORBIT}.console.log"
