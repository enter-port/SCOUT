#!/bin/bash
# tp15_start_arms.sh -- start the THREE arms of one TRANSPORT-9-15-p1 seed
# (pass@1 campaign; 2026-09-15 user order: redo 3 seeds x 3 arms with
# ETRIES=1 after fixing the wandb explore/pass@10 label bug in round_tp.sh --
# backfill keys are now explore/pass@{ETRIES} taken from the rollout json).
#
# Prereq: tp15_prep_base.sh has copied the base three-piece + written the
# ROUND0 TOTAL gate line into data/2026_9_15_transport_p1/TRANSPORT-s<SEED>.
#
# usage: ATY_SCALE=<s> ORB_SIGMA=<sig> [ORB_SIGMA_DECAY=.5 ORB_LAM=.5
#          ORB_DELTA=.25 ORB_ANNEAL=2 ATT_CAP=2.5] \
#        bash scripts/transport/tp15_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>
#   Run ON the container the seed lives on (233/2333 -> port 1022;
#   23333 -> port 1024; shared CPFS so prep done from 1022 is visible).
set -u
SEED=${1:?usage: tp15_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>}
G_DP=${2:?set DP GPU}; G_ATY=${3:?set ATY GPU}; G_ORB=${4:?set ORBIT GPU}
ATT_CAP=${ATT_CAP:-2.5}
ATY_SCALE=${ATY_SCALE:?set ATY_SCALE=<post-probe raw dose (th95 mirror: 0.5)>}
ORB_LAM=${ORB_LAM:-0.5}
ORB_DELTA=${ORB_DELTA:-0.25}
ORB_SIGMA=${ORB_SIGMA:?set ORB_SIGMA=<post-probe tangential noise std (th95 mirror: 0.05)>}
ORB_SIGMA_DECAY=${ORB_SIGMA_DECAY:-0.5}
ORB_ANNEAL=${ORB_ANNEAL:-2}
unset DATA_ROOT XMODE WNAME_BASE DP_EPOCHS_SOE DYN_EPOCHS_SOE

ROOT=/root/workspace/baojiachun/scout
DR=$ROOT/data/2026_9_15_transport_p1/TRANSPORT-s$SEED
cd "$ROOT" || exit 1

# round0 base must be prepped (copied) before arms start
grep -q "ROUND0 transport seed=$SEED TOTAL" "$DR/transport/round.log" 2>/dev/null \
  || { echo "[arms] FATAL: round0 TOTAL gate line missing for s$SEED -- run tp15_prep_base.sh first"; exit 1; }

for spec in "tp15_dp_s$SEED:$G_DP:DP" "tp15_aty_s$SEED:$G_ATY:ATY" "tp15_orbit_s$SEED:$G_ORB:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[arms] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=$SEED GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 CORE_N=20 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=1 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=$ATT_CAP ATY_SCALE=$ATY_SCALE ORB_LAM=$ORB_LAM ORB_DELTA=$ORB_DELTA ORB_SIGMA=$ORB_SIGMA ORB_SIGMA_DECAY=$ORB_SIGMA_DECAY ORB_ANNEAL=$ORB_ANNEAL bash scripts/transport/tp15_chain.sh"
  echo "[arms] spawned $S @GPU$G ($A, ETRIES=1 pass@1, ATY_SCALE=$ATY_SCALE ORB_SIGMA=$ORB_SIGMA)"
done
echo "[arms] seed s$SEED arms spawned $(date '+%F %T'); consoles: $DR/chain_{DP,ATY,ORBIT}.console.log"
