#!/bin/bash
# tp13_start_arms.sh -- start the THREE arms of one TRANSPORT-9-13 seed AFTER
# the dose decision (2026-09-13 user flow: round0 -> dose probe -> user picks
# dose -> THIS script). Requires the final dose knobs as explicit envs so the
# decision is pinned in the tmux command strings (P1-1 惯例: every dose knob
# unset from the operator shell AND pinned inside each tmux string).
#
# usage: ATY_SCALE=<s> ORB_SIGMA=<sig> [ORB_SIGMA_DECAY=.5 ORB_LAM=.5
#          ORB_DELTA=.25 ORB_ANNEAL=2 ATT_CAP=2.5] \
#        bash scripts/transport/tp13_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU> [PORT]
#   PORT: ssh port of the container the seed lives on (1022 default; 23333=1024).
#   Run ON that container (or via the matching ssh port).
set -u
SEED=${1:?usage: tp13_start_arms.sh <SEED> <DP_GPU> <ATY_GPU> <ORBIT_GPU>}
G_DP=${2:?set DP GPU}; G_ATY=${3:?set ATY GPU}; G_ORB=${4:?set ORBIT GPU}
ATT_CAP=${ATT_CAP:-2.5}
ATY_SCALE=${ATY_SCALE:?set ATY_SCALE=<final post-probe raw dose for atypical+orbit>}
ORB_LAM=${ORB_LAM:-0.5}
ORB_DELTA=${ORB_DELTA:-0.25}
ORB_SIGMA=${ORB_SIGMA:?set ORB_SIGMA=<final post-probe tangential noise std>}
ORB_SIGMA_DECAY=${ORB_SIGMA_DECAY:-0.5}
ORB_ANNEAL=${ORB_ANNEAL:-2}
unset DATA_ROOT XMODE WNAME_BASE DP_EPOCHS_SOE DYN_EPOCHS_SOE

ROOT=/root/workspace/baojiachun/scout
DR=$ROOT/data/2026_9_13_transport/TRANSPORT-s$SEED
cd "$ROOT" || exit 1

# round0 must be complete before arms start
grep -q "ROUND0 transport seed=$SEED TOTAL" "$DR/transport/round.log" 2>/dev/null \
  || { echo "[arms] FATAL: round0 TOTAL line missing for s$SEED -- run round0 first"; exit 1; }

for spec in "tp13_dp_s$SEED:$G_DP:DP" "tp13_aty_s$SEED:$G_ATY:ATY" "tp13_orbit_s$SEED:$G_ORB:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[arms] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=$SEED GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 CORE_N=20 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=$ATT_CAP ATY_SCALE=$ATY_SCALE ORB_LAM=$ORB_LAM ORB_DELTA=$ORB_DELTA ORB_SIGMA=$ORB_SIGMA ORB_SIGMA_DECAY=$ORB_SIGMA_DECAY ORB_ANNEAL=$ORB_ANNEAL bash scripts/transport/tp13_chain.sh"
  echo "[arms] spawned $S @GPU$G ($A, ATY_SCALE=$ATY_SCALE ORB_SIGMA=$ORB_SIGMA)"
done
echo "[arms] seed s$SEED arms spawned $(date '+%F %T'); consoles: $DR/chain_{DP,ATY,ORBIT}.console.log"
