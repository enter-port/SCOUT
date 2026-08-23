#!/bin/bash
# resume_chain.sh -- run rounds S..E for one arm, optionally replaying the
# first round's [2/3]+[3/3] only (SKIP_ROLLOUT=1, rollout data already valid).
# Usage: resume_chain.sh <task> <SCOUT|DP> <start> <end> [skip-first-rollout]
# Env:   GPU TSEED DATA_ROOT WPROJ (required, passed through to round.sh)
set -u
TASK=${1:?usage: resume_chain.sh <task> <arm> <start> <end> [skip-first]}
A=${2:?usage: resume_chain.sh <task> <arm> <start> <end> [skip-first]}
S=${3:?usage: resume_chain.sh <task> <arm> <start> <end> [skip-first]}
E=${4:?usage: resume_chain.sh <task> <arm> <start> <end> [skip-first]}
SKIPF=${5:-0}
cd /root/workspace/baojiachun/scout || exit 1
# one-shot claim (2026-08-23): a sequential watcher may re-invoke this same
# resume hours later -- refuse to re-run a range that was already claimed
# (crash recovery: delete the claim file to allow a manual re-run).
CLAIM="${DATA_ROOT:?set DATA_ROOT}/.resume_claim_${TASK}_${A}_${S}-${E}"
if [ -f "$CLAIM" ]; then
  echo "[chain] resume $TASK $A $S..$E already claimed ($(cat "$CLAIM")) -- skipping"
  exit 0
fi
echo "$(date '+%F %T') gpu=${GPU:-?}" > "$CLAIM"
for n in $(seq "$S" "$E"); do
  if [ "$n" = "$S" ] && [ "$SKIPF" = "1" ]; then
    SKIP_ROLLOUT=1 bash soe_scripts/round.sh "$TASK" "$A" "$n" \
      || { echo "[chain] round $n FAILED"; exit 1; }
  else
    bash soe_scripts/round.sh "$TASK" "$A" "$n" \
      || { echo "[chain] round $n FAILED"; exit 1; }
  fi
done
echo "[chain] $TASK a=$A rounds $S..$E ALL DONE $(date '+%F %T')"
