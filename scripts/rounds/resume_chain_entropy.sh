#!/bin/bash
# resume_chain_entropy.sh -- run rounds S..E of ONE arm of the FORMAL entropy
# experiment (round_entropy.sh; worktree repo), after an interruption.
# Copy of resume_chain.sh (2026-08-23) with: REPO=scout-entropy,
# round_entropy.sh, NENV passthrough (2026-08-25 user: env=50 relaunch).
# Usage: resume_chain_entropy.sh <task> <SCOUT|DP> <start> <end> [skip-first-rollout]
# Env:   GPU TSEED DATA_ROOT WPROJ (required)  NENV=50 NROUNDS=6
set -u
TASK=${1:?usage: resume_chain_entropy.sh <task> <arm> <start> <end> [skip-first]}
A=${2:?usage: resume_chain_entropy.sh <task> <arm> <start> <end> [skip-first]}
S=${3:?usage: resume_chain_entropy.sh <task> <arm> <start> <end> [skip-first]}
E=${4:?usage: resume_chain_entropy.sh <task> <arm> <start> <end> [skip-first]}
SKIPF=${5:-0}
cd /root/workspace/baojiachun/scout-entropy || exit 1
CLAIM="${DATA_ROOT:?set DATA_ROOT}/.resume_claim_${TASK}_${A}_${S}-${E}"
if [ -f "$CLAIM" ]; then
  echo "[chain] resume $TASK $A $S..$E already claimed ($(cat "$CLAIM")) -- skipping"
  exit 0
fi
echo "$(date '+%F %T') gpu=${GPU:-?} nenv=${NENV:-12}" > "$CLAIM"
for n in $(seq "$S" "$E"); do
  if [ "$n" = "$S" ] && [ "$SKIPF" = "1" ]; then
    SKIP_ROLLOUT=1 bash soe_scripts/round_entropy.sh "$TASK" "$A" "$n" \
      || { echo "[chain] round $n FAILED"; exit 1; }
  else
    bash soe_scripts/round_entropy.sh "$TASK" "$A" "$n" \
      || { echo "[chain] round $n FAILED"; exit 1; }
  fi
done
echo "[chain] $TASK a=$A seed=${TSEED:?} rounds $S..$E DONE $(date '+%F %T')"
