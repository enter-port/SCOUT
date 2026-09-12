#!/bin/bash
# one-off recovery (2026-08-19): round1 rollout already completed (a driver
# bug -- missing RC assignment -- killed the chain AT the post-rollout check,
# data intact). Finish round1's retrain stages via SKIP_ROLLOUT=1, then loop
# rounds 2..5 with the last round eval-only (same convention as run_rounds).
#   GPU=<n> bash soe_scripts/e3_resume.sh <task> <A>
set -u
TASK=${1:?task}; A=${2:?a}
SKIP_ROLLOUT=1 bash soe_scripts/round_e3.sh "$TASK" "$A" 1 full || { echo '[resume] round1 resume FAILED'; exit 1; }
for n in 2 3 4; do
  bash soe_scripts/round_e3.sh "$TASK" "$A" "$n" full || { echo "[resume] round $n FAILED"; exit 1; }
done
bash soe_scripts/round_e3.sh "$TASK" "$A" 5 eval-only || { echo '[resume] round 5 FAILED'; exit 1; }
echo "[resume $(date '+%F %T')] $TASK a=$A rounds 1..5 complete"
