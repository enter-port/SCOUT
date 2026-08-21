#!/bin/bash
# Sequential round LOOP driver (user 2026-08-18): run one chain's rounds
# 1..N inside a single session -- no external auto-chaining needed.
#   GPU=<n> bash soe_scripts/run_rounds.sh <task> <A> [driver] [N]
# Driver defaults to round_e2.sh; N defaults to 6. Stops the loop on the
# first FAILED round (rc != 0).
# 2026-08-18 user: the LAST round (n == N) defaults to eval-only -- final
# measurement of the round N-1 retrained DP, no explore/retrain.
TASK=${1:?task}
A=${2:?a}
DRIVER=${3:-round_e2.sh}
N=${4:-6}
for n in $(seq 1 "$N"); do
  if [ "$n" -eq "$N" ]; then
    MODE=eval-only
  else
    MODE=full
  fi
  echo "[loop $(date '+%F %T')] === $TASK a=$A round $n/$N ($MODE) begin (driver $DRIVER) ==="
  if ! bash "soe_scripts/$DRIVER" "$TASK" "$A" "$n" "$MODE"; then
    echo "[loop $(date '+%F %T')] round $n FAILED -- stopping loop"
    exit 1
  fi
done
echo "[loop $(date '+%F %T')] === $TASK a=$A all $N rounds complete ==="
