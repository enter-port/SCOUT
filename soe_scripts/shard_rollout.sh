#!/bin/bash
# shard_rollout.sh -- multicore scene-sharded rescue rollout (orbit-dev, 2026-09-01).
#
# Launches P run_rollout workers on ONE GPU (each single-core, py-spy 09-01:
# the guided rollout is CPU-bound serial Python), waits, then merges the
# suffixed shard outputs into the monolithic-equivalent result set.
#
# Usage:
#   bash soe_scripts/shard_rollout.sh P OUT_JSON OUT_SUCCESS OUT_ALL CORE_HDF5 -- <run_rollout args...>
#
#   P           number of shards (= worker processes; try 2-4 per GPU first)
#   OUT_*       the UNSUFFIXED merge targets; workers write
#               ${OUT}-shard{i}of{P} themselves, the merge writes OUT_*
#   CORE_HDF5   core-only hdf5 for the success/all union merge
#   <args...>   everything you would pass to run_rollout for the MONOLITHIC
#               run, INCLUDING --output-success/--output-all pointing at the
#               UNSUFFIXED paths (the suffixing happens inside run_rollout).
#               Do NOT pass --output-json per shard (driver passes the merged
#               json path via --output-json so all shards share the pattern).
#
# Example (square orbit rescue, 4 workers on GPU0):
#   bash soe_scripts/shard_rollout.sh 4 \
#       data/shardtest/merged.json data/shardtest/success.hdf5 \
#       data/shardtest/all.hdf5 data/square_core.hdf5 -- \
#       --config configs/eval_square_entropy.yaml --task square \
#       --explore-mode rescue --explore-try-times 10 --guide orbit \
#       --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 \
#       --n-envs 12 --base-dp-ckpt ... --vib-ckpt ... \
#       --output-dir data/shardtest --output-success data/shardtest/success.hdf5 \
#       --output-all data/shardtest/all.hdf5
#
# Statistical-equivalence note: shards draw from per-worker RNG streams -- the
# merged result equals the monolithic protocol on the same scene set but is
# NOT bit-identical to a monolithic run (see merge_sharded.py docstring).
#
# INTENDED FLOW (else near-zero speedup -- slicing happens AFTER phase 1, so
# without a frozen failed set every worker redundantly rolls the FULL eval):
#   1) ONCE, monolithic: run_rollout ... --save-failed-set failed.json
#   2) then shards:      pass --failed-set-json failed.json in <args...>
# Do NOT combine --save-failed-set with shards (all P workers would race on
# the same json path). CORE_HDF5 must be the SAME file the workers' --core-hdf5
# points at (the hdf5 union classifies appended demos by demo-id >= the core's
# max -- a mismatched core silently corrupts the merge).
set -e
P=$1; OUT_JSON=$2; OUT_SUCCESS=$3; OUT_ALL=$4; CORE=$5
[ "$6" = "--" ] || { echo "expected '--' separator after CORE_HDF5"; exit 2; }
shift 6
ARGS="$@"
mkdir -p "$(dirname "$OUT_JSON")" "$(dirname "$OUT_SUCCESS")" "$(dirname "$OUT_ALL")"

PIDS=()
for i in $(seq 0 $((P - 1))); do
  echo "[shard_rollout] launching shard $i/$P"
  python -m scout.eval.run_rollout --scene-slice "$i:$P" \
      --output-json "$OUT_JSON" "$ARGS" \
      > "$(dirname "$OUT_JSON")/shard$i.stdout" 2>&1 &
  PIDS+=($!)
done
RC=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || { echo "[shard_rollout] worker $pid FAILED"; RC=1; }
done
[ $RC -eq 0 ] || { echo "[shard_rollout] one or more workers failed -- NOT merging"; exit 1; }

SUFFIXED_JSON=(); SUFFIXED_S=(); SUFFIXED_A=()
for i in $(seq 0 $((P - 1))); do
  SUFFIXED_JSON+=("${OUT_JSON%.*}-shard${i}of${P}.${OUT_JSON##*.}")
  SUFFIXED_S+=("${OUT_SUCCESS%.*}-shard${i}of${P}.${OUT_SUCCESS##*.}")
  SUFFIXED_A+=("${OUT_ALL%.*}-shard${i}of${P}.${OUT_ALL##*.}")
done
python -m scout.eval.merge_sharded \
    --jsons "${SUFFIXED_JSON[@]}" --out-json "$OUT_JSON" \
    --success-hdf5s "${SUFFIXED_S[@]}" --out-success "$OUT_SUCCESS" \
    --all-hdf5s "${SUFFIXED_A[@]}" --out-all "$OUT_ALL" \
    --core-hdf5 "$CORE"
echo "[shard_rollout] merged -> $OUT_JSON $OUT_SUCCESS $OUT_ALL"
