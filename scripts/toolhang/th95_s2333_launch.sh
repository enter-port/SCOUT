#!/bin/bash
# th95_s2333_launch.sh -- TOOLHANG-9-5-orbit-s2333 orchestrator (2026-09-08
# user order: identical three-arm spec to TOOLHANG-9-5-orbit-s233, seed
# 233->2333, r6 eval must carry SR + pass@5). Differences from th95_launch.sh:
#   * NO pre-copied base -- seed 2333 has no round0 assets anywhere, so this
#     runs the REAL round0 (split 40/200 rng 2333 + base DP 600ep + dyn-base,
#     ~3.5h per the 9-4 round0 TOTAL 195m46s) on GPU5, then spawns the arms
#     on rc=0. Kick this off inside tmux th95_r0_2333 (never via blocking ssh).
#   * SEED=2333 everywhere; the chain derives DATA_ROOT
#     .../2026_9_5_toolhang/TOOLHANG-s2333 and WPROJ
#     TOOLHANG-9-5-orbit-s2333 from SEED (chain/round scripts untouched).
#   * arms @GPU0/2/4 same mapping as s233 (all free as of 09-08; GPU6 = SOE
#     baseline campaign, GPU7 ECC-forbidden).
#   * r6 = eval-pk inside th95_chain.sh (SR + pass@ETRIES, retrains skipped).
# Review P1-1 pattern kept: every dose knob is unset from the operator shell
# AND pinned explicitly inside each tmux command string.
set -u
ROOT=/root/workspace/baojiachun/scout-th95
DR=$ROOT/data/2026_9_5_toolhang/TOOLHANG-s2333
cd "$ROOT" || exit 1
unset DATA_ROOT ATT_CAP ATY_SCALE ORB_LAM ORB_DELTA ORB_SIGMA ORB_SIGMA_DECAY \
      ORB_ANNEAL DP_EPOCHS_SOE DYN_EPOCHS_SOE XMODE WNAME_BASE
mkdir -p "$DR"

echo "[launch] round0 START (REAL: split rng 2333 + DP 600ep + dyn-base, GPU5) $(date '+%F %T')"
SEED=2333 GPU=5 ARM=BASE DATA_ROOT=$DR bash scripts/toolhang/th95_chain.sh
rc=$?
echo "[launch] round0 rc=$rc $(date '+%F %T')"
if [ $rc -ne 0 ]; then
  echo "[launch] round0 FAILED -- arms NOT started (see $DR/chain_BASE.console.log)"
  exit 1
fi

for spec in "th95_dp_2333:0:DP" "th95_aty_2333:2:ATY" "th95_orbit_2333:4:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[launch] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=2333 GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=2.5 ATY_SCALE=0.5 ORB_LAM=0.5 ORB_DELTA=0.25 ORB_SIGMA=0.05 ORB_SIGMA_DECAY=0.5 ORB_ANNEAL=2 bash scripts/toolhang/th95_chain.sh"
done
echo "[launch] arms spawned: th95_dp_2333@GPU0 th95_aty_2333@GPU2 th95_orbit_2333@GPU4 $(date '+%F %T')"
