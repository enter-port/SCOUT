#!/bin/bash
# th6_restart_launch.sh -- TOOLHANG-9-1-orbit-s233 restart orchestrator (2026-09-03).
# User spec: seed 233 fresh campaign -- round0 (20/200 seeded split + base DP
# 600ep + dyn-base) on GPU1, then two arms SCOUT(orbit)@GPU1 + DP@GPU3,
# 6 rounds (r1-5 full + r6 eval-only), explore = 4 shard workers x 25 envs.
set -u
ROOT=/root/workspace/baojiachun/scout-orbit
mkdir -p "$ROOT/data/2026_9_1_toolhang"
cd "$ROOT" || exit 1
unset DATA_ROOT   # stale export must not redirect the real campaign
export NROUNDS=6 SHARD_P=4 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6

echo "[launch] round0 START $(date '+%F %T') (GPU1, seed 233)"
SEED=233 GPU=1 ARM=BASE bash soe_scripts/th_chain.sh
rc=$?
echo "[launch] round0 rc=$rc $(date '+%F %T')"
if [ $rc -ne 0 ]; then
  echo "[launch] round0 FAILED -- arms NOT started"
  exit 1
fi

tmux new-session -d -s th6_orb_233 \
  "cd $ROOT && SEED=233 GPU=1 ARM=SCOUT NROUNDS=6 SHARD_P=4 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 bash soe_scripts/th_chain.sh"
tmux new-session -d -s th6_dp_233 \
  "cd $ROOT && SEED=233 GPU=3 ARM=DP NROUNDS=6 SHARD_P=4 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 bash soe_scripts/th_chain.sh"
echo "[launch] arms spawned: th6_orb_233@GPU1 th6_dp_233@GPU3 $(date '+%F %T')"
