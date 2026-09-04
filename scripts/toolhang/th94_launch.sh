#!/bin/bash
# th94_launch.sh -- TOOLHANG-9-4-orbit-s233 arms-only orchestrator (2026-09-04
# user order). Round0 already trained (th9_4_round0_launch.sh, rc=0 17:16;
# 40-demo core + DP-base 600ep + dyn-base 20260904-170403) -- this starts the
# two chain arms directly: SCOUT(atypical s1.0/cap2.5/gst50 raw)@GPU1 +
# DP(guide off)@GPU3, 6 rounds (r1-5 full + r6 eval-only), 4 shard workers x
# 25 envs per arm, explore with --flush-every 100 (TrajSpool OOM fix).
# wandb project TOOLHANG-9-4-orbit-s233, run names DP-round{i}/SCOUT-round{i}.
set -u
ROOT=/root/workspace/baojiachun/scout-th94
cd "$ROOT" || exit 1
unset DATA_ROOT   # stale export must not redirect this campaign

for spec in "th94_scout_233:1:SCOUT" "th94_dp_233:3:DP"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[launch] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=233 GPU=$G ARM=$A NROUNDS=6 SHARD_P=4 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=10 WNAME_BASE=$A bash scripts/toolhang/th94_chain.sh"
done
echo "[launch] arms spawned: th94_scout_233@GPU1 th94_dp_233@GPU3 $(date '+%F %T')"
