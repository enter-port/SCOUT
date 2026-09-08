#!/bin/bash
# launch_thsoe233.sh -- SOE tool_hang baseline, seed 233 (user order
# 2026-09-07): the SOE counterpart of TOOLHANG-9-5-orbit-s233.
#
# Protocol mirroring TH9-5 (verified against round_th95.sh / th95_chain.sh):
#   * 100 eval scenes, seed base 42..141 (SCOUT-identical mechanism);
#   * rescue x ETRIES=5 from the same init states -> pass@5;
#   * 6 SOE rounds k=0..5 == SCOUT rounds r1..r6; every round is
#     train-then-eval, so the LAST round ends with eval+rescue only and no
#     training happens after the final measurement (k=5 trains on k=4's
#     accumulated data to PRODUCE the r6 model, exactly like SCOUT r6 uses
#     the r5-retrained DP);
#   * core 40 demos @ s233, proven identical to SCOUT's by prep_toolhang_soe.sh
#     (count/length cross-check + value-level pos/grip/rot6d comparison);
#   * DP retrain budget = SOE baseline convention EP0=EPOCHS=1000 x 100 iters
#     (known, recorded difference vs SCOUT's 300ep).
#
# Env = can/square SOE baseline defaults except (all via round_soe.sh env):
#   HORIZON=700  (tool_hang max_steps; can 300 / square 500 precedent)
#   ETRIES=5     (user order: pass@5; can/square were 10)
#   VISGATE=0    (vis_validate_soe greps agentview_image which tool_hang
#                 lacks -- same off-ruling as square; SCOUT th95 rounds run
#                 no render gate either)
#   SAVE0=SAVE=200 (square precedent, disk hygiene; policy_last.ckpt is what
#                 round_soe consumes)
#   DATASETS=soe_data/datasets/tool_hang, TEMPLATE default resolves to
#                 config_template/tool_hang_soe.json (five obs keys identical
#                 to SCOUT's toolhang DP: sideview+wrist rgb, eef pos/quat/
#                 gripper_qpos; no object).
# GPU6 (idle, ECC=0 at launch time; GPU2/4 held by th95, GPU7 defective).
# Cosmetic caveat: round_soe.sh's log line prints "pass@10=" from metrics
# field pass_at_10 -- with try_times=5 that field is an alias of pass@5.
set -eu
BASE=/root/workspace/baojiachun
SEED=233
GPU=6
LOGROOT=$BASE/soe_data/2026_9_7_soeth/logs
ROOT=$BASE/soe_data/2026_9_7_soeth/SOE-s$SEED
CORE=$BASE/soe_data/datasets/tool_hang/tool_hang_core_soe_s$SEED.hdf5
mkdir -p "$LOGROOT" "$ROOT"

[ -f "$CORE" ] || { echo "FATAL: $CORE missing -- run prep_toolhang_soe.sh first"; exit 1; }
[ -f "$BASE/SOE/simulation/config_template/tool_hang_soe.json" ] \
  || { echo "FATAL: tool_hang_soe.json template missing"; exit 1; }
if tmux has-session -t "soe_th${SEED}_chain" 2>/dev/null; then
  echo "tmux soe_th${SEED}_chain already exists -- abort (no double spawn)"
  exit 1
fi

tmux new-session -d -s "soe_th${SEED}_chain" \
  "GPU=$GPU TSEED=$SEED DATA_ROOT=$ROOT WPROJ=TOOLHANG-9-7-SOE-s$SEED \
   DATASETS=$BASE/soe_data/datasets/tool_hang HORIZON=700 ETRIES=5 VISGATE=0 \
   SAVE0=200 SAVE=200 \
   bash $BASE/SOE_scripts_2/chain_soe.sh tool_hang 2>&1 | tee -a $LOGROOT/soe_th${SEED}_chain.console.log"
echo "launched soe_th${SEED}_chain GPU$GPU -> $ROOT (wandb TOOLHANG-9-7-SOE-s$SEED)"
