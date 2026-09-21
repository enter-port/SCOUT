#!/bin/bash
# fix_aty_s2333_dyn.sh v2 -- manual [3/3] dyn recovery, TRANSPORT-s2333 ATY r3.
# v1 died twice on server-outbound wandb.ai (18:45 init CommError, 19:26 login
# traceback -- network wave). v2 = online attempt with WANDB_INIT_TIMEOUT=300,
# automatic WANDB_MODE=offline fallback (local metrics only; cloud run
# dalayzo7 keeps its phase-A eval record -- cosmetic gap, sync optional later).
# On rc=0: patch round.log (exact formats), clean GHOST tmux tp13_aty_s2333
# (pane dead since 18:45:57 chain exit), respawn chain verbatim -> r4.
set -u
ROOT=/root/workspace/baojiachun/scout
DR=$ROOT/data/2026_9_13_transport/TRANSPORT-s2333
OUTDYN=$DR/transport/train/dyn/dyn-ATY-exp3
RL=$DR/transport/round.log
PY=/root/workspace/baojiachun/.venv/bin/python
cd "$ROOT" || exit 1

cp -n "$OUTDYN/train.log" "$OUTDYN/train.log.crash2_wandblogin" 2>/dev/null

T1=$(date +%s)
echo "[$(date '+%F %T')] FIXv2: dyn online attempt (GPU3, RID=dalayzo7, init_timeout=300)"
env CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  WANDB_RUN_ID=dalayzo7 WANDB_RESUME=must WANDB_INIT_TIMEOUT=300 \
  $PY -m scout.train_vib --config "$OUTDYN/config.yaml" \
  > "$OUTDYN/train.log" 2>&1
RC=$?
if [ $RC -ne 0 ]; then
  echo "[$(date '+%F %T')] FIXv2: online rc=$RC -- fallback WANDB_MODE=offline (local-only metrics)"
  cp -n "$OUTDYN/train.log" "$OUTDYN/train.log.crash3_online" 2>/dev/null
  env CUDA_VISIBLE_DEVICES=3 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    WANDB_MODE=offline \
    $PY -m scout.train_vib --config "$OUTDYN/config.yaml" \
    > "$OUTDYN/train.log" 2>&1
  RC=$?
fi
T2=$(date +%s); D1=$((T2-T1))
echo "[$(date '+%F %T')] FIXv2: dyn rc=$RC in $((D1/60))m$((D1%60))s"
if [ $RC -ne 0 ]; then
  echo "FIXv2 FAILED both modes -- round.log NOT patched, chain NOT respawned. See $OUTDYN/train.log"
  exit 1
fi
ls "$OUTDYN"/*/scout_vib.ckpt >/dev/null 2>&1 || {
  echo "FIXv2: scout_vib.ckpt NOT FOUND after rc=0 -- abort patch"; exit 1; }

T0=$(date -d "2026-09-14 11:13:34" +%s); TOT=$((T2-T0))
printf '[%s] [3/3] dyn retrain rc=0 in %dm%ds\n' "$(date '+%F %T')" $((D1/60)) $((D1%60)) >> "$RL"
printf '[%s] === ROUND transport a=ATY seed=2333 round=3 TOTAL: %dm%ds ===\n' "$(date '+%F %T')" $((TOT/60)) $((TOT%60)) >> "$RL"
echo "round.log patched (r3 TOTAL from original START 11:13:34, honest wall time)."

PP=$(tmux list-panes -t tp13_aty_s2333 -F "#{pane_pid}" 2>/dev/null)
if [ -z "$PP" ]; then
  tmux kill-session -t tp13_aty_s2333 2>/dev/null && echo "ghost tmux tp13_aty_s2333 cleaned (pane was already dead)"
fi
tmux new-session -d -s tp13_aty_s2333 "cd $ROOT && SEED=2333 GPU=3 ARM=ATY DATA_ROOT=$DR NROUNDS=6 CORE_N=20 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=ATY ATT_CAP=2.5 ATY_SCALE=0.5 ORB_LAM=0.5 ORB_DELTA=0.25 ORB_SIGMA=0.05 ORB_SIGMA_DECAY=0.5 ORB_ANNEAL=2 bash scripts/transport/tp13_chain.sh"
echo "chain tp13_aty_s2333 respawned (done_round skips r1-r3, enters r4)."
