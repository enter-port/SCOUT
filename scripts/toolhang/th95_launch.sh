#!/bin/bash
# th95_launch.sh -- TOOLHANG-9-5-orbit-s233 launch orchestrator (2026-09-05
# user order). Three arms, SIMULTANEOUS launch (user-picked despite the RAM
# note: 3x8=24 heavy explore workers > the 16-concurrent OOM incident):
#   DP    guide off                        @GPU0
#   ATY   atypical raw s0.5/k2.5/gst50     @GPU2
#   ORBIT orbit raw s0.5 + lam.5/delta.25/sig.05x0.5^(r-1)/anneal2/fb-soft @GPU4
# 6 rounds (r1-5 full + r6 eval-only), 8 shard workers x 25 envs per arm,
# ETRIES=5 (pass@5), --flush-every 100 (TrajSpool OOM fix).
# Round0 assets (40-demo core + DP-base 599.ckpt + dyn-base 20260904-170403)
# are COPIED from the TOOLHANG-9-4-orbit-s233 root (same seed 233 ->
# training-identical weights, skips a redundant 600ep base retrain); the
# launch script itself runs the BASE arm only as an idempotent round0 rc
# check (instant skip: all three assets already present).
# wandb project TOOLHANG-9-5-orbit-s233, run names DP-round{i}/ATY-round{i}/
# ORBIT-round{i}.
set -u
ROOT=/root/workspace/baojiachun/scout-th95
DR=$ROOT/data/2026_9_5_toolhang/TOOLHANG-s233
SRC=/root/workspace/baojiachun/scout-th94/data/2026_9_4_toolhang/TOOLHANG-s233/tool_hang
DST=$DR/tool_hang
cd "$ROOT" || exit 1
# review P1-1/P2-2: stale operator-shell exports must not bend this campaign --
# unset the redirect AND every dose knob the round driver reads from env, then
# pin all of them explicitly inside each tmux command string below.
unset DATA_ROOT ATT_CAP ATY_SCALE ORB_LAM ORB_DELTA ORB_SIGMA ORB_SIGMA_DECAY \
      ORB_ANNEAL DP_EPOCHS_SOE DYN_EPOCHS_SOE XMODE WNAME_BASE

# ---- pre-copy round0 assets from TOOLHANG-9-4 (idempotent, cp -L for
# ---- symlinked ckpts; abort BEFORE spawning arms if the source is broken)
for f in "$SRC/rollout/tool_hang_core.hdf5" \
         "$SRC/train/DP/DP-base/checkpoints/599.ckpt" ; do
  [ -f "$f" ] || { echo "[launch] FATAL: source asset missing: $f"; exit 1; }
done
VIBTS=$(ls "$SRC/train/dyn/dyn-base/" | grep -E '^[0-9]{8}-[0-9]{6}$' | head -1)
[ -n "$VIBTS" ] && [ -f "$SRC/train/dyn/dyn-base/$VIBTS/scout_vib.ckpt" ] \
  || { echo "[launch] FATAL: source dyn-base scout_vib.ckpt missing"; exit 1; }

mkdir -p "$DST/rollout" "$DST/train/DP/DP-base/checkpoints" "$DST/train/dyn/dyn-base/$VIBTS"
[ -f "$DST/rollout/tool_hang_core.hdf5" ] || cp "$SRC/rollout/tool_hang_core.hdf5" "$DST/rollout/"
[ -f "$DST/train/DP/DP-base/checkpoints/599.ckpt" ] || cp -L "$SRC/train/DP/DP-base/checkpoints/599.ckpt" "$DST/train/DP/DP-base/checkpoints/599.ckpt"
for f in scout_vib.ckpt config.yaml; do
  [ -f "$DST/train/dyn/dyn-base/$VIBTS/$f" ] || cp "$SRC/train/dyn/dyn-base/$VIBTS/$f" "$DST/train/dyn/dyn-base/$VIBTS/"
done
echo "[launch] assets ready: core=$(du -h "$DST/rollout/tool_hang_core.hdf5" | cut -f1) dp599=$(du -h "$DST/train/DP/DP-base/checkpoints/599.ckpt" | cut -f1) vib=$VIBTS $(date '+%F %T')"

# ---- round0 rc check (asset-copy round0 must instant-skip; non-zero rc or a
# ---- missing TOTAL line means the assets are NOT complete -> no arms)
echo "[launch] round0 START $(date '+%F %T') (GPU0, seed 233)"
SEED=233 GPU=0 ARM=BASE bash scripts/toolhang/th95_chain.sh
rc=$?
echo "[launch] round0 rc=$rc $(date '+%F %T')"
if [ $rc -ne 0 ]; then
  echo "[launch] round0 FAILED -- arms NOT started"
  exit 1
fi

# ---- three arms, simultaneous (user order 2026-09-05) ---- #
for spec in "th95_dp_233:0:DP" "th95_aty_233:2:ATY" "th95_orbit_233:4:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[launch] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=233 GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=2.5 ATY_SCALE=0.5 ORB_LAM=0.5 ORB_DELTA=0.25 ORB_SIGMA=0.05 ORB_SIGMA_DECAY=0.5 ORB_ANNEAL=2 bash scripts/toolhang/th95_chain.sh"
done
echo "[launch] arms spawned: th95_dp_233@GPU0 th95_aty_233@GPU2 th95_orbit_233@GPU4 $(date '+%F %T')"
