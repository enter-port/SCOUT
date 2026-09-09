#!/bin/bash
# th95_s23333_launch.sh -- TOOLHANG-9-5-orbit-s23333 orchestrator (2026-09-08
# user order: new GPU batch reached via ssh port 1024 on the same host -- a
# DIFFERENT pod (dsw-801389-...) mounting the SAME CPFS (/root/workspace ->
# /mnt/workspace, inode-verified), 8x H20 all idle, EGL render smoke PASSED
# (mujoco.Renderer 84x84 on GPU2; exit-time GLContext.__del__ noise only).
# Spec = TOOLHANG-9-5-orbit-s233 verbatim except seed 23333:
#   DP(guide off)@GPU0 / ATY(atypical raw s0.5/k2.5/gst50)@GPU2 /
#   ORBIT(orbit raw s0.5 + lam.5/delta.25/sig.05 x0.5^(r-1)/anneal2/fb-soft)@GPU4
#   ETRIES=5, SHARD_P=8/arm x 25 env, 6 rounds, r6=eval-pk (SR + pass@5).
# Like s2333 (no seed-23333 assets anywhere): REAL round0 first (split 40/200
# rng 23333 + base DP 600ep + dyn-base, ~3.2h, GPU5), arms auto-spawn on rc=0.
# Runs INSIDE tmux th95_r0_23333 (never via blocking ssh). Kick-off:
#   tmux new-session -d -s th95_r0_23333 'bash .../th95_s23333_launch.sh'
# Review P1-1 pattern kept: dose knobs unset from operator shell AND pinned
# inside each tmux command string. GPU7 left untouched (ECC=0 here, but the
# red line stays unambiguous and the cards are plentiful).
set -u
ROOT=/root/workspace/baojiachun/scout-th95
DR=$ROOT/data/2026_9_5_toolhang/TOOLHANG-s23333
cd "$ROOT" || exit 1
unset DATA_ROOT ATT_CAP ATY_SCALE ORB_LAM ORB_DELTA ORB_SIGMA ORB_SIGMA_DECAY \
      ORB_ANNEAL DP_EPOCHS_SOE DYN_EPOCHS_SOE XMODE WNAME_BASE
mkdir -p "$DR"

echo "[launch] round0 START (REAL: split rng 23333 + DP 600ep + dyn-base, GPU5) $(date '+%F %T')"
SEED=23333 GPU=5 ARM=BASE DATA_ROOT=$DR bash scripts/toolhang/th95_chain.sh
rc=$?
echo "[launch] round0 rc=$rc $(date '+%F %T')"
if [ $rc -ne 0 ]; then
  echo "[launch] round0 FAILED -- arms NOT started (see $DR/chain_BASE.console.log)"
  exit 1
fi

for spec in "th95_dp_23333:0:DP" "th95_aty_23333:2:ATY" "th95_orbit_23333:4:ORBIT"; do
  S=${spec%%:*}; rest=${spec#*:}; G=${rest%%:*}; A=${rest##*:}
  if tmux has-session -t "$S" 2>/dev/null; then
    echo "[launch] tmux $S already exists -- SKIP (not idempotent-safe to double-spawn)"
    continue
  fi
  tmux new-session -d -s "$S" \
    "cd $ROOT && SEED=23333 GPU=$G ARM=$A DATA_ROOT=$DR NROUNDS=6 SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 DYN_FREEZE_AFTER=6 ETRIES=5 FLUSH_EVERY=100 WNAME_BASE=$A ATT_CAP=2.5 ATY_SCALE=0.5 ORB_LAM=0.5 ORB_DELTA=0.25 ORB_SIGMA=0.05 ORB_SIGMA_DECAY=0.5 ORB_ANNEAL=2 bash scripts/toolhang/th95_chain.sh"
done
echo "[launch] arms spawned: th95_dp_23333@GPU0 th95_aty_23333@GPU2 th95_orbit_23333@GPU4 $(date '+%F %T')"
