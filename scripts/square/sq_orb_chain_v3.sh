#!/bin/bash
# SQUARE s233 orbit-native 6-round chain V3 (user order 2026-09-02 late):
# same spec as v2 (rounds 1-5 full, round 6 EVAL-ONLY, sharded worker code)
# but on the MERGED orbit-dev code @e110ffc (PR #2 hparam fixes + gate/flock
# removal) and with the FINAL cross-task hparam group wired in round_orbit.sh:
#   kappa 2.5 / lam 0.5 / delta 0.25 unchanged; sigma 0.16 x 0.5^(round-1),
#   eta_tilde 0.33 (dimless), fb-clamp soft, noise-anneal 2.
# v2 history: launched 09-02 19:38 on pre-merge code with sigma 0.25 and no
# hparam flags -- user killed it (misalignment); its partial r1 products were
# removed (round.log archived as round.log.v2dead) and its wandb runs deleted.
# v1 monolithic products remain untouched under scout-rand/data/2026_9_1_orbchain/.
# Usage: GPU=<id> bash /tmp/sq_orb_chain_v3.sh
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
TSEED=233
ROOT=/root/workspace/baojiachun/scout-orbit
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
DATA_ROOT=$ROOT/data/2026_9_1_orbchain/ORBIT-s233
T=$DATA_ROOT/square
WPROJ=SQUARE-9-1-orbit-s233

mkdir -p "$T/rollout" "$T/train/DP/DP-base/checkpoints" "$T/train/dyn/dyn-base"
[ -f "$T/rollout/square_core.hdf5" ] || cp "$R26/core_rebuild/square_core_s233.hdf5" "$T/rollout/square_core.hdf5"
ln -sf "$R26/SQUARE-entropy-s233/square/train/DP/DP-base/checkpoints/599.ckpt" \
      "$T/train/DP/DP-base/checkpoints/599.ckpt"
ln -sfn "$R26/SQUARE-entropy-s233/square/train/dyn/dyn-base/20260826-112119" \
       "$T/train/dyn/dyn-base/20260826-112119"

cd "$ROOT" || exit 1
for N in 1 2 3 4 5 6; do
  MODE=full; [ "$N" = 6 ] && MODE=eval-only
  echo "[chain] round $N ($MODE) START $(date '+%F %T')"
  GPU=$GPU TSEED=$TSEED DATA_ROOT=$DATA_ROOT WPROJ=$WPROJ ATT_CAP=2.5 \
    bash soe_scripts/round_orbit.sh square SCOUT "$N" "$MODE"
  rc=$?
  echo "[chain] round $N rc=$rc $(date '+%F %T')"
  [ $rc -ne 0 ] && { echo "[chain] ABORT at round $N (see $T/round.log)"; exit $rc; }
done
echo "[chain] ALL 6 ROUNDS DONE (r6 eval-only) $(date '+%F %T')"
