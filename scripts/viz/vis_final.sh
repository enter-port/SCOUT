#!/bin/bash
# vis_final.sh -- final-policy visualization rollouts (user 2026-08-20). v2
#
# For each experiment exp1..exp6 (USER numbering, see spec()) roll the chain's
# FINAL checkpoints for 20 episodes on the seed-42 scene set
# (inits = env.reset(seed=42+i), i=0..19 -- IDENTICAL scenes across every
# run / group / experiment) and write an all.hdf5 (core 20 + 20 rollouts,
# obs+states recorded) that scout.eval.visualize_trajectories renders.
#
#   g1_guide_on  : SCOUT-arm final DP + its final VIB, guided (the
#                  experiment's own guidance_scale, from its eval config)
#   g1_guide_off : the same SCOUT-arm final DP, plain (no guidance);
#                  doubles as the group-2 "SCOUT" panel (g2_SCOUT symlink)
#   g2_DP        : the DP baseline arm's final DP, plain (round-matched
#                  to the SCOUT arm's last COMPLETE round)
#
# v2 (2026-08-20 late): offscreen-render corruption observed at high
# concurrent-env counts under machine load (camera frames degenerate to a
# constant garbage / frozen frame -- policy blind, rollouts hover; see
# experiments/experiment_log.md). This driver therefore
#   * rolls at NENVS=4 by default (NENVS_TRIES="4 2": retry at 2 on failure);
#   * VALIDATES every finished all.hdf5 with soe_scripts/vis_validate.py
#     (per-demo frame-mean diversity) and auto-retries corrupt runs;
#   * skips a key only if all.hdf5 exists AND passes validation.
#
# Usage:  GPU=<id> bash soe_scripts/vis_final.sh <key> [<key>...]  (sequential)
# Keys:   exp{1..6}_{on,off,dp}.  exp4_dp is NOT a run: e3's DP baseline is
#         shared with exp3 -> symlinked instead.
set -u
GPU=${GPU:?set GPU=<cuda id>}
NENVS_TRIES=${NENVS_TRIES:-"4 2"}
export MUJOCO_GL=egl TMPDIR=/tmp

REPO=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv/bin/python
cd "$REPO" || exit 1

newest_ckpt(){ ls -t "$1"/checkpoints/*.ckpt 2>/dev/null | head -1; }
newest_vib(){  ls -t "$1"/*/scout_vib.ckpt  2>/dev/null | head -1; }
E2=$REPO/data/experiment2/can
E3=$REPO/data/experiment3/can
E4=$REPO/data/experiment4/can
E5=$REPO/data/experiment5/can
OUTROOT=$REPO/data/vis_final
NINIT=20            # scenes = seeds 42..61 (seed 42 + i) -- shared everywhere
NEXPLORE=20

spec(){ # spec <key> -> "CFG|COREDIR|DPDIR|VIBDIR" ('-' VIBDIR = unguided)
  case "$1" in
    # experiment2: SCOUT arm stopped after round 5 (round-6 driver crash --
    # rollout data written but no retrain) -> last COMPLETE round = exp5.
    # SCOUT01 + DP arms completed round 6 -> exp6.
    # exp1 uses ROUND-4 ckpts (user 2026-08-21): dyn-SCOUT-exp5's guidance
    # gradient blew up (|dNLL/da| 9->6->10->39->109 across dyn-exp1..5; real
    # round-6 rollout with the exp5 pair = worst jerk 0.487 of the chain) --
    # verified noisy, fall back one round. DP arm round-matched to exp4.
    exp1_on)  echo "configs/eval_can_e2.yaml|$E2|$E2/train/DP/DP-SCOUT-exp4|$E2/train/dyn/dyn-SCOUT-exp4";;
    exp1_off) echo "configs/eval_can_e2.yaml|$E2|$E2/train/DP/DP-SCOUT-exp4|-";;
    exp1_dp)  echo "configs/eval_can_e2.yaml|$E2|$E2/train/DP/DP-DP-exp4|-";;
    # exp1 round-3 variant (user 2026-08-21): |dNLL/da|=10 (healthy range).
    exp1r3_on)  echo "configs/eval_can_e2.yaml|$E2|$E2/train/DP/DP-SCOUT-exp3|$E2/train/dyn/dyn-SCOUT-exp3";;
    exp1r3_off) echo "configs/eval_can_e2.yaml|$E2|$E2/train/DP/DP-SCOUT-exp3|-";;
    exp1r3_dp)  echo "configs/eval_can_e2.yaml|$E2|$E2/train/DP/DP-DP-exp3|-";;
    exp2_on)  echo "configs/eval_can_e2s01.yaml|$E2|$E2/train/DP/DP-SCOUT01-exp6|$E2/train/dyn/dyn-SCOUT01-exp6";;
    exp2_off) echo "configs/eval_can_e2s01.yaml|$E2|$E2/train/DP/DP-SCOUT01-exp6|-";;
    exp2_dp)  echo "configs/eval_can_e2s01.yaml|$E2|$E2/train/DP/DP-DP-exp6|-";;
    # experiment3: all three arms completed round 5 -> exp5 everywhere.
    exp3_on)  echo "configs/eval_can_e3.yaml|$E3|$E3/train/DP/DP-SCOUT-exp5|$E3/train/dyn/dyn-SCOUT-exp5";;
    exp3_off) echo "configs/eval_can_e3.yaml|$E3|$E3/train/DP/DP-SCOUT-exp5|-";;
    exp3_dp)  echo "configs/eval_can_e3.yaml|$E3|$E3/train/DP/DP-DP-exp5|-";;
    exp4_on)  echo "configs/eval_can_e3s01.yaml|$E3|$E3/train/DP/DP-SCOUT01-exp5|$E3/train/dyn/dyn-SCOUT01-exp5";;
    exp4_off) echo "configs/eval_can_e3s01.yaml|$E3|$E3/train/DP/DP-SCOUT01-exp5|-";;
    # experiment4 = e4 (guide 0.5, 10 explore scenes/round), rounds 1-5 full,
    # round 6 eval-only -> last trained = exp5.
    exp5_on)  echo "configs/eval_can_e4.yaml|$E4|$E4/train/DP/DP-SCOUT-exp5|$E4/train/dyn/dyn-SCOUT-exp5";;
    exp5_off) echo "configs/eval_can_e4.yaml|$E4|$E4/train/DP/DP-SCOUT-exp5|-";;
    exp5_dp)  echo "configs/eval_can_e4.yaml|$E4|$E4/train/DP/DP-DP-exp5|-";;
    # experiment5 = e5 (guide 0.5, 50 explore scenes/round), same shape as e4.
    exp6_on)  echo "configs/eval_can_e5.yaml|$E5|$E5/train/DP/DP-SCOUT-exp5|$E5/train/dyn/dyn-SCOUT-exp5";;
    exp6_off) echo "configs/eval_can_e5.yaml|$E5|$E5/train/DP/DP-SCOUT-exp5|-";;
    exp6_dp)  echo "configs/eval_can_e5.yaml|$E5|$E5/train/DP/DP-DP-exp5|-";;
    *) echo "unknown key $1" >&2; return 1;;
  esac
}

outdir(){ # outdir <key> -> $OUTROOT/<expdir>/<group>
  local d g
  case "$1" in
    exp1_*) d=exp1_e2-SCOUT05;;  exp2_*) d=exp2_e2-SCOUT01;;
    exp3_*) d=exp3_e3-SCOUT05;;  exp4_*) d=exp4_e3-SCOUT01;;
    exp5_*) d=exp5_e4-SCOUT05;;  exp6_*) d=exp6_e5-SCOUT05;;
    exp1r3_*) d=exp1_e2-SCOUT05-round3;;
  esac
  case "$1" in *_on) g=g1_guide_on;; *_off) g=g1_guide_off;; *_dp) g=g2_DP;; esac
  echo "$OUTROOT/$d/$g"
}

launch_run(){ # launch_run <key> <dir> <cfg> <core> <dp> <vib> <nenvs> -> rc
  local KEY=$1 DIR=$2 CFG=$3 CORE=$4 DP=$5 VIB=$6 NE=$7
  local GARGS=(--guide off)
  [ "$VIB" != "-" ] && GARGS=(--guide dyn --vib-ckpt "$VIB")
  echo "[$KEY] run n_envs=$NE GPU$GPU dp=${DP##*/} $(date '+%F %T')"
  env CUDA_VISIBLE_DEVICES=$GPU MUJOCO_GL=egl TMPDIR=/tmp "$PY" -m scout.eval.run_rollout \
    --config "$CFG" --task can \
    --base-dp-ckpt "$DP" \
    --core-hdf5 "$CORE" \
    "${GARGS[@]}" \
    --eval-seed 42 --explore-seed 42 \
    --n-explore $NEXPLORE --explore-try-times 1 \
    --n-init-states $NINIT --n-envs "$NE" \
    --exp-num 0 \
    --output-dir "$DIR" \
    --output-success "$DIR/success.hdf5" \
    --output-all "$DIR/all.hdf5" \
    --no-wandb \
    > "$DIR/rollout.stdout" 2>&1
  return $?
}

for KEY in "$@"; do
  LINE=$(spec "$KEY") || exit 1
  IFS='|' read -r CFG COREDIR DPDIR VIBDIR <<< "$LINE"
  DP=$(newest_ckpt "$DPDIR")
  [ -n "$DP" ] || { echo "[$KEY] FATAL: no ckpt under $DPDIR"; exit 1; }
  VIB="-"
  if [ "$VIBDIR" != "-" ]; then
    VIB=$(newest_vib "$VIBDIR")
    [ -n "$VIB" ] || { echo "[$KEY] FATAL: no scout_vib.ckpt under $VIBDIR"; exit 1; }
  fi
  CORE=$COREDIR/rollout/can_core.hdf5
  [ -f "$CORE" ] || { echo "[$KEY] FATAL: missing core $CORE"; exit 1; }
  DIR=$(outdir "$KEY"); mkdir -p "$DIR"

  # skip only if the existing output passes image validation
  if [ -f "$DIR/all.hdf5" ] && \
     "$PY" soe_scripts/vis_validate.py "$DIR/all.hdf5" 20 >/dev/null 2>&1; then
    echo "[$KEY] skip (all.hdf5 exists + validation OK)"
    continue
  fi

  OK=0
  for NE in $NENVS_TRIES; do
    rm -f "$DIR/all.hdf5" "$DIR/success.hdf5"
    rm -rf "$DIR/log"
    launch_run "$KEY" "$DIR" "$CFG" "$CORE" "$DP" "$VIB" "$NE"
    RC=$?
    if [ $RC -ne 0 ]; then
      echo "[$KEY] n_envs=$NE rc=$RC -- tail:"
      tail -n 8 "$DIR/rollout.stdout"
      continue
    fi
    if "$PY" soe_scripts/vis_validate.py "$DIR/all.hdf5" 20 > "$DIR/validate.log" 2>&1; then
      echo "[$KEY] n_envs=$NE OK (validation passed) $(date '+%F %T')"
      OK=1; break
    fi
    echo "[$KEY] n_envs=$NE RENDER CORRUPTION -- validator says:"
    grep CORRUPT "$DIR/validate.log" | head -3
  done
  [ $OK -eq 1 ] || { echo "[$KEY] FAILED after all n_envs tries"; exit 1; }
done
echo "[vis_final] all keys done $(date '+%F %T')"
