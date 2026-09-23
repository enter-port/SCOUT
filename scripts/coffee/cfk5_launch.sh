#!/bin/bash
# cfk5_launch.sh -- COFFEE-MG-p5 one-shot orchestrator (2026-09-23).
# Coffee rerun of the THM3 recipe (user order 2026-09-23: eta+kappa
# per-round recalibration, B2 DP recipe 600ep b64, A3 dyn recipe 300ep
# beta=1e-5). Runs on the port-1022 host, GPUs 0-5:
#   A) stage core20 hdf5 + DP-base ckpt per seed from COFFEE-MG-p1
#      (idempotent, size-verified, cp -L). The dyn-base is NOT copied --
#      it is RE-TRAINED with beta=1e-5 (thm2_launch pattern: the whole
#      chain must live on one beta);
#   B) round0 = dyn-base x3 seeds (beta=1e-5) in parallel via tmux
#      cfk5_base_s<seed> on GPU 0/1/2; wait for the 3 "ROUND0 coffee
#      seed=<s> TOTAL" lines, then kill those sessions;
#   C) measure BASE_ETA per seed: cfk_rcalib on the base pair
#      (eta_prev=ETA0=5.6, kappa_prev=2.5) -> base_rcalib.json (the kappa
#      C-anchor constant the ATY chains carry);
#   D) spawn the 6 chains cfk5_{dp,aty}_s<seed>, one GPU each:
#      s233 G0/G1, s2333 G2/G3, s23333 G4/G5.
#
# usage: tmux new-session -d -s cfk5_launch 'bash scripts/coffee/cfk5_launch.sh'
#        COPY_ONLY=1 bash scripts/coffee/cfk5_launch.sh   # stage only
set -u
set -o pipefail
ROOT=/root/workspace/baojiachun/scout
SRC=$ROOT/data/2026_9_20_coffee_mg_p1
DST=$ROOT/data/2026_9_23_coffee_mg_p5
SEEDS="233 2333 23333"
PY=/root/workspace/baojiachun/.venv_mg/bin/python
ETA0=5.6
KAP0=2.5
BETA=1.0e-5
ROUND0_TIMEOUT_S=$((4*3600))
cd "$ROOT" || exit 1

log(){ echo "[$(date '+%F %T')] [cfk5] $*"; }

# --- gate: GPUs 0-5 must be idle (shared-server rule) --- #
for g in 0 1 2 3 4 5; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$used" -lt 1000 ] || { log "FATAL GPU$g busy (${used}MiB)"; exit 1; }
done

# --- A) stage core20 + DP-base per seed (idempotent, size-verified) --- #
for s in $SEEDS; do
  S1=$SRC/COFFEE-MG-p1-s$s/coffee
  S2=$DST/COFFEE-MG-p5-s$s/coffee
  DPNEW=$(ls -t $S1/train/DP/DP-base/checkpoints/*.ckpt 2>/dev/null | head -1)
  [ -n "$DPNEW" ] || { log "FATAL no DP-base ckpt for s$s"; exit 1; }
  [ -f "$S1/rollout/coffee_core.hdf5" ] || { log "FATAL missing source core for s$s"; exit 1; }
  mkdir -p "$S2/rollout" "$S2/train/DP/DP-base/checkpoints" "$S2/train/dyn"
  copy_chk(){  # copy with SIZE verification (a half-written leftover re-copies)
    local srcf=$1 dstf=$2
    if [ ! -f "$dstf" ] || [ "$(stat -c%s "$srcf")" != "$(stat -c%s "$dstf")" ]; then
      cp -L "$srcf" "$dstf" || { log "FATAL copy failed: $srcf"; exit 1; }
    fi
  }
  copy_chk "$S1/rollout/coffee_core.hdf5" "$S2/rollout/coffee_core.hdf5"
  copy_chk "$DPNEW" "$S2/train/DP/DP-base/checkpoints/$(basename $DPNEW)"
  for aux in config.yaml train.log; do
    [ -f "$S1/train/DP/DP-base/$aux" ] && { [ -f "$S2/train/DP/DP-base/$aux" ] \
      || cp -L "$S1/train/DP/DP-base/$aux" "$S2/train/DP/DP-base/"; }
  done
  [ -f "$DST/COFFEE-MG-p5-s$s/round.log" ] \
    || echo "[$(date '+%F %T')] [cfk5] core + DP-base staged from COFFEE-MG-p1-s$s; dyn-base to be retrained at beta=$BETA" >> "$DST/COFFEE-MG-p5-s$s/round.log"
  log "s$s staged: core=$(stat -c%s $S2/rollout/coffee_core.hdf5)B dp=$(basename $(ls -t $S2/train/DP/DP-base/checkpoints/*.ckpt | head -1))"
done

if [ "${COPY_ONLY:-0}" = 1 ]; then
  log "COPY_ONLY=1 -- staged, not spawning"
  exit 0
fi

# --- B) round0 = dyn-base x3 with beta=1e-5 (parallel, GPU 0/1/2) --- #
g=0
for s in $SEEDS; do
  RL=$DST/COFFEE-MG-p5-s$s/coffee/round.log
  if grep -q "ROUND0 coffee seed=$s TOTAL" "$RL" 2>/dev/null; then
    log "dyn-base s$s already done -- skip"; g=$((g+1)); continue
  fi
  if tmux has-session -t "cfk5_base_s$s" 2>/dev/null; then
    log "tmux cfk5_base_s$s exists -- reuse"
  else
    tmux new-session -d -s "cfk5_base_s$s" \
      "cd $ROOT && GPU=$g TSEED=$s DATA_ROOT=$DST/COFFEE-MG-p5-s$s WPROJ=COFFEE-MG-p5-s$s BETA=$BETA bash scripts/coffee/round_cfk5.sh coffee BASE 0"
    log "dyn-base s$s launched @GPU$g (tmux cfk5_base_s$s, beta=$BETA)"
  fi
  g=$((g+1))
done
T0=$(date +%s)
while true; do
  done_n=0
  for s in $SEEDS; do
    grep -q "ROUND0 coffee seed=$s TOTAL" "$DST/COFFEE-MG-p5-s$s/coffee/round.log" 2>/dev/null && done_n=$((done_n+1))
  done
  [ "$done_n" = 3 ] && { log "all 3 dyn-base TOTAL lines present"; break; }
  [ $(( $(date +%s) - T0 )) -gt "$ROUND0_TIMEOUT_S" ] && { log "FATAL: dyn-base wait timeout (${done_n}/3 done)"; exit 1; }
  sleep 60
done
for s in $SEEDS; do tmux kill-session -t "cfk5_base_s$s" 2>/dev/null; done

# --- C) BASE_ETA per seed: cfk_rcalib on the base pair (GPU0, ~min each) --- #
for s in $SEEDS; do
  S2=$DST/COFFEE-MG-p5-s$s/coffee
  BJ=$S2/rollout/base_rcalib.json
  [ -f "$BJ" ] && { log "s$s BASE_ETA json exists -- skip"; continue; }
  env CUDA_VISIBLE_DEVICES=0 $PY scripts/coffee/cfk_rcalib.py \
    --eval-config configs/eval_coffee_entropy.yaml \
    --dp-ckpt "$(ls -t $S2/train/DP/DP-base/checkpoints/*.ckpt | head -1)" \
    --vib-ckpt "$(ls -t $S2/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)" \
    --core-hdf5 "$S2/rollout/coffee_core.hdf5" \
    --eta-prev "$ETA0" --kappa-prev "$KAP0" \
    --out "$BJ" > "$S2/rollout/base_rcalib.stdout" 2>&1 \
    || { log "FATAL: BASE_ETA rcalib failed for s$s -- see $S2/rollout/base_rcalib.stdout"; exit 1; }
  log "s$s BASE_ETA measured (json: $BJ)"
done

# --- D) spawn the 6 chains (GPU map: s233 G0/G1, s2333 G2/G3, s23333 G4/G5) --- #
g=0
for s in $SEEDS; do
  BJ=$DST/COFFEE-MG-p5-s$s/coffee/rollout/base_rcalib.json
  BE=$($PY -c "import json;print('%.6g'%json.load(open('$BJ'))['eta'])")
  [ -n "$BE" ] || { log "FATAL BASE_ETA unreadable for s$s ($BJ)"; exit 1; }
  for ARM in DP ATY; do
    tmux new-session -d -s "cfk5_${ARM,,}_s$s" \
      "cd $ROOT && SEED=$s GPU=$g ARM=$ARM NROUNDS=6 DATA_ROOT=$DST/COFFEE-MG-p5-s$s WPROJ=COFFEE-MG-p5-s$s ETA0=$ETA0 ATT_CAP=$KAP0 BETA=$BETA ETRIES=5 BASE_ETA=$BE bash scripts/coffee/cfk5_chain.sh"
    log "spawned cfk5_${ARM,,}_s$s gpu=$g BASE_ETA=$BE"
    g=$((g+1))
  done
done
log "LAUNCH_DONE $(date '+%F %T')"
