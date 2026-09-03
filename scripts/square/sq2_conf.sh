#!/bin/bash
# SQUARE beat-SOE mixing CONFIRMATION driver (2026-08-31). Launches one
# process per arm at its allocated try budget; pooled per-scene pass@10
# across arms = the stratified-exploration measurement (10 tries/scene total).
# Fresh explore seed 43 (wave-1 used 42 -- same RNG stream would be optimistically
# biased by the split search). Eval seed stays 42 (scene set unchanged).
#
# Env: SPLIT="orb025:6,att:3,plc:1"   (arm:k pairs, k>=1, sum<=10)
#      SEED=43 GATE=1|0 GPU=g bash /tmp/sq2_conf.sh <233|2333|23333>
# SEED = retry-RNG seed, plumbed to --rescue-seed (fresh-seed confirmation;
#       2026-08-31: --seed used to collide with the failed-set base_seed
#       guard -> scene set is now hard-wired to 42 and only the retry RNG
#       varies). SEED=42 -> bit-identical historical stream.
# GATE=1 uses the first-20-failures subset as the failed set (pass@20 gate);
# GATE=0 the full frozen failed set (pass@10 measurement).
set -uo pipefail
: "${SPLIT:?set SPLIT=arm:k,arm:k...}"
RSEED=${SEED:-43}; GATE=${GATE:-1}
export TMPDIR=/tmp
cd /root/workspace/baojiachun/scout-rand
PY=/root/workspace/baojiachun/.venv/bin/python
R26=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy
CORE=$R26/core_rebuild
S=${1:?seed}
OUTROOT=data/particle/sq2_conf_s${S}_seed${RSEED}_gate${GATE}
mkdir -p "$OUTROOT"
dp=$R26/SQUARE-entropy-s$S/square/train/DP/DP-base/checkpoints/599.ckpt
case $S in
  233)   ts=20260826-112119 ;;
  2333)  ts=20260826-112147 ;;
  23333) ts=20260829-025739 ;;
esac
vib=$R26/SQUARE-entropy-s$S/square/train/dyn/dyn-base/$ts/scout_vib.ckpt

arm_args () {  # $1=arm -> echoes extra argv
  case $1 in
    plc)      echo "--guide off" ;;
    att)      echo "--guide atypical --atypical-cap 2.5 --vib-ckpt $vib" ;;
    par)      echo "--guide particle --atypical-cap 2.5 --pg-lambda 0.25 --pg-start 0 --vib-ckpt $vib" ;;
    orb025)   echo "--guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 --vib-ckpt $vib" ;;
    orb015)   echo "--guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.15 --vib-ckpt $vib" ;;
    orbk15)   echo "--guide orbit --atypical-cap 1.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 --vib-ckpt $vib" ;;
    orbk40)   echo "--guide orbit --atypical-cap 4.0 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 --vib-ckpt $vib" ;;
    orbdet)   echo "--guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 --orbit-sector det --orbit-sector-seed $RSEED --vib-ckpt $vib" ;;
    ray)      echo "--guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 --orbit-climb ray --orbit-ray-seed $RSEED --vib-ckpt $vib" ;;
    ray0)     echo "--guide orbit --atypical-cap 2.5 --orbit-lam 0.0 --orbit-delta 0.25 --orbit-sigma 0.0 --orbit-climb ray --orbit-ray-seed $RSEED --vib-ckpt $vib" ;;
    *) echo "UNKNOWN_ARM_$1"; return 1 ;;
  esac
}

IFS=',' read -ra PAIRS <<< "$SPLIT"
pids=(); arms_=()
for pr in "${PAIRS[@]}"; do
  arm=${pr%%:*}; k=${pr##*:}
  out=$OUTROOT/$arm
  mkdir -p "$out/log"
  # failed set: full frozen set, or its first-20-failure subset for the gate
  "$PY" - "$S" "$out" "$GATE" << 'PYEOF'
import json, sys
S, out, gate = sys.argv[1], sys.argv[2], int(sys.argv[3])
src = {"233": "sq2_orb025_s233", "2333": "sq2_plc_s2333",
       "23333": "sq2_plc_s23333"}[S]
spec = json.load(open(f"data/particle/{src}/failed_set.json"))
if gate:
    keep = [i for i in spec["failed_init_indices"] if i < 20]
    spec = dict(spec); spec["failed_init_indices"] = keep
json.dump(spec, open(f"{out}/failed_set.json", "w"), indent=1)
print(f"[conf] {out}: {len(spec['failed_init_indices'])} failed inits")
PYEOF
  args=$(arm_args "$arm") || { echo "bad arm $arm"; exit 2; }
  echo "[$(date +%m-%d\ %H:%M:%S)] CONF $S rseed=$RSEED gate=$GATE arm=$arm tries=$k"
  if [ -n "${DRYRUN:-}" ]; then echo "DRYRUN: out=$out args=$args k=$k"; continue; fi
  # --rescue-seed only when != 42: explicit 42 re-seeds at phase-2 start,
  # which is NOT bit-identical to the historical evolved-42 stream.
  rs_args=""
  [ "$RSEED" -ne 42 ] && rs_args="--rescue-seed $RSEED"
  env CUDA_VISIBLE_DEVICES=$GPU SCOUT_RENDER_GPU=$GPU "$PY" -u -m scout.eval.run_rollout \
    --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
    --base-dp-ckpt "$dp" --core-hdf5 "$CORE/square_core_s$S.hdf5" \
    --try-times "$k" --explore-try-times "$k" \
    --seed 42 $rs_args --explore-mode rescue \
    --failed-set-json "$out/failed_set.json" --save-failed-set "$out/failed_set.json" \
    --n-envs 50 --no-wandb --output-dir "$out" $args \
    >> "$OUTROOT/$arm.log" 2>&1 &
  pids+=($!); arms_+=("$arm")
  echo "  pid=$! ($arm x$k)"
done
rc_all=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}"; rc=$?
  arm=${arms_[$i]}
  j="$OUTROOT/$arm/log/square_SCOUT_rollout_exp0.json"
  if [ "$rc" -ne 0 ] || [ ! -f "$j" ]; then
    echo "[$(date +%m-%d\ %H:%M:%S)] CONF ARM_FAIL $S arm=$arm rc=$rc json_present=$([ -f "$j" ] && echo yes || echo no)"
    rc_all=1
  else
    echo "[$(date +%m-%d\ %H:%M:%S)] CONF ARM_OK $S arm=$arm rc=0"
  fi
done
echo "[$(date +%m-%d\ %H:%M:%S)] CONF $S rseed=$RSEED gate=$GATE SPLIT=$SPLIT all arms done rc_all=$rc_all"
exit $rc_all
