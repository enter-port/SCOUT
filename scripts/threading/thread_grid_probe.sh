#!/bin/bash
# thread_grid_probe.sh -- ONE grid-search cell: (ARM, ETA) on the s233 base
# (2026-09-16; the ONLY scanned parameter = shared raw --guidance-scale, both
# guided arms take the same dose at a cell; orbit's sigma/lam/delta/anneal
# stay at the round_threading.sh th95 defaults, NOT scanned).
#
# Cell = MODE=eval-pk round 1 (full two-phase rescue rollout, retrains
# skipped): phase A eval 100 fixed scenes (unguided first-try SR + freeze
# failed set) -> phase B sharded rescue xETRIES=5 on failed scenes -> merged
# json pass_at_5 = pass@5. Criterion = pass@5 (user 2026-09-16); telemetry
# (mean_inject/jerk) is recorded but NOT the criterion.
#
# The base three-piece (core hdf5 + DP-base + dyn-base) is SYMLINKED
# read-only from the s233 chain root; probe writes land under
# data/2026_9_16_threading_p1/GRID_s233/<CELL>/ and are throwaway.
# Hard 60min cap via timeout (user order 2026-09-17: grid cap 放宽到1小时,
# 30min cap 实测不够 phase-A eval + phase-B 265 条 guided rescue;链上协议
# 不受影响); rc=124 -> TIMEOUT row, no verdict.
#
# usage: ARM=<ATY|ORBIT> ETA=<float> GPU=<id> [CELL=<name>] \
#          bash scripts/threading/thread_grid_probe.sh
set -u
ARM=${ARM:?set ARM=<ATY|ORBIT>}
ETA=${ETA:?set ETA=<raw guidance scale>}
GPU=${GPU:?set GPU=<cuda id>}
CELL=${CELL:-${ARM}_eta${ETA}}
SEED=233
ETRIES=5
WNAME_BASE=${WNAME_BASE:-${ARM}_eta${ETA}}   # distinct wandb names per cell
ROOT=/root/workspace/baojiachun/scout
BASE_ROOT=$ROOT/data/2026_9_16_threading_p1/THREADING-MG-p1-s233
GRIDROOT=$ROOT/data/2026_9_16_threading_p1/GRID_s233
CELLDIR=$GRIDROOT/$CELL
CSV=$GRIDROOT/grid_results.csv

[ -f "$BASE_ROOT/threading/rollout/threading_core.hdf5" ] || { echo "[grid] FATAL: s233 core missing"; exit 1; }
ls "$BASE_ROOT"/threading/train/DP/DP-base/checkpoints/*.ckpt >/dev/null 2>&1 || { echo "[grid] FATAL: s233 DP-base missing"; exit 1; }
ls "$BASE_ROOT"/threading/train/dyn/dyn-base/*/scout_vib.ckpt >/dev/null 2>&1 || { echo "[grid] FATAL: s233 dyn-base missing"; exit 1; }

mkdir -p "$CELLDIR/threading/rollout" "$CELLDIR/threading/train/DP" "$CELLDIR/threading/train/dyn"
ln -sfn "$BASE_ROOT/threading/rollout/threading_core.hdf5" "$CELLDIR/threading/rollout/threading_core.hdf5"
ln -sfn "$BASE_ROOT/threading/train/DP/DP-base" "$CELLDIR/threading/train/DP/DP-base"
ln -sfn "$BASE_ROOT/threading/train/dyn/dyn-base" "$CELLDIR/threading/train/dyn/dyn-base"

cd "$ROOT" || exit 1
T0=$(date +%s)
echo "[grid] cell=$CELL arm=$ARM eta=$ETA gpu=$GPU start $(date '+%F %T')"
timeout -k 30 3600 env GPU=$GPU TSEED=$SEED DATA_ROOT=$CELLDIR WPROJ=THREADING-MG-GRID \
  ETRIES=$ETRIES SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 CORE_N=20 WNAME_BASE=$ARM \
  ATT_CAP=2.5 ATY_SCALE=$ETA \
  bash scripts/threading/round_threading.sh threading "$ARM" 1 eval-pk
rc=$?
T1=$(date +%s)
echo "[grid] cell=$CELL rc=$rc in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"

# ---- readout -> csv (pass@5 criterion; SR + rescued recorded alongside) --- #
[ -f "$CSV" ] || echo "cell,arm,eta,wall_min,rc,eval_sr,rescued,pass5" > "$CSV"
/root/workspace/baojiachun/.venv_mg/bin/python - "$CELL" "$ARM" "$ETA" "$rc" "$(( (T1-T0)/60 ))" "$CSV" <<'PYEOF'
import sys, os, json, glob
cell, arm, eta, rc, wall, csv = sys.argv[1:8]
rdir = os.path.join(os.path.dirname(csv), cell, "threading", "rollout", f"{arm}-exp1")
sr = rescued = pass5 = ""
rj = glob.glob(os.path.join(rdir, "log", "*_rollout_exp1.json")) or \
     glob.glob(os.path.join(rdir, "log", "*_SCOUT_rollout_exp1.json"))
if rj:
    d = json.load(open(sorted(rj)[-1]))
    sr = d.get("success_rate", "")
ej = os.path.join(rdir, "log", f"threading_{arm}_explore_exp1.json")
if os.path.isfile(ej):
    d = json.load(open(ej))
    rescued = d.get("exploration_rescued", "")
    pass5 = d.get("pass_at_5", "")
line = f"{cell},{arm},{eta},{wall},{rc},{sr},{rescued},{pass5}"
with open(csv, "a") as f:
    f.write(line + "\n")
print(f"[grid-readout] {line}")
PYEOF
