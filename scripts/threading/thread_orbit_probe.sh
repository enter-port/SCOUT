#!/bin/bash
# thread_orbit_probe.sh -- ONE cell of the ORBIT parameter grid (user order
# 2026-09-17: after ATY calibration (eta=3, kappa=10, pass@5 .70), sweep
# orbit's OWN parameters at the SAME shared eta/kappa; acceptance = beat
# ATY's max). Same lineage as thread_kgrid_probe.sh with these deltas ONLY:
#   * ARM fixed to ORBIT; ETA/KAP pinned per-cell by env but typically 3/10;
#   * SIG and LAM are REQUIRED parameters -> ORB_SIGMA / ORB_LAM;
#   * CELL default / WNAME_BASE include sigma and lam;
#   * csv = GRID_s233/ogrid_results.csv with sigma/lam columns.
# Everything else identical: MODE=eval-pk round1 on the s233 base (symlinked
# read-only), pass@5 criterion, NO hard timeout (user order 2026-09-17).
#
# usage: ETA=<f> KAP=<f> SIG=<f> LAM=<f> GPU=<id> [CELL=<name>] \
#          bash scripts/threading/thread_orbit_probe.sh
set -u
ARM=ORBIT
ETA=${ETA:?set ETA}
KAP=${KAP:?set KAP}
SIG=${SIG:?set SIG}
LAM=${LAM:?set LAM}
GPU=${GPU:?set GPU}
CELL=${CELL:-${ARM}_s${SIG}_l${LAM}}
SEED=233
ETRIES=5
WNAME_BASE=${WNAME_BASE:-${ARM}_s${SIG}_l${LAM}}
ROOT=/root/workspace/baojiachun/scout
BASE_ROOT=$ROOT/data/2026_9_16_threading_p1/THREADING-MG-p1-s233
GRIDROOT=$ROOT/data/2026_9_16_threading_p1/GRID_s233
CELLDIR=$GRIDROOT/$CELL
CSV=$GRIDROOT/ogrid_results.csv

[ -f "$BASE_ROOT/threading/rollout/threading_core.hdf5" ] || { echo "[ogrid] FATAL: s233 core missing"; exit 1; }
ls "$BASE_ROOT"/threading/train/DP/DP-base/checkpoints/*.ckpt >/dev/null 2>&1 || { echo "[ogrid] FATAL: s233 DP-base missing"; exit 1; }
ls "$BASE_ROOT"/threading/train/dyn/dyn-base/*/scout_vib.ckpt >/dev/null 2>&1 || { echo "[ogrid] FATAL: s233 dyn-base missing"; exit 1; }

mkdir -p "$CELLDIR/threading/rollout" "$CELLDIR/threading/train/DP" "$CELLDIR/threading/train/dyn"
ln -sfn "$BASE_ROOT/threading/rollout/threading_core.hdf5" "$CELLDIR/threading/rollout/threading_core.hdf5"
ln -sfn "$BASE_ROOT/threading/train/DP/DP-base" "$CELLDIR/threading/train/DP/DP-base"
ln -sfn "$BASE_ROOT/threading/train/dyn/dyn-base" "$CELLDIR/threading/train/dyn/dyn-base"

cd "$ROOT" || exit 1
T0=$(date +%s)
echo "[ogrid] cell=$CELL eta=$ETA kappa=$KAP sigma=$SIG lam=$LAM gpu=$GPU start $(date '+%F %T')"
env GPU=$GPU TSEED=$SEED DATA_ROOT=$CELLDIR WPROJ=THREADING-MG-GRID \
  ETRIES=$ETRIES SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 CORE_N=20 WNAME_BASE=$ARM \
  ATT_CAP=$KAP ATY_SCALE=$ETA \
  ORB_LAM=$LAM ORB_DELTA=0.25 ORB_SIGMA=$SIG ORB_SIGMA_DECAY=0.5 ORB_ANNEAL=2 \
  bash scripts/threading/round_threading.sh threading "$ARM" 1 eval-pk
rc=$?
T1=$(date +%s)
echo "[ogrid] cell=$CELL rc=$rc in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"

# ---- readout -> csv (pass@5 criterion; SR + rescued recorded alongside) --- #
[ -f "$CSV" ] || echo "cell,arm,eta,kappa,sigma,lam,wall_min,rc,eval_sr,rescued,pass5" > "$CSV"
/root/workspace/baojiachun/.venv_mg/bin/python - "$CELL" "$ARM" "$ETA" "$KAP" "$SIG" "$LAM" "$rc" "$(( (T1-T0)/60 ))" "$CSV" <<'PYEOF'
import sys, os, json, glob
cell, arm, eta, kap, sig, lam, rc, wall, csv = sys.argv[1:10]
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
line = f"{cell},{arm},{eta},{kap},{sig},{lam},{wall},{rc},{sr},{rescued},{pass5}"
with open(csv, "a") as f:
    f.write(line + "\n")
print(f"[ogrid-readout] {line}")
PYEOF
