#!/bin/bash
# thread_kgrid_probe.sh -- ONE cell of the eta x kappa 2-D grid (user order
# 2026-09-17: eta {2,4,8,16} x kappa {2.5,5,10} = 12 cells, ATY mechanism,
# one GPU per cell, port 1022 + 1024). Copy of thread_grid_probe.sh with
# these deltas ONLY:
#   * KAP is a REQUIRED parameter, passed through to round_threading.sh as
#     ATT_CAP (the KL-cost cap kappa);
#   * CELL default / WNAME_BASE include the kappa value;
#   * csv = GRID_s233/kgrid_results.csv with a kappa column (the historical
#     16-cell grid_results.csv schema stays untouched).
# Everything else identical: MODE=eval-pk round1 on the s233 base (symlinked
# read-only), pass@5 criterion, 3600s hard cap, throwaway cell dir.
#
# usage: ARM=ATY ETA=<f> KAP=<f> GPU=<id> [CELL=<name>] \
#          bash scripts/threading/thread_kgrid_probe.sh
set -u
ARM=${ARM:?set ARM=ATY}
ETA=${ETA:?set ETA=<raw guidance scale>}
KAP=${KAP:?set KAP=<KL cap kappa>}
GPU=${GPU:?set GPU=<cuda id>}
CELL=${CELL:-${ARM}_eta${ETA}_k${KAP}}
SEED=233
ETRIES=5
WNAME_BASE=${WNAME_BASE:-${ARM}_eta${ETA}_k${KAP}}
ROOT=/root/workspace/baojiachun/scout
BASE_ROOT=$ROOT/data/2026_9_16_threading_p1/THREADING-MG-p1-s233
GRIDROOT=$ROOT/data/2026_9_16_threading_p1/GRID_s233
CELLDIR=$GRIDROOT/$CELL
CSV=$GRIDROOT/kgrid_results.csv

[ -f "$BASE_ROOT/threading/rollout/threading_core.hdf5" ] || { echo "[kgrid] FATAL: s233 core missing"; exit 1; }
ls "$BASE_ROOT"/threading/train/DP/DP-base/checkpoints/*.ckpt >/dev/null 2>&1 || { echo "[kgrid] FATAL: s233 DP-base missing"; exit 1; }
ls "$BASE_ROOT"/threading/train/dyn/dyn-base/*/scout_vib.ckpt >/dev/null 2>&1 || { echo "[kgrid] FATAL: s233 dyn-base missing"; exit 1; }

mkdir -p "$CELLDIR/threading/rollout" "$CELLDIR/threading/train/DP" "$CELLDIR/threading/train/dyn"
ln -sfn "$BASE_ROOT/threading/rollout/threading_core.hdf5" "$CELLDIR/threading/rollout/threading_core.hdf5"
ln -sfn "$BASE_ROOT/threading/train/DP/DP-base" "$CELLDIR/threading/train/DP/DP-base"
ln -sfn "$BASE_ROOT/threading/train/dyn/dyn-base" "$CELLDIR/threading/train/dyn/dyn-base"

cd "$ROOT" || exit 1
T0=$(date +%s)
echo "[kgrid] cell=$CELL arm=$ARM eta=$ETA kappa=$KAP gpu=$GPU start $(date '+%F %T')"
env GPU=$GPU TSEED=$SEED DATA_ROOT=$CELLDIR WPROJ=THREADING-MG-GRID \
  ETRIES=$ETRIES SHARD_P=8 SHARD_ENVS=25 EVALNENV=25 CORE_N=20 WNAME_BASE=$ARM \
  ATT_CAP=$KAP ATY_SCALE=$ETA \
  bash scripts/threading/round_threading.sh threading "$ARM" 1 eval-pk
rc=$?
T1=$(date +%s)
echo "[kgrid] cell=$CELL rc=$rc in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"

# ---- readout -> csv (pass@5 criterion; SR + rescued recorded alongside) --- #
[ -f "$CSV" ] || echo "cell,arm,eta,kappa,wall_min,rc,eval_sr,rescued,pass5" > "$CSV"
/root/workspace/baojiachun/.venv_mg/bin/python - "$CELL" "$ARM" "$ETA" "$KAP" "$rc" "$(( (T1-T0)/60 ))" "$CSV" <<'PYEOF'
import sys, os, json, glob
cell, arm, eta, kap, rc, wall, csv = sys.argv[1:8]
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
line = f"{cell},{arm},{eta},{kap},{wall},{rc},{sr},{rescued},{pass5}"
with open(csv, "a") as f:
    f.write(line + "\n")
print(f"[kgrid-readout] {line}")
PYEOF
