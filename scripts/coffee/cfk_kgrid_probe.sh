#!/bin/bash
# cfk_kgrid_probe.sh -- ONE cell of the eta x kappa 2-D grid for
# COFFEE-MG-p1 (2026-09-20; user order: grid 范围探针后自动定,不停).
# COPY of scripts/coffee_prep/ckp_kgrid_probe.sh with these deltas ONLY:
#   * task/paths coffee (campaign 2026_9_20_coffee_mg_p1);
#   * PRESEED optimization: the canonical 100-scene eval + failed.json is
#     run ONCE into GRID_s233/PRESEED (unguided eval, identical across cells
#     since eval never attaches the planner) and copied into each fresh cell
#     (wandb_run_id stripped so cells never backfill into a shared run);
#     the round script's existing resume branch then skips phase A. Saves
#     ~20-40min eval per cell; pass@5 comparability IMPROVES (all cells
#     share the bit-identical failure set);
#   * timeout 10800s hard cap (rc=124 -> TIMEOUT row, no verdict);
#   * csv = GRID_s233/kgrid_results.csv with kappa + jerk columns.
# Everything else identical: MODE=eval-pk round1 on the s233 base (symlinked
# read-only), pass@5 criterion, throwaway cell dir.
#
# usage: ARM=ATY ETA=<f> KAP=<f> GPU=<id> [CELL=<name>] \
#          bash scripts/coffee/cfk_kgrid_probe.sh
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
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
BASE_ROOT=$CAMP/COFFEE-MG-p1-s233
GRIDROOT=$CAMP/GRID_s233
PRESEED=$GRIDROOT/PRESEED
CELLDIR=$GRIDROOT/$CELL
CSV=$GRIDROOT/kgrid_results.csv

[ -f "$BASE_ROOT/coffee/rollout/coffee_core.hdf5" ] || { echo "[kgrid] FATAL: s233 core missing"; exit 1; }
ls "$BASE_ROOT"/coffee/train/DP/DP-base/checkpoints/*.ckpt >/dev/null 2>&1 || { echo "[kgrid] FATAL: s233 DP-base missing"; exit 1; }
ls "$BASE_ROOT"/coffee/train/dyn/dyn-base/*/scout_vib.ckpt >/dev/null 2>&1 || { echo "[kgrid] FATAL: s233 dyn-base missing"; exit 1; }

mkdir -p "$CELLDIR/coffee/rollout" "$CELLDIR/coffee/train/DP" "$CELLDIR/coffee/train/dyn"
ln -sfn "$BASE_ROOT/coffee/rollout/coffee_core.hdf5" "$CELLDIR/coffee/rollout/coffee_core.hdf5"
ln -sfn "$BASE_ROOT/coffee/train/DP/DP-base" "$CELLDIR/coffee/train/DP/DP-base"
ln -sfn "$BASE_ROOT/coffee/train/dyn/dyn-base" "$CELLDIR/coffee/train/dyn/dyn-base"

# -- preseed: reuse the canonical frozen failure set + eval json (SCOUT-named,
#    wandb_run_id stripped) so the round script's resume branch skips phase A.
#    Works for ANY arm incl. the DP placebo cell (eval json falls back to the
#    SCOUT-named file; guide=off cells reuse the same unguded eval result).
RDIR=$CELLDIR/coffee/rollout/$ARM-exp1
if [ -f "$PRESEED/failed.json" ] && [ ! -f "$RDIR/failed.json" ] && [ ! -f "$RDIR/all.hdf5" ]; then
  mkdir -p "$RDIR/log"
  cp "$PRESEED/failed.json" "$RDIR/failed.json"
  PJ=$(ls "$PRESEED"/log/coffee_SCOUT_rollout_exp1.json 2>/dev/null | head -1)
  if [ -n "$PJ" ]; then
    /root/workspace/baojiachun/.venv_mg/bin/python - "$PJ" "$RDIR/log/coffee_SCOUT_rollout_exp1.json" <<'PYEOF'
import sys, json
d = json.load(open(sys.argv[1]))
d.pop("wandb_run_id", None)          # cells must not backfill into the preseed run
json.dump(d, open(sys.argv[2], "w"), indent=1)
PYEOF
  fi
  echo "[kgrid] preseeded failure set -> $RDIR/failed.json ($(stat -c%s "$RDIR/failed.json") bytes)"
fi

cd "$ROOT" || exit 1
T0=$(date +%s)
echo "[kgrid] cell=$CELL arm=$ARM eta=$ETA kappa=$KAP gpu=$GPU start $(date '+%F %T')"
timeout -k 60 10800 env GPU=$GPU TSEED=$SEED DATA_ROOT=$CELLDIR WPROJ=COFFEE-MG-GRID \
  ETRIES=$ETRIES SHARD_P=5 SHARD_ENVS=25 EVALNENV=25 CORE_N=20 WNAME_BASE=$ARM \
  ATT_CAP=$KAP ATY_SCALE=$ETA \
  bash scripts/coffee/round_cfk.sh coffee "$ARM" 1 eval-pk
rc=$?
T1=$(date +%s)
echo "[kgrid] cell=$CELL rc=$rc in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"

# ---- readout -> csv (pass@5 criterion; SR + rescued + jerk alongside) ---- #
[ -f "$CSV" ] || echo "cell,arm,eta,kappa,wall_min,rc,eval_sr,rescued,pass5,avg_jerk" > "$CSV"
/root/workspace/baojiachun/.venv_mg/bin/python - "$CELL" "$ARM" "$ETA" "$KAP" "$rc" "$(( (T1-T0)/60 ))" "$CSV" <<'PYEOF'
import sys, os, json, glob
cell, arm, eta, kap, rc, wall, csv = sys.argv[1:8]
rdir = os.path.join(os.path.dirname(csv), cell, "coffee", "rollout", f"{arm}-exp1")
sr = rescued = pass5 = jerk = ""
rj = glob.glob(os.path.join(rdir, "log", "*_rollout_exp1.json")) or \
     glob.glob(os.path.join(rdir, "log", "*_SCOUT_rollout_exp1.json"))
if rj:
    d = json.load(open(sorted(rj)[-1]))
    sr = d.get("success_rate", "")
ej = os.path.join(rdir, "log", f"coffee_{arm}_explore_exp1.json")
if os.path.isfile(ej):
    d = json.load(open(ej))
    rescued = d.get("exploration_rescued", "")
    pass5 = d.get("pass_at_5", "")
    jerk = d.get("avg_jerk", "")
line = f"{cell},{arm},{eta},{kap},{wall},{rc},{sr},{rescued},{pass5},{jerk}"
with open(csv, "a") as f:
    f.write(line + "\n")
print(f"[kgrid-readout] {line}")
PYEOF
