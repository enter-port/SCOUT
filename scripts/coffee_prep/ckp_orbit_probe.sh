#!/bin/bash
# ckp_orbit_probe.sh -- ONE cell of the ORBIT parameter grid for
# COFFEEPREP-MG-p1 (2026-09-20). COPY of scripts/threading/thread_orbit_probe.sh
# with these deltas ONLY:
#   * task/paths coffee_prep (campaign 2026_9_19_coffeeprep_mg_p1);
#   * PRESEED optimization (same as ckp_kgrid_probe.sh: canonical frozen
#     failure set + eval json copied in, resume branch skips phase A);
#   * timeout 10800s hard cap (rc=124 -> TIMEOUT row, no verdict);
#   * csv = GRID_s233/ogrid_results.csv with an avg_jerk column.
# Everything else identical: ARM fixed to ORBIT; ETA/KAP pinned per-cell by
# env (the shared ATY verdict); SIG and LAM are REQUIRED parameters ->
# ORB_SIGMA / ORB_LAM; MODE=eval-pk round1 on the s233 base (symlinked
# read-only), pass@5 criterion.
#
# usage: ETA=<f> KAP=<f> SIG=<f> LAM=<f> GPU=<id> [CELL=<name>] \
#          bash scripts/coffee_prep/ckp_orbit_probe.sh
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
CAMP=$ROOT/data/2026_9_19_coffeeprep_mg_p1
BASE_ROOT=$CAMP/COFFEEPREP-MG-p1-s233
GRIDROOT=$CAMP/GRID_s233
PRESEED=$GRIDROOT/PRESEED
CELLDIR=$GRIDROOT/$CELL
CSV=$GRIDROOT/ogrid_results.csv

[ -f "$BASE_ROOT/coffee_prep/rollout/coffee_prep_core.hdf5" ] || { echo "[ogrid] FATAL: s233 core missing"; exit 1; }
ls "$BASE_ROOT"/coffee_prep/train/DP/DP-base/checkpoints/*.ckpt >/dev/null 2>&1 || { echo "[ogrid] FATAL: s233 DP-base missing"; exit 1; }
ls "$BASE_ROOT"/coffee_prep/train/dyn/dyn-base/*/scout_vib.ckpt >/dev/null 2>&1 || { echo "[ogrid] FATAL: s233 dyn-base missing"; exit 1; }

mkdir -p "$CELLDIR/coffee_prep/rollout" "$CELLDIR/coffee_prep/train/DP" "$CELLDIR/coffee_prep/train/dyn"
ln -sfn "$BASE_ROOT/coffee_prep/rollout/coffee_prep_core.hdf5" "$CELLDIR/coffee_prep/rollout/coffee_prep_core.hdf5"
ln -sfn "$BASE_ROOT/coffee_prep/train/DP/DP-base" "$CELLDIR/coffee_prep/train/DP/DP-base"
ln -sfn "$BASE_ROOT/coffee_prep/train/dyn/dyn-base" "$CELLDIR/coffee_prep/train/dyn/dyn-base"

# -- preseed (arm-aware; ORBIT cells share the canonical unguded failure set)
RDIR=$CELLDIR/coffee_prep/rollout/$ARM-exp1
if [ -f "$PRESEED/failed.json" ] && [ ! -f "$RDIR/failed.json" ] && [ ! -f "$RDIR/all.hdf5" ]; then
  mkdir -p "$RDIR/log"
  cp "$PRESEED/failed.json" "$RDIR/failed.json"
  PJ=$(ls "$PRESEED"/log/coffee_prep_SCOUT_rollout_exp1.json 2>/dev/null | head -1)
  if [ -n "$PJ" ]; then
    /root/workspace/baojiachun/.venv_mg/bin/python - "$PJ" "$RDIR/log/coffee_prep_SCOUT_rollout_exp1.json" <<'PYEOF'
import sys, json
d = json.load(open(sys.argv[1]))
d.pop("wandb_run_id", None)          # cells must not backfill into the preseed run
json.dump(d, open(sys.argv[2], "w"), indent=1)
PYEOF
  fi
  echo "[ogrid] preseeded failure set -> $RDIR/failed.json ($(stat -c%s "$RDIR/failed.json") bytes)"
fi

cd "$ROOT" || exit 1
T0=$(date +%s)
echo "[ogrid] cell=$CELL eta=$ETA kappa=$KAP sigma=$SIG lam=$LAM gpu=$GPU start $(date '+%F %T')"
timeout -k 60 10800 env GPU=$GPU TSEED=$SEED DATA_ROOT=$CELLDIR WPROJ=COFFEEPREP-MG-GRID \
  ETRIES=$ETRIES SHARD_P=5 SHARD_ENVS=25 EVALNENV=25 CORE_N=20 WNAME_BASE=$ARM \
  ATT_CAP=$KAP ATY_SCALE=$ETA \
  ORB_LAM=$LAM ORB_DELTA=0.25 ORB_SIGMA=$SIG ORB_SIGMA_DECAY=0.5 ORB_ANNEAL=2 \
  bash scripts/coffee_prep/round_ckp.sh coffee_prep "$ARM" 1 eval-pk
rc=$?
T1=$(date +%s)
echo "[ogrid] cell=$CELL rc=$rc in $(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"

# ---- readout -> csv (pass@5 criterion; SR + rescued + jerk alongside) ---- #
[ -f "$CSV" ] || echo "cell,arm,eta,kappa,sigma,lam,wall_min,rc,eval_sr,rescued,pass5,avg_jerk" > "$CSV"
/root/workspace/baojiachun/.venv_mg/bin/python - "$CELL" "$ARM" "$ETA" "$KAP" "$SIG" "$LAM" "$rc" "$(( (T1-T0)/60 ))" "$CSV" <<'PYEOF'
import sys, os, json, glob
cell, arm, eta, kap, sig, lam, rc, wall, csv = sys.argv[1:10]
rdir = os.path.join(os.path.dirname(csv), cell, "coffee_prep", "rollout", f"{arm}-exp1")
sr = rescued = pass5 = jerk = ""
rj = glob.glob(os.path.join(rdir, "log", "*_rollout_exp1.json")) or \
     glob.glob(os.path.join(rdir, "log", "*_SCOUT_rollout_exp1.json"))
if rj:
    d = json.load(open(sorted(rj)[-1]))
    sr = d.get("success_rate", "")
ej = os.path.join(rdir, "log", f"coffee_prep_{arm}_explore_exp1.json")
if os.path.isfile(ej):
    d = json.load(open(ej))
    rescued = d.get("exploration_rescued", "")
    pass5 = d.get("pass_at_5", "")
    jerk = d.get("avg_jerk", "")
line = f"{cell},{arm},{eta},{kap},{sig},{lam},{wall},{rc},{sr},{rescued},{pass5},{jerk}"
with open(csv, "a") as f:
    f.write(line + "\n")
print(f"[ogrid-readout] {line}")
PYEOF
