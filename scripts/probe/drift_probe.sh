#!/bin/bash
# drift_probe.sh -- cloudrep(方案一) vs SCOUT(atypical) pass@5 A/B on the
# round0 frozen-failure-set of CAN-8-24-entropy-s233 / SQUARE-8-26-entropy-s233
# (user /goal 2026-09-09, branch drift-dev).
#
# Protocol (user order):
#   * failed set = DP-base eval (seed 42..141, 100 scenes) failures, derived
#     once by step 0 (cached per task; can824 precedent: 43; square's
#     gae-era recount was 61-63 -- the runtime derivation is authoritative).
#   * both arms: P=4 CPU shard workers, n_envs=25, ETRIES=5 (pass@5),
#     eta 3.0 / kappa 2.5 / gst 100 -- IDENTICAL except --guide atypical vs
#     cloudrep. Retry 0 is MECHANISM-bitwise atypical (empty cloud), but the
#     start-gate forks the two arms' batch composition -> the shared RNG
#     stream lands on different scenes: read the A/B as a same-failed-set
#     STATISTICAL comparison (retry-0 flips = RNG realization noise, never
#     cloud effect).
#   * 45-min hard backstop per arm (timeout -k 60 2700); orphans pkill'd by
#     the unique probe path pattern.
#   * read-only reuse of the campaigns' train/ ckpts; outputs ONLY under
#     data/drift_probe_<task>/. Never touches campaign data or running chains.
#
# usage: TASK=can|square ROUND=1 [OFF=0] [N=0] [TAU=0.5] [P=4] [TRIES=5]
#        [SCALE=3.0] [CAP=2.5] [GST=100] [GPU_ATY=3] [GPU_CREP=1]
#        [BACKSTOP=2700] [DRY_RUN=1] bash scripts/probe/drift_probe.sh
# Windowed rounds (th_p10_probe pattern): full square set may exceed the
# 45-min budget; each ROUND runs failed[OFF:OFF+N] (N=0 = full set), the
# ledger CUMULATES rescued/n across rounds. Step 0 (failed-set derivation)
# runs automatically when the cached json is missing (~10 min unguided eval);
# force with PREP_ONLY=1.
set -uo pipefail
TASK=${TASK:?set TASK=can|square}
ROUND=${ROUND:?set ROUND=<iteration round >=1>}
OFF=${OFF:-0}
N=${N:-0}
TAU=${TAU:-0.5}
P=${P:-4}
TRIES=${TRIES:-5}
SCALE=${SCALE:-3.0}
CAP=${CAP:-2.5}
GST=${GST:-100}
GPU_ATY=${GPU_ATY:-3}
GPU_CREP=${GPU_CREP:-1}
BACKSTOP=${BACKSTOP:-2700}
# worktree root: CPFS is at 0 available (2026-09-09) -- the drift-dev
# deployment lives on the pod-local 1TB NVMe (/tmp/scout-drift, 671G free)
# so the probe never writes to the full shared volume; ckpts/datasets on
# CPFS are read-only inputs. Override to relocate.
DRIFT_HOME=${DRIFT_HOME:-/tmp/scout-drift}

cd "$DRIFT_HOME" || { echo "[drift-probe] FATAL: DRIFT_HOME=$DRIFT_HOME missing"; exit 1; }
PY=/root/workspace/baojiachun/.venv/bin/python

case "$TASK" in
  can)
    TH=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
    DYN_TS=20260824-232156
    CFG=configs/eval_can_entropy.yaml
    COREF=can_core.hdf5 ;;
  square)
    TH=/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square
    DYN_TS=20260826-112119
    CFG=configs/eval_square_entropy.yaml
    COREF=square_core.hdf5 ;;
  *) echo "[drift-probe] FATAL: TASK=$TASK unsupported"; exit 1 ;;
esac
DP=$TH/train/DP/DP-base/checkpoints/599.ckpt
VIB=$TH/train/dyn/dyn-base/$DYN_TS/scout_vib.ckpt
OFFICIAL=/root/workspace/baojiachun/scout/data/robomimic/$TASK/ph/image_v141_abs.hdf5
ROOT=data/drift_probe_${TASK}
CORE=$ROOT/$COREF
FAILED=$ROOT/failed_dp.json
T=$ROOT/r${ROUND}
[ -f "$DP" ] && [ -f "$VIB" ] && [ -f "$OFFICIAL" ] \
  || { echo "[drift-probe] FATAL: campaign ckpt / official dataset missing ($DP | $VIB | $OFFICIAL)"; exit 1; }
mkdir -p "$ROOT" "$T/log"

# ---- step 0a: rebuild the seeded 20/200 core (deterministic, TSEED 233) -- #
if [ ! -f "$CORE" ]; then
  echo "[drift-probe] rebuilding core: split_core 20/200 seed 233"
  $PY scripts/analysis/split_core.py "$OFFICIAL" "$CORE" 20 233 \
    || { echo "[drift-probe] FATAL: split_core failed"; exit 1; }
fi

export MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
# the venv lives on the 0-available CPFS -- never let CPython try to write
# new .pyc files there (ENOSPC mid-import is silent corruption territory).
export PYTHONDONTWRITEBYTECODE=1

# ---- step 0b: derive the DP-base failed set (once) ----------------------- #
if [ ! -f "$FAILED" ]; then
  echo "[drift-probe] deriving failed set: DP-base eval seed42 x100 (GPU$GPU_CREP)"
  mkdir -p "$ROOT/evalonly"
  timeout -k 60 2400 env CUDA_VISIBLE_DEVICES=$GPU_CREP \
    SCOUT_RENDER_GPU=$GPU_CREP \
    $PY -m scout.eval.run_rollout \
      --config "$CFG" --task "$TASK" --exp-num 0 \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
      --guide off --explore-mode rescue --eval-only \
      --save-failed-set "$FAILED" \
      --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
      --output-dir "$ROOT/evalonly" \
      --output-json "$ROOT/evalonly/eval.json" \
    > "$ROOT/evalonly.stdout" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$FAILED" ] \
    || { echo "[drift-probe] FATAL: failed-set derivation rc=$rc"; exit 1; }
fi
$PY - "$FAILED" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
n = len(spec["failed_init_indices"])
print(f"[drift-probe] failed set: {n} inits, recorded SR "
      f"{spec.get('baseline_solved')}/100")
PYEOF
[ "${PREP_ONLY:-0}" = "1" ] && { echo "[drift-probe] PREP_ONLY done"; exit 0; }

# ---- window cut: failed[OFF:OFF+N], original indices kept (th pattern) --- #
$PY - "$FAILED" "$T/win.json" "$OFF" "$N" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
off, n = int(sys.argv[3]), int(sys.argv[4])
allf = spec["failed_init_indices"]
idx = allf if n <= 0 else allf[off:off + n]
if n > 0:
    assert len(idx) == n, f"window underflow: wants [{off}:{off+n}] of {len(allf)}"
spec["failed_init_indices"] = idx
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print(f"[drift-probe] window [{'0' if n <= 0 else str(off)}:{'all' if n <= 0 else str(off+n)}] = {idx}")
PYEOF
WIN=$T/win.json

# ---- per-arm config copies (dose is config-only, th_p10_probe pattern) --- #
for spec in "aty:${SCALE}" "crep:${SCALE}"; do
  arm=${spec%%:*}; sc=${spec#*:}
  $PY - "$CFG" "$T/cfg_${arm}.yaml" "$sc" "$GST" <<'PYEOF'
import sys, yaml
src, out, sc, gst = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["exploration"]["guidance_scale"] = float(sc)
cfg["exploration"]["guidance_start_timestep"] = int(gst)
cfg["wandb"]["use_wandb"] = False
cfg["wandb"]["project"] = "drift-probe"
cfg["wandb"]["tags"] = ["driftprobe"]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[drift-probe-cfg] {out}: guidance_scale={sc} gst={gst}")
PYEOF
done

# ---- one arm = P shard workers on one GPU + merge, backstop-capped ------- #
run_arm() { # name gpu guide extra...
  local name=$1 gpu=$2 guide=$3; shift 3
  local extra=("$@")
  echo "[drift-probe] arm=$name gpu=$gpu guide=$guide P=$P extra=${extra[*]:-} (backstop ${BACKSTOP}s)"
  local t0=$(date +%s)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: timeout -k 60 $BACKSTOP env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu PYTHON=$PY CLEANUP_SHARDS=0 bash scripts/infra/shard_rollout.sh $P $T/$name/log/explore.json $T/$name/success.hdf5 $T/$name/all.hdf5 $CORE -- --config $T/cfg_${name}.yaml --task $TASK --exp-num $ROUND --base-dp-ckpt $DP --core-hdf5 $CORE --vib-ckpt $VIB --guide $guide ${extra[@]+"${extra[@]}"} --explore-mode rescue --explore-try-times $TRIES --failed-set-json $WIN --n-envs 25 --seed 42 --eval-seed 42 --no-wandb --output-dir $T/$name --output-success $T/$name/success.hdf5 --output-all $T/$name/all.hdf5 > $T/$name.stdout 2>&1"
    return 0
  fi
  mkdir -p "$T/$name/log"
  timeout -k 60 "$BACKSTOP" env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu \
    PYTHON="$PY" CLEANUP_SHARDS=0 \
    bash scripts/infra/shard_rollout.sh "$P" \
      "$T/$name/log/explore.json" "$T/$name/success.hdf5" "$T/$name/all.hdf5" \
      "$CORE" -- \
      --config "$T/cfg_${name}.yaml" --task "$TASK" --exp-num "$ROUND" \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" --vib-ckpt "$VIB" \
      --guide "$guide" ${extra[@]+"${extra[@]}"} \
      --explore-mode rescue --explore-try-times "$TRIES" \
      --failed-set-json "$WIN" \
      --n-envs 25 --seed 42 --eval-seed 42 \
      --no-wandb \
      --output-dir "$T/$name" --output-success "$T/$name/success.hdf5" \
      --output-all "$T/$name/all.hdf5" \
      > "$T/$name.stdout" 2>&1
  local rc=$?
  if [ "$rc" = "124" ]; then
    local pat="drift_probe_${TASK}/r${ROUND}/${name}"
    pkill -f "$pat" 2>/dev/null && sleep 3
  fi
  local t1=$(date +%s)
  echo "$rc" > "$T/$name.rc"
  echo "[drift-probe] arm=$name rc=$rc wall=$(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
}

echo "[drift-probe] TASK=$TASK ROUND=$ROUND OFF=$OFF N=$N P=$P TRIES=$TRIES scale/cap/gst=$SCALE/$CAP/$GST tau=$TAU gpus aty/crep=$GPU_ATY/$GPU_CREP"
CREP_ARGS=(--atypical-cap "$CAP" --cloudrep-tau "$TAU")
run_arm crep "$GPU_CREP" cloudrep ${CREP_ARGS[@]+"${CREP_ARGS[@]}"} &
CREP_PID=$!
run_arm aty "$GPU_ATY" atypical --atypical-cap "$CAP" &
ATY_PID=$!
wait "$CREP_PID"; wait "$ATY_PID"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[drift-probe] DRY_RUN done"; exit 0; }

# ---- summary: pass@5 per arm + telemetry tails --------------------------- #
sleep 2
$PY - "$T" "$ROUND" "$TAU" "$ROOT/ledger.csv" <<'PYEOF'
import csv, glob, json, os, re, sys
T, rnd, tau, ledger = sys.argv[1:5]
rows = []
for arm in ("aty", "crep"):
    row = {"round": rnd, "arm": arm,
           "params": (f"tau{tau}" if arm == "crep" else "aty"),
           "tau": tau}
    rc_f = f"{T}/{arm}.rc"
    row["rc"] = open(rc_f).read().strip() if os.path.exists(rc_f) else "?"
    js = f"{T}/{arm}/log/explore.json"
    if os.path.exists(js):
        d = json.load(open(js))
        row["rescued"] = d.get("exploration_rescued")
        row["n"] = d.get("n_failed")
        row["p5_rescue_rate"] = (round(d["exploration_rescued"] / d["n_failed"], 3)
                                 if d.get("n_failed") else None)
        row["jerk"] = round(d.get("avg_jerk") or 0, 4)
        row["collected"] = d.get("collected_trajs")
    else:
        for k in ("rescued", "n", "p5_rescue_rate", "jerk"):
            row[k] = "NOJSON"
        row["collected"] = ""
    inject = crep_tel = pool = None
    for p in sorted(glob.glob(f"{T}/{arm}/log/shard*.stdout")):
        txt = open(p, errors="ignore").read()
        m = re.findall(r"mean_inject=([0-9.eE+-]+)", txt)
        if m: inject = m[-1]
        m = re.findall(r"mean_S=([0-9.eE+-]+)", txt)
        if m: crep_tel = m[-1]
        m = re.findall(r"mean_pool=([0-9.eE+-]+)", txt)
        if m: pool = m[-1]
    row["mean_inject"] = inject or ""
    row["crep_mean_S"] = crep_tel or ""
    row["crep_mean_pool"] = pool or ""
    rows.append(row)
    print(f"[drift-probe] {arm}: {row}")
aty = next((r for r in rows if r["arm"] == "aty"), {})
crep = next((r for r in rows if r["arm"] == "crep"), {})
try:
    a, c = int(aty["rescued"]), int(crep["rescued"])
    na, nc = int(aty["n"]), int(crep["n"])
    verdict = ("CREP WINS" if c > a else ("TIE" if c == a else "SCOUT wins"))
    print(f"[drift-probe] VERDICT: {verdict} (rescued@5 crep={c}/{nc} vs "
          f"aty={a}/{na})")
except (TypeError, ValueError, KeyError):
    print("[drift-probe] VERDICT: INCOMPLETE (missing merged json)")
new = not os.path.exists(ledger)
if rows:
    with open(ledger, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new: w.writeheader()
        w.writerows(rows)
from collections import defaultdict
if not new:
    acc = defaultdict(lambda: [0, 0])
    with open(ledger) as f:
        for r in csv.DictReader(f):
            try:
                acc[r["arm"]][0] += int(r["rescued"])
                acc[r["arm"]][1] += int(r["n"])
            except (TypeError, ValueError, KeyError):
                pass
    print("[drift-probe] CUMULATIVE (all rounds so far): " + " ".join(
        f"{a}={s}/{n} rescue_rate={s/max(n,1):.3f}"
        for a, (s, n) in sorted(acc.items())))
PYEOF

# ---- post-hoc render-integrity check (report-only) ----------------------- #
for arm in aty crep; do
  if [ -f "$T/$arm/all.hdf5" ]; then
    $PY scripts/viz/vis_validate.py "$T/$arm/all.hdf5" \
      > "$T/$arm.vischeck.txt" 2>&1 \
      && echo "[drift-probe] $arm render: $(tail -1 $T/$arm.vischeck.txt)" \
      || echo "[drift-probe] $arm render: CHECK FAILED (see $T/$arm.vischeck.txt)"
  fi
done
echo "[drift-probe] TASK=$TASK ROUND $ROUND DONE $(date '+%F %T')"
