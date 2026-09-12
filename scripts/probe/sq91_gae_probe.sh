#!/bin/bash
# sq91_gae_probe.sh -- GAElike(anchor/recent weighting) vs SCOUT(atypical)
# pass@10 A/B on the SQUARE-9-1-orbit-s233 round0 (DP-base 599.ckpt +
# dyn-base 20260826-112119), frozen UNGUIDED-DP failure set
# (user order 2026-09-09, branch GAElike-dev; square twin of
# can824_gae_probe.sh).
#
# Protocol (identical to the can probe):
#   * failed set = DP-base UNGUIDED eval (seed 42..141, 100 scenes, env=25)
#     failures, derived ONCE by step 0 (~10-15 min). The campaign's own r1
#     failed.json came from an ORBIT-guided eval and is NOT reused; the
#     re-derivation uses the same --guide off recipe as can824 so both
#     tasks' probes share one derivation semantics.
#   * arms: P=4 shard workers, n_envs=25, ETRIES=10 (pass@10),
#     eta 3.0 / kappa 2.5 / gst 100 -- IDENTICAL except --guide.
#     Default pair = aty + gae (anchor weighting). Run a second time with
#     GAE_EXTRA="--gae-weighting recent" GAE_LABEL=recent (ROUND=2, same
#     OFF/N window) for the recent variant; its aty arm doubles as a
#     run-to-run replicate.
#   * 45-min hard backstop per arm (timeout -k 60 2700); orphans pkill'd by
#     the unique probe path pattern.
#   * read-only reuse of the chain's ckpts/core (square_core.hdf5 = the
#     round0 training core, 20 demos); outputs ONLY under data/gae_probe_sq91/.
#     Never touches campaign data or running chains.
#
# usage: ROUND=1 [OFF=0] [N=16] [GAE_GAMMA=0.9] [GAE_NORM=1] [GAE_LABEL=anchor]
#        [P=4] [TRIES=10] [ATY_SCALE=3.0] [CAP=2.5] [GST=100] [GPU_ATY=3] [GPU_GAE=1]
#        [GAE_EXTRA="--gae-weighting recent"] [BACKSTOP=2700] [DRY_RUN=1] [PREP_ONLY=1]
#        bash scripts/probe/sq91_gae_probe.sh
# Window note: square horizon 500 (can 300). Campaign r1: ~63 scenes x10
# tries on 2 workers took ~4h -> ~7.6 min/scene/worker. N=16 -> slowest
# shard ~4 scenes (~30 min) + ~4 min startup, inside the 45-min backstop.
set -uo pipefail
ROUND=${ROUND:?set ROUND=<iteration round >=1>}
OFF=${OFF:-0}
N=${N:-16}
GAE_GAMMA=${GAE_GAMMA:-0.9}
GAE_NORM=${GAE_NORM:-1}
GAE_LABEL=${GAE_LABEL:-anchor}
P=${P:-4}
TRIES=${TRIES:-10}
ATY_SCALE=${ATY_SCALE:-3.0}
CAP=${CAP:-2.5}
GST=${GST:-100}
GPU_ATY=${GPU_ATY:-3}
GPU_GAE=${GPU_GAE:-1}
BACKSTOP=${BACKSTOP:-2700}

cd /root/workspace/baojiachun/scout-gae || exit 1
PY=/root/workspace/baojiachun/.venv/bin/python
TH=/root/workspace/baojiachun/scout-orbit/data/2026_9_1_orbchain/ORBIT-s233/square
DP=$TH/train/DP/DP-base/checkpoints/599.ckpt
VIB=$TH/train/dyn/dyn-base/20260826-112119/scout_vib.ckpt
CORE=$TH/rollout/square_core.hdf5
ROOT=data/gae_probe_sq91
FAILED=$ROOT/failed_dp.json
T=$ROOT/r${ROUND}
[ -f "$DP" ] && [ -f "$VIB" ] && [ -f "$CORE" ] \
  || { echo "[sqprobe] FATAL: chain ckpt / core missing"; exit 1; }
mkdir -p "$ROOT" "$T/log"

export MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- step 0: derive the DP-base failed set (once, UNGUIDED) -------------- #
if [ ! -f "$FAILED" ]; then
  echo "[sqprobe] deriving failed set: DP-base unguided eval seed42 x100 (GPU$GPU_GAE)"
  mkdir -p "$ROOT/evalonly"
  timeout -k 60 2400 env CUDA_VISIBLE_DEVICES=$GPU_GAE \
    SCOUT_RENDER_GPU=$GPU_GAE \
    $PY -m scout.eval.run_rollout \
      --config configs/eval_square_entropy.yaml --task square --exp-num 0 \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
      --guide off --explore-mode rescue --eval-only \
      --save-failed-set "$FAILED" \
      --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
      --output-dir "$ROOT/evalonly" \
      --output-json "$ROOT/evalonly/eval.json" \
    > "$ROOT/evalonly.stdout" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$FAILED" ] \
    || { echo "[sqprobe] FATAL: failed-set derivation rc=$rc"; exit 1; }
fi
$PY - "$FAILED" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
n = len(spec["failed_init_indices"])
print(f"[sqprobe] failed set: {n} inits, recorded SR "
      f"{spec.get('baseline_solved')}/100 "
      f"(reference: campaign r1 ORBIT-guided eval = 63 failed / SR .37; "
      f"this derivation is UNGUIDED)")
PYEOF
[ "${PREP_ONLY:-0}" = "1" ] && { echo "[sqprobe] PREP_ONLY done"; exit 0; }

# ---- window cut: failed[OFF:OFF+N], original indices kept ---------------- #
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
print(f"[sqprobe] window [{'0' if n <= 0 else str(off)}:{'all' if n <= 0 else str(off+n)}] = {idx}")
PYEOF
WIN=$T/win.json

# ---- per-arm config copies (dose is config-only, th_p10_probe pattern) --- #
for spec in "aty:${ATY_SCALE}" "gae:${ATY_SCALE}"; do
  arm=${spec%%:*}; sc=${spec#*:}
  $PY - "configs/eval_square_entropy.yaml" "$T/cfg_${arm}.yaml" "$sc" "$GST" <<'PYEOF'
import sys, yaml
src, out, sc, gst = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["exploration"]["guidance_scale"] = float(sc)
cfg["exploration"]["guidance_start_timestep"] = int(gst)
cfg["wandb"]["use_wandb"] = False
cfg["wandb"]["project"] = "GAElike-sq91probe"
cfg["wandb"]["tags"] = ["gaeprobe-sq"]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[sqprobe-cfg] {out}: guidance_scale={sc} gst={gst}")
PYEOF
done

# ---- one arm = P shard workers on one GPU + merge, backstop-capped ------- #
run_arm() { # name gpu guide extra...
  local name=$1 gpu=$2 guide=$3; shift 3
  local extra=("$@")
  echo "[sqprobe] arm=$name gpu=$gpu guide=$guide P=$P extra=${extra[*]:-} (backstop ${BACKSTOP}s)"
  local t0=$(date +%s)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: timeout -k 60 $BACKSTOP env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu PYTHON=$PY CLEANUP_SHARDS=0 bash scripts/infra/shard_rollout.sh $P $T/$name/log/explore.json $T/$name/success.hdf5 $T/$name/all.hdf5 $CORE -- --config $T/cfg_${name}.yaml --task square --exp-num $ROUND --base-dp-ckpt $DP --core-hdf5 $CORE --vib-ckpt $VIB --guide $guide ${extra[@]+"${extra[@]}"} --explore-mode rescue --explore-try-times $TRIES --failed-set-json $WIN --n-envs 25 --seed 42 --eval-seed 42 --no-wandb --output-dir $T/$name --output-success $T/$name/success.hdf5 --output-all $T/$name/all.hdf5 > $T/$name.stdout 2>&1"
    return 0
  fi
  mkdir -p "$T/$name/log"
  timeout -k 60 "$BACKSTOP" env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu \
    PYTHON="$PY" CLEANUP_SHARDS=0 \
    bash scripts/infra/shard_rollout.sh "$P" \
      "$T/$name/log/explore.json" "$T/$name/success.hdf5" "$T/$name/all.hdf5" \
      "$CORE" -- \
      --config "$T/cfg_${name}.yaml" --task square --exp-num "$ROUND" \
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
    local pat="gae_probe_sq91/r${ROUND}/${name}"
    pkill -f "$pat" 2>/dev/null && sleep 3
  fi
  local t1=$(date +%s)
  echo "$rc" > "$T/$name.rc"
  echo "[sqprobe] arm=$name rc=$rc wall=$(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
}

echo "[sqprobe] ROUND=$ROUND OFF=$OFF N=$N P=$P TRIES=$TRIES aty(scale/cap/gst)=$ATY_SCALE/$CAP/$GST gae(gamma/norm/label)=$GAE_GAMMA/$GAE_NORM/$GAE_LABEL gpus aty/gae=$GPU_ATY/$GPU_GAE"
GAE_ARGS=(--atypical-cap "$CAP" --gae-gamma "$GAE_GAMMA" --gae-norm "$GAE_NORM")
# extra gae-arm CLI flags (whitespace-split), e.g. GAE_EXTRA="--gae-weighting recent"
if [ -n "${GAE_EXTRA:-}" ]; then
  read -ra _GE <<< "$GAE_EXTRA"
  GAE_ARGS+=("${_GE[@]}")
fi
run_arm gae "$GPU_GAE" gaelike ${GAE_ARGS[@]+"${GAE_ARGS[@]}"} &
GAE_PID=$!
run_arm aty "$GPU_ATY" atypical --atypical-cap "$CAP" &
ATY_PID=$!
wait "$GAE_PID"; wait "$ATY_PID"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[sqprobe] DRY_RUN done"; exit 0; }

# ---- summary: pass@10 per arm + telemetry tails -------------------------- #
sleep 2
$PY - "$T" "$ROUND" "$GAE_GAMMA" "$GAE_NORM" "$GAE_LABEL" "$ROOT/ledger.csv" <<'PYEOF'
import csv, glob, json, os, re, sys
T, rnd, gamma, norm, label, ledger = sys.argv[1:7]
rows = []
for arm in ("aty", "gae"):
    row = {"round": rnd, "arm": arm,
           "params": (f"g{gamma},n{norm},{label}" if arm == "gae" else "aty"),
           "gamma": gamma, "norm": norm}
    rc_f = f"{T}/{arm}.rc"
    row["rc"] = open(rc_f).read().strip() if os.path.exists(rc_f) else "?"
    js = f"{T}/{arm}/log/explore.json"
    if os.path.exists(js):
        d = json.load(open(js))
        row["rescued"] = d.get("exploration_rescued")
        row["n"] = d.get("n_failed")
        row["p10_rescue_rate"] = (round(d["exploration_rescued"] / d["n_failed"], 3)
                                  if d.get("n_failed") else None)
        row["pass_at_10"] = (round((d.get("baseline_solved", 0)
                                    + d["exploration_rescued"]) / 100, 3)
                             if d.get("n_failed") is not None else None)
        row["jerk"] = round(d.get("avg_jerk") or 0, 4)
        row["collected"] = d.get("collected_trajs")
    else:
        for k in ("rescued", "n", "p10_rescue_rate", "pass_at_10", "jerk"):
            row[k] = "NOJSON"
        row["collected"] = ""
    inject = gae_tel = None
    for p in sorted(glob.glob(f"{T}/{arm}/log/shard*.stdout")):
        txt = open(p, errors="ignore").read()
        m = re.findall(r"mean_inject=([0-9.eE+-]+)", txt)
        if m: inject = m[-1]
        m = re.findall(r"mean_kl=([0-9.eE+-]+)", txt)
        if m: gae_tel = m[-1]
    row["mean_inject"] = inject or ""
    row["gae_mean_kl"] = gae_tel or ""
    rows.append(row)
    print(f"[sqprobe] {arm}: {row}")
aty = next((r for r in rows if r["arm"] == "aty"), {})
gae = next((r for r in rows if r["arm"] == "gae"), {})
try:
    a, g = float(aty["pass_at_10"]), float(gae["pass_at_10"])
    verdict = "GAElike WINS" if g > a else ("TIE" if g == a else "SCOUT wins")
    print(f"[sqprobe] VERDICT: {verdict} (pass@10 gae={g} vs aty={a})")
except (TypeError, ValueError, KeyError):
    print("[sqprobe] VERDICT: INCOMPLETE (missing merged json)")
new = not os.path.exists(ledger)
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
    print("[sqprobe] CUMULATIVE (all rounds so far): " + " ".join(
        f"{a}={s}/{n} rescue_rate={s/max(n,1):.3f}"
        for a, (s, n) in sorted(acc.items())))
PYEOF

# ---- post-hoc render-integrity check (report-only) ----------------------- #
for arm in aty gae; do
  if [ -f "$T/$arm/all.hdf5" ]; then
    $PY scripts/viz/vis_validate.py "$T/$arm/all.hdf5" \
      > "$T/$arm.vischeck.txt" 2>&1 \
      && echo "[sqprobe] $arm render: $(tail -1 $T/$arm.vischeck.txt)" \
      || echo "[sqprobe] $arm render: CHECK FAILED (see $T/$arm.vischeck.txt)"
  fi
done
echo "[sqprobe] ROUND $ROUND DONE $(date '+%F %T')"
