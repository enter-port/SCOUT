#!/bin/bash
# can824_gae_probe.sh -- GAElike vs SCOUT(atypical) pass@10 A/B on the
# CAN-8-24-entropy-s233 round0 (DP-base 599.ckpt + dyn-base) frozen-failure-set
# (user /goal 2026-09-08, branch GAElike-dev).
#
# Protocol (user order, mirrors the campaign's r1 rescue protocol exactly):
#   * failed set = DP-base eval (seed 42..141, 100 scenes) failures -- derived
#     ONCE by step 0 below (the campaign's own rollout/ jsons were deleted in
#     the 09-01 cleanup; the seeded eval is deterministic, and the SQUARE
#     grid precedent rebuilt failed sets that matched history bit-for-bit);
#     expect SR .62 / 38 failed (NAMES.md CAN-8-24-entropy r1 historical).
#   * both arms: P=4 CPU shard workers, n_envs=25, ETRIES=10 (pass@10),
#     eta 3.0 / kappa 2.5 / gst 100 (eval_can_entropy.yaml, round_entropy.sh
#     ATT_CAP=2.5) -- IDENTICAL except --guide atypical vs gaelike.
#   * 45-min hard backstop per arm (timeout -k 60 2700); orphans pkill'd by
#     the unique probe path pattern.
#   * read-only reuse of the campaign's train/ ckpts; outputs ONLY under
#     data/gae_probe_can824/. Never touches campaign data or running chains.
#
# usage: ROUND=1 [OFF=0] [N=22] [GAE_GAMMA=0.9] [GAE_NORM=1] [P=4] [TRIES=10]
#        [ATY_SCALE=3.0] [CAP=2.5] [GST=100] [GPU_ATY=3] [GPU_GAE=1]
#        [BACKSTOP=2700] [DRY_RUN=1] bash scripts/probe/can824_gae_probe.sh
# Windowed rounds (th_p10_probe pattern, 2026-09-04 user precedent): the
# stride sharding (index %% P) is lumpy on this failed set ({9,10,10,14}
# scenes/stride at P=4), so the FULL 43-scene set needs ~55 min on the
# slowest shard -- over the 45-min budget. Each ROUND therefore runs a
# [OFF:OFF+N] window of the failed list (N=0 = full set); the ledger
# CUMULATES rescued/n across rounds, so OFF=0 N=22 + OFF=22 N=21 covers
# the whole set in two budget-respecting rounds. Step 0 (failed-set
# derivation) runs automatically when failed_dp.json is missing (~10 min,
# unguided eval); force with PREP_ONLY=1.
set -uo pipefail
ROUND=${ROUND:?set ROUND=<iteration round >=1>}
OFF=${OFF:-0}
N=${N:-22}
GAE_GAMMA=${GAE_GAMMA:-0.9}
GAE_NORM=${GAE_NORM:-1}
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
TH=/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy/CAN-entropy-s233/can
DP=$TH/train/DP/DP-base/checkpoints/599.ckpt
VIB=$TH/train/dyn/dyn-base/20260824-232156/scout_vib.ckpt
OFFICIAL=/root/workspace/baojiachun/scout/data/robomimic/can/ph/image_v141_abs.hdf5
ROOT=data/gae_probe_can824
CORE=$ROOT/can_core.hdf5
FAILED=$ROOT/failed_dp.json
T=$ROOT/r${ROUND}
[ -f "$DP" ] && [ -f "$VIB" ] && [ -f "$OFFICIAL" ] \
  || { echo "[gae-probe] FATAL: campaign ckpt / official dataset missing"; exit 1; }
mkdir -p "$ROOT" "$T/log"

# ---- step 0a: rebuild the seeded 20/200 core (deterministic, TSEED 233) -- #
if [ ! -f "$CORE" ]; then
  echo "[gae-probe] rebuilding core: split_core 20/200 seed 233"
  $PY scripts/analysis/split_core.py "$OFFICIAL" "$CORE" 20 233 \
    || { echo "[gae-probe] FATAL: split_core failed"; exit 1; }
fi

export MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- step 0b: derive the DP-base failed set (once) ----------------------- #
if [ ! -f "$FAILED" ]; then
  echo "[gae-probe] deriving failed set: DP-base eval seed42 x100 (GPU$GPU_GAE)"
  mkdir -p "$ROOT/evalonly"
  timeout -k 60 2400 env CUDA_VISIBLE_DEVICES=$GPU_GAE \
    SCOUT_RENDER_GPU=$GPU_GAE \
    $PY -m scout.eval.run_rollout \
      --config configs/eval_can_entropy.yaml --task can --exp-num 0 \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
      --guide off --explore-mode rescue --eval-only \
      --save-failed-set "$FAILED" \
      --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
      --output-dir "$ROOT/evalonly" \
      --output-json "$ROOT/evalonly/eval.json" \
    > "$ROOT/evalonly.stdout" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$FAILED" ] \
    || { echo "[gae-probe] FATAL: failed-set derivation rc=$rc"; exit 1; }
fi
$PY - "$FAILED" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
n = len(spec["failed_init_indices"])
print(f"[gae-probe] failed set: {n} inits, recorded SR "
      f"{spec.get('baseline_solved')}/100 "
      f"(historical r1 reference: 38 failed / SR .62)")
PYEOF
[ "${PREP_ONLY:-0}" = "1" ] && { echo "[gae-probe] PREP_ONLY done"; exit 0; }

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
print(f"[gae-probe] window [{'0' if n <= 0 else str(off)}:{'all' if n <= 0 else str(off+n)}] = {idx}")
PYEOF
WIN=$T/win.json

# ---- per-arm config copies (dose is config-only, th_p10_probe pattern) --- #
for spec in "aty:${ATY_SCALE}" "gae:${ATY_SCALE}"; do
  arm=${spec%%:*}; sc=${spec#*:}
  $PY - "configs/eval_can_entropy.yaml" "$T/cfg_${arm}.yaml" "$sc" "$GST" <<'PYEOF'
import sys, yaml
src, out, sc, gst = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["exploration"]["guidance_scale"] = float(sc)
cfg["exploration"]["guidance_start_timestep"] = int(gst)
cfg["wandb"]["use_wandb"] = False
cfg["wandb"]["project"] = "GAElike-p824probe"
cfg["wandb"]["tags"] = ["gaeprobe"]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[gae-probe-cfg] {out}: guidance_scale={sc} gst={gst}")
PYEOF
done

# ---- one arm = P shard workers on one GPU + merge, backstop-capped ------- #
run_arm() { # name gpu guide extra...
  local name=$1 gpu=$2 guide=$3; shift 3
  local extra=("$@")
  echo "[gae-probe] arm=$name gpu=$gpu guide=$guide P=$P extra=${extra[*]:-} (backstop ${BACKSTOP}s)"
  local t0=$(date +%s)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: timeout -k 60 $BACKSTOP env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu PYTHON=$PY CLEANUP_SHARDS=0 bash scripts/infra/shard_rollout.sh $P $T/$name/log/explore.json $T/$name/success.hdf5 $T/$name/all.hdf5 $CORE -- --config $T/cfg_${name}.yaml --task can --exp-num $ROUND --base-dp-ckpt $DP --core-hdf5 $CORE --vib-ckpt $VIB --guide $guide ${extra[@]+"${extra[@]}"} --explore-mode rescue --explore-try-times $TRIES --failed-set-json $WIN --n-envs 25 --seed 42 --eval-seed 42 --no-wandb --output-dir $T/$name --output-success $T/$name/success.hdf5 --output-all $T/$name/all.hdf5 > $T/$name.stdout 2>&1"
    return 0
  fi
  mkdir -p "$T/$name/log"
  timeout -k 60 "$BACKSTOP" env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu \
    PYTHON="$PY" CLEANUP_SHARDS=0 \
    bash scripts/infra/shard_rollout.sh "$P" \
      "$T/$name/log/explore.json" "$T/$name/success.hdf5" "$T/$name/all.hdf5" \
      "$CORE" -- \
      --config "$T/cfg_${name}.yaml" --task can --exp-num "$ROUND" \
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
    local pat="gae_probe_can824/r${ROUND}/${name}"
    pkill -f "$pat" 2>/dev/null && sleep 3
  fi
  local t1=$(date +%s)
  echo "$rc" > "$T/$name.rc"
  echo "[gae-probe] arm=$name rc=$rc wall=$(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
}

echo "[gae-probe] ROUND=$ROUND OFF=$OFF N=$N P=$P TRIES=$TRIES aty(scale/cap/gst)=$ATY_SCALE/$CAP/$GST gae(gamma/norm)=$GAE_GAMMA/$GAE_NORM gpus aty/gae=$GPU_ATY/$GPU_GAE"
GAE_ARGS=(--atypical-cap "$CAP" --gae-gamma "$GAE_GAMMA" --gae-norm "$GAE_NORM")
# extra gae-arm CLI flags (reflection-round config sweep, whitespace-split):
# e.g. GAE_EXTRA="--gae-agg add --gae-hist-weight 0.15"
if [ -n "${GAE_EXTRA:-}" ]; then
  read -ra _GE <<< "$GAE_EXTRA"
  GAE_ARGS+=("${_GE[@]}")
fi
run_arm gae "$GPU_GAE" gaelike ${GAE_ARGS[@]+"${GAE_ARGS[@]}"} &
GAE_PID=$!
run_arm aty "$GPU_ATY" atypical --atypical-cap "$CAP" &
ATY_PID=$!
wait "$GAE_PID"; wait "$ATY_PID"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[gae-probe] DRY_RUN done"; exit 0; }

# ---- summary: pass@10 per arm + telemetry tails -------------------------- #
sleep 2
$PY - "$T" "$ROUND" "$GAE_GAMMA" "$GAE_NORM" "$ROOT/ledger.csv" <<'PYEOF'
import csv, glob, json, os, re, sys
T, rnd, gamma, norm, ledger = sys.argv[1:6]
rows = []
for arm in ("aty", "gae"):
    row = {"round": rnd, "arm": arm,
           "params": (f"g{gamma},n{norm}" if arm == "gae" else "aty"),
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
    print(f"[gae-probe] {arm}: {row}")
aty = next((r for r in rows if r["arm"] == "aty"), {})
gae = next((r for r in rows if r["arm"] == "gae"), {})
try:
    a, g = float(aty["pass_at_10"]), float(gae["pass_at_10"])
    verdict = "GAElike WINS" if g > a else ("TIE" if g == a else "SCOUT wins")
    print(f"[gae-probe] VERDICT: {verdict} (pass@10 gae={g} vs aty={a})")
except (TypeError, ValueError, KeyError):
    print("[gae-probe] VERDICT: INCOMPLETE (missing merged json)")
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
    print("[gae-probe] CUMULATIVE (all rounds so far): " + " ".join(
        f"{a}={s}/{n} rescue_rate={s/max(n,1):.3f}"
        for a, (s, n) in sorted(acc.items())))
PYEOF

# ---- post-hoc render-integrity check (report-only) ----------------------- #
for arm in aty gae; do
  if [ -f "$T/$arm/all.hdf5" ]; then
    $PY scripts/viz/vis_validate.py "$T/$arm/all.hdf5" \
      > "$T/$arm.vischeck.txt" 2>&1 \
      && echo "[gae-probe] $arm render: $(tail -1 $T/$arm.vischeck.txt)" \
      || echo "[gae-probe] $arm render: CHECK FAILED (see $T/$arm.vischeck.txt)"
  fi
done
echo "[gae-probe] ROUND $ROUND DONE $(date '+%F %T')"
