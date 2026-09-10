#!/bin/bash
# th94_walkcloud_probe.sh -- WalkCloud vs SCOUT(atypical) pass@5 A/B on
# the CAN-8-24-entropy-s233 round0 (DP-base 599.ckpt + dyn-base)
# frozen-failure-set (user 2026-09-10, idea/walkcloud_plan.md §5, branch
# drift-dev).
#
# Protocol (walkcloud_plan §5, mirrors the 2026_9_10_drift_probe campaign):
#   * failed set = the TH9-4 round0 r1 DP-base eval failures (69 failed /
#     SR .31, rollout/DP-exp1/failed.json), pre-copied into the probe root;
#     step 0 re-derives only if that copy is missing (~30 min unguided eval).
#   * both arms: P=4 CPU shard workers, n_envs=25, ETRIES=5 (pass@5 --
#     matches the drift campaign's aty baseline 14/43 for direct
#     comparison), eta 3.0 / kappa 2.5 / gst 100 -- IDENTICAL except
#     --guide atypical vs walkcloud.
#   * walkcloud consumes ZERO guidance RNG and its t=1/t=2 are bitwise
#     atypical (empty cloud / {anchor} fast path; CPU bitwise, GPU may
#     differ at ulp -- same caveat as _kl_rows), so the two arms share
#     RNG streams exactly -- a PAIRED A/B on the same seeds (unlike
#     cloudrep, whose start gate forked the streams).
#   * dose sanity check: wc mean_inject vs aty mean_inject in the ledger
#     (soft-min renormalizes the gradient mix; S <= min_j KL_j) --
#     divergence >~3x means shared eta is NOT shared displacement: rerun
#     with --aty-eta-dimless (eta_dimless) for a displacement-matched arm.
#   * WINDOWED (toolhang horizon 700: 69 scenes x 5 tries does not fit one
#     45-min round at ~1.9 guided trajs/min/worker): default N=48; cover
#     with ROUND=1 OFF=0 N=48 + ROUND=2 OFF=48 N=21 (ledger CUMULATES).
#   * pre-registered falsification (plan §5): can <= 14 with normal
#     mean_S telemetry -> the i-axis falsification extends to KERNEL
#     aggregation, stop. Watch jerk (self-repulsion may jitter the walk
#     tail).
#   * 45-min hard backstop per arm (timeout -k 60 2700); orphans pkill'd
#     by the unique probe path pattern.
#   * read-only reuse of the campaign's train/ ckpts; outputs ONLY under
#     data/walkcloud_probe_can824/. Never touches campaign data or
#     running chains.
#   * readout discipline (drift campaign lesson): the MERGED explore.json
#     has produced bad values (r4 aty reported 0, shard grep said 14) --
#     authoritative = the sum of per-shard exploration_rescued greps.
#
# usage: ROUND=1 [OFF=0] [N=48] [WC_TAU_MODE=adapt] [WC_TAU=0.5]
#        [WC_HIST_MAX=0] [P=4] [TRIES=5] [ATY_SCALE=3.0] [CAP=2.5] [GST=100]
#        [GPU_ATY=3] [GPU_WC=1] [BACKSTOP=2700] [SCOUT_DIR=/tmp/scout-drift]
#        [WC_EXTRA="--cloud-tau-frac 0.15"] [DRY_RUN=1]
# Step 0 (failed-set derivation) runs automatically when failed_dp.json is
# missing (~10 min, unguided eval); force with PREP_ONLY=1.
set -uo pipefail
ROUND=${ROUND:?set ROUND=<iteration round >=1>}
OFF=${OFF:-0}
N=${N:-48}
WC_TAU_MODE=${WC_TAU_MODE:-adapt}
WC_TAU=${WC_TAU:-0.5}
WC_HIST_MAX=${WC_HIST_MAX:-0}
P=${P:-4}
TRIES=${TRIES:-5}
ATY_SCALE=${ATY_SCALE:-3.0}
CAP=${CAP:-2.5}
GST=${GST:-100}
GPU_ATY=${GPU_ATY:-3}
GPU_WC=${GPU_WC:-1}
BACKSTOP=${BACKSTOP:-2700}
SCOUT_DIR=${SCOUT_DIR:-/tmp/scout-drift}

cd "$SCOUT_DIR" || { echo "[th-wc-probe] FATAL: scout dir $SCOUT_DIR missing"; exit 1; }
PY=/root/workspace/baojiachun/.venv/bin/python
TH=/root/workspace/baojiachun/scout-orbit/data/2026_9_4_toolhang/TOOLHANG-s233/tool_hang
DP=$TH/train/DP/DP-base/checkpoints/599.ckpt
VIB=$TH/train/dyn/dyn-base/20260904-170403/scout_vib.ckpt
OFFICIAL=/root/workspace/baojiachun/scout/data/robomimic/tool_hang/ph/image_v141_abs.hdf5
ROOT=data/walkcloud_probe_th94
CORE=/root/workspace/baojiachun/scout/data/robomimic/tool_hang/ph/image_v141_abs_core20.hdf5
FAILED=$ROOT/failed_dp.json
T=$ROOT/r${ROUND}
[ -f "$DP" ] && [ -f "$VIB" ] && [ -f "$OFFICIAL" ] \
  || { echo "[th-wc-probe] FATAL: campaign ckpt / official dataset missing"; exit 1; }
mkdir -p "$ROOT" "$T/log"

# ---- step 0a: rebuild the seeded 20/200 core (deterministic, TSEED 233) -- #
if [ ! -f "$CORE" ]; then
  echo "[th-wc-probe] rebuilding core: split_core 20/200 seed 233"
  $PY scripts/analysis/split_core.py "$OFFICIAL" "$CORE" 20 233 \
    || { echo "[th-wc-probe] FATAL: split_core failed"; exit 1; }
fi

export MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- step 0b: derive the DP-base failed set (once) ----------------------- #
if [ ! -f "$FAILED" ]; then
  echo "[th-wc-probe] deriving failed set: DP-base eval seed42 x100 (GPU$GPU_WC)"
  mkdir -p "$ROOT/evalonly"
  timeout -k 60 2400 env CUDA_VISIBLE_DEVICES=$GPU_WC \
    SCOUT_RENDER_GPU=$GPU_WC \
    $PY -m scout.eval.run_rollout \
      --config configs/eval_tool_hang_entropy.yaml --task tool_hang --exp-num 0 \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" \
      --guide off --explore-mode rescue --eval-only \
      --save-failed-set "$FAILED" \
      --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
      --output-dir "$ROOT/evalonly" \
      --output-json "$ROOT/evalonly/eval.json" \
    > "$ROOT/evalonly.stdout" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$FAILED" ] \
    || { echo "[th-wc-probe] FATAL: failed-set derivation rc=$rc"; exit 1; }
fi
$PY - "$FAILED" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
n = len(spec["failed_init_indices"])
print(f"[th-wc-probe] failed set: {n} inits, recorded SR "
      f"{spec.get('baseline_solved')}/100 "
      f"(TH9-4 round0 r1 reference: 69 failed / SR .31)")
PYEOF
[ "${PREP_ONLY:-0}" = "1" ] && { echo "[th-wc-probe] PREP_ONLY done"; exit 0; }

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
print(f"[th-wc-probe] window [{'0' if n <= 0 else str(off)}:{'all' if n <= 0 else str(off+n)}] = {idx}")
PYEOF
WIN=$T/win.json

# ---- per-arm config copies (dose is config-only, th_p10_probe pattern) --- #
for spec in "aty:${ATY_SCALE}" "wc:${ATY_SCALE}"; do
  arm=${spec%%:*}; sc=${spec#*:}
  $PY - "configs/eval_tool_hang_entropy.yaml" "$T/cfg_${arm}.yaml" "$sc" "$GST" <<'PYEOF'
import sys, yaml
src, out, sc, gst = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["exploration"]["guidance_scale"] = float(sc)
cfg["exploration"]["guidance_start_timestep"] = int(gst)
cfg["wandb"]["use_wandb"] = False
cfg["wandb"]["project"] = "WALKCLOUD-p824probe"
cfg["wandb"]["tags"] = ["wcprobe"]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[wc-probe-cfg] {out}: guidance_scale={sc} gst={gst}")
PYEOF
done

# ---- one arm = P shard workers on one GPU + merge, backstop-capped ------- #
run_arm() { # name gpu guide extra...
  local name=$1 gpu=$2 guide=$3; shift 3
  local extra=("$@")
  echo "[th-wc-probe] arm=$name gpu=$gpu guide=$guide P=$P extra=${extra[*]:-} (backstop ${BACKSTOP}s)"
  local t0=$(date +%s)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: timeout -k 60 $BACKSTOP env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu PYTHON=$PY CLEANUP_SHARDS=0 bash scripts/infra/shard_rollout.sh $P $T/$name/log/explore.json $T/$name/success.hdf5 $T/$name/all.hdf5 $CORE -- --config $T/cfg_${name}.yaml --task tool_hang --exp-num $ROUND --base-dp-ckpt $DP --core-hdf5 $CORE --vib-ckpt $VIB --guide $guide ${extra[@]+"${extra[@]}"} --explore-mode rescue --explore-try-times $TRIES --failed-set-json $WIN --n-envs 25 --seed 42 --eval-seed 42 --no-wandb --output-dir $T/$name --output-success $T/$name/success.hdf5 --output-all $T/$name/all.hdf5 > $T/$name.stdout 2>&1"
    return 0
  fi
  mkdir -p "$T/$name/log"
  timeout -k 60 "$BACKSTOP" env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu \
    PYTHON="$PY" CLEANUP_SHARDS=0 \
    bash scripts/infra/shard_rollout.sh "$P" \
      "$T/$name/log/explore.json" "$T/$name/success.hdf5" "$T/$name/all.hdf5" \
      "$CORE" -- \
      --config "$T/cfg_${name}.yaml" --task tool_hang --exp-num "$ROUND" \
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
    local pat="walkcloud_probe_th94/r${ROUND}/${name}"
    pkill -f "$pat" 2>/dev/null && sleep 3
  fi
  local t1=$(date +%s)
  echo "$rc" > "$T/$name.rc"
  echo "[th-wc-probe] arm=$name rc=$rc wall=$(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
}

echo "[th-wc-probe] ROUND=$ROUND OFF=$OFF N=$N P=$P TRIES=$TRIES aty(scale/cap/gst)=$ATY_SCALE/$CAP/$GST wc(tau_mode/tau/hist_max)=$WC_TAU_MODE/$WC_TAU/$WC_HIST_MAX gpus aty/wc=$GPU_ATY/$GPU_WC"
WC_ARGS=(--atypical-cap "$CAP" --cloud-tau-mode "$WC_TAU_MODE"
         --cloud-tau "$WC_TAU" --cloud-hist-max "$WC_HIST_MAX")
# extra walkcloud-arm CLI flags (reflection-round config sweep, whitespace-split):
# e.g. WC_EXTRA="--cloud-tau-frac 0.15 --cloud-tau-min 0.01"
if [ -n "${WC_EXTRA:-}" ]; then
  read -ra _WE <<< "$WC_EXTRA"
  WC_ARGS+=("${_WE[@]}")
fi
run_arm wc "$GPU_WC" walkcloud ${WC_ARGS[@]+"${WC_ARGS[@]}"} &
WC_PID=$!
run_arm aty "$GPU_ATY" atypical --atypical-cap "$CAP" &
ATY_PID=$!
wait "$WC_PID"; wait "$ATY_PID"
[ "${DRY_RUN:-0}" = "1" ] && { echo "[th-wc-probe] DRY_RUN done"; exit 0; }

# ---- summary: pass@5 per arm + shard-grep hard tally + telemetry tails --- #
sleep 2
$PY - "$T" "$ROUND" "$WC_TAU_MODE" "$WC_TAU" "$WC_HIST_MAX" "$ROOT/ledger.csv" <<'PYEOF'
import csv, glob, json, os, re, sys
T, rnd, tau_mode, tau, hmax, ledger = sys.argv[1:7]
rows = []
for arm in ("aty", "wc"):
    row = {"round": rnd, "arm": arm,
           "params": (f"{tau_mode},t{tau},h{hmax}" if arm == "wc" else "aty"),
           "tau_mode": tau_mode, "tau": tau, "hist_max": hmax}
    rc_f = f"{T}/{arm}.rc"
    row["rc"] = open(rc_f).read().strip() if os.path.exists(rc_f) else "?"
    js = f"{T}/{arm}/log/explore.json"
    if os.path.exists(js):
        d = json.load(open(js))
        row["rescued"] = d.get("exploration_rescued")
        row["n"] = d.get("n_failed")
        row["p5_rescue_rate"] = (round(d["exploration_rescued"] / d["n_failed"], 3)
                                 if d.get("n_failed") else None)
        row["pass_at_5"] = (round((d.get("baseline_solved", 0)
                                   + d["exploration_rescued"]) / 100, 3)
                            if d.get("n_failed") is not None else None)
        row["jerk"] = round(d.get("avg_jerk") or 0, 4)
        row["collected"] = d.get("collected_trajs")
    else:
        for k in ("rescued", "n", "p5_rescue_rate", "pass_at_5", "jerk"):
            row[k] = "NOJSON"
        row["collected"] = ""
    # hard tally: merged json has lied before (drift r4) -- recompute from
    # the per-shard jsons with a grep-style sum (shard workers write
    # <stem>-shard{i}of{P}.json next to the merged explore.json)
    shard_rescued = []
    for p in sorted(glob.glob(f"{T}/{arm}/log/*-shard*.json")):
        try:
            sd = json.load(open(p))
            if sd.get("exploration_rescued") is not None:
                shard_rescued.append(int(sd["exploration_rescued"]))
        except (json.JSONDecodeError, OSError):
            pass
    row["shard_tally"] = sum(shard_rescued) if shard_rescued else ""
    inject = s_tel = h_tel = None
    for p in sorted(glob.glob(f"{T}/{arm}/log/shard*.stdout")):
        txt = open(p, errors="ignore").read()
        m = re.findall(r"mean_inject=([0-9.eE+-]+)", txt)
        if m: inject = m[-1]
        m = re.findall(r"mean_S=([0-9.eE+-]+)", txt)
        if m: s_tel = m[-1]
        m = re.findall(r"mean_hist_len=([0-9.eE+-]+)", txt)
        if m: h_tel = m[-1]
    row["mean_inject"] = inject or ""
    row["wc_mean_S"] = s_tel or ""
    row["wc_hist_len"] = h_tel or ""
    rows.append(row)
    print(f"[th-wc-probe] {arm}: {row}")
aty = next((r for r in rows if r["arm"] == "aty"), {})
wc = next((r for r in rows if r["arm"] == "wc"), {})

def _n(r, k):
    v = r.get(k)
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str) and v != "" and v != "NOJSON":
        try:
            return float(v)
        except ValueError:
            pass
    return None

a_r, w_r = _n(aty, "shard_tally") or _n(aty, "rescued"), \
           _n(wc, "shard_tally") or _n(wc, "rescued")
if w_r is not None and a_r is not None:
    partial = any(str(r.get("rc")) not in ("0", "?") for r in (aty, wc))
    verdict = "WALKCLOUD WINS" if w_r > a_r else ("TIE" if w_r == a_r
                                                  else "SCOUT(aty) wins")
    if partial:
        verdict += " (BACKSTOP/PARTIAL -- shard tally is a lower bound; " \
                   "do NOT act on the falsification line)"
    print(f"[th-wc-probe] VERDICT: {verdict} (rescued@5 wc={w_r:g} vs aty={a_r:g})")
    if w_r <= 14 and not partial:
        print("[th-wc-probe] PRE-REGISTERED FALSIFICATION LINE: wc<=14 -- if "
              "mean_S telemetry is normal, the i-axis falsification "
              "EXTENDS to kernel aggregation (walkcloud_plan §5): STOP.")
else:
    print("[th-wc-probe] VERDICT: INCOMPLETE (no rescued readout)")
new = not os.path.exists(ledger)
with open(ledger, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new: w.writeheader()
    w.writerows(rows)
from collections import defaultdict
if not new:
    # re-running the same ROUND (config iteration loop) appends duplicate
    # rows -- count only the LAST row per (round, arm, params)
    last = {}
    with open(ledger) as f:
        for r in csv.DictReader(f):
            last[(r["round"], r["arm"], r["params"])] = r
    acc = defaultdict(lambda: [0, 0])
    for r in last.values():
        try:
            acc[r["arm"]][0] += int(r["rescued"])
            acc[r["arm"]][1] += int(r["n"])
        except (TypeError, ValueError, KeyError):
            pass
    print("[th-wc-probe] CUMULATIVE (last row per round/arm/params): " + " ".join(
        f"{a}={s}/{n} rescue_rate={s/max(n,1):.3f}"
        for a, (s, n) in sorted(acc.items())))
PYEOF

# ---- post-hoc render-integrity check (report-only) ----------------------- #
for arm in aty wc; do
  if [ -f "$T/$arm/all.hdf5" ]; then
    $PY scripts/viz/vis_validate.py "$T/$arm/all.hdf5" \
      > "$T/$arm.vischeck.txt" 2>&1 \
      && echo "[th-wc-probe] $arm render: $(tail -1 $T/$arm.vischeck.txt)" \
      || echo "[th-wc-probe] $arm render: CHECK FAILED (see $T/$arm.vischeck.txt)"
  fi
done
echo "[th-wc-probe] ROUND $ROUND DONE $(date '+%F %T')"
