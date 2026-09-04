#!/bin/bash
# th_p10_probe.sh -- TOOLHANG approx-pass@10 iteration probe (user /goal 2026-09-04).
# Goal: atypical & orbit beat DP on round-1 explore (DP full-set pass@10 = 0.52,
# i.e. 41/89 on the failed set); our bar = approx pass@10 >= 0.6.
#
# One round = one rotating window of N failed inits x 10 tries, THREE arms in
# parallel on three GPUs: DP (guide=off) / atypical / orbit. User order
# 2026-09-04: P=3 shard workers per arm (single worker measured ~2.4
# rollouts/min at n_envs=25 -- lockstep slots are NOT a multiplier), so a
# 20-min budget fits ~90 rollouts = N=9 envs x 10 tries. Hard backstop
# `timeout -k 60 1500` per arm; on timeout the arm's orphan workers are
# pkill'd by their unique path pattern.
# approx pass@10 (arm, round) = exploration_rescued / n_failed (merged shard
# json; every window init completes its full 10 tries). Cumulative per arm =
# sum(rescued)/sum(N) across rounds; windows rotate failed[OFF:OFF+N] so the
# cumulative ratio estimates the full-set pass@10. DP's r1 per-init outcomes
# on each window are printed alongside (post-hoc from r1 explore_detail).
#
# Read-only reuse of the r1 frozen trio + SCOUT-exp1/failed.json; outputs only
# under data/2026_9_4_p10probe/. Never touches campaign data or chains.
# usage: ROUND=4 [OFFSET=9] [N=9] [P=3] [ATY_SCALE=12] [ORB_ETA=1.2]
#        [ORB_SIGMA=0.025] [ORB_LAM=0.15] [ORB_DELTA=0.25] [ORB_FBCLAMP=none]
#        [TRIES=10] [DRY_RUN=1] GPU_DP=3 GPU_ATY=2 GPU_ORB=1
#        bash soe_scripts/th_p10_probe.sh
set -uo pipefail
ROUND=${ROUND:?set ROUND=<iteration round >=1>}
N=${N:-9}
OFF=${OFF:-$(( (ROUND-1)*N ))}
P=${P:-3}
ATY_SCALE=${ATY_SCALE:-12.0}
ORB_ETA=${ORB_ETA:-1.2}
ORB_SIGMA=${ORB_SIGMA:-0.025}
ORB_LAM=${ORB_LAM:-0.15}
ORB_DELTA=${ORB_DELTA:-0.25}
ORB_FBCLAMP=${ORB_FBCLAMP:-none}
ATY_CAP=${ATY_CAP:-2.5}
ORB_CAP=${ORB_CAP:-2.5}
ORB_DIMLESS=${ORB_DIMLESS:-0}
TRIES=${TRIES:-10}
GPU_DP=${GPU_DP:-3}; GPU_ATY=${GPU_ATY:-2}; GPU_ORB=${GPU_ORB:-1}
WPROJ=${WPROJ:-TOOLHANG-9-4-p10probe}
BACKSTOP=${BACKSTOP:-1500}

cd /root/workspace/baojiachun/scout-orbit || exit 1
PY=/root/workspace/baojiachun/.venv/bin/python
TH=data/2026_9_1_toolhang/TOOLHANG-s233/tool_hang
DP=$TH/train/DP/DP-base/checkpoints/599.ckpt
VIB=$(ls -t $TH/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$TH/rollout/tool_hang_core.hdf5
FAILED=$TH/rollout/SCOUT-exp1/failed.json
ROOT=data/2026_9_4_p10probe
T=$ROOT/r${ROUND}
[ -f "$FAILED" ] && [ -f "$CORE" ] && [ -n "$DP" ] && [ -n "$VIB" ] \
  || { echo "[p10] FATAL: trio/failed.json missing"; exit 1; }
mkdir -p "$T/log" "$ROOT"

# ---- window cut: failed inits [OFF:OFF+N], original indices kept -----------
$PY - "$FAILED" "$T/win.json" "$OFF" "$N" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
off, n = int(sys.argv[3]), int(sys.argv[4])
allf = spec["failed_init_indices"]
idx = allf[off:off + n]
assert len(idx) == n, f"window underflow: wants [{off}:{off+n}] of {len(allf)}"
spec["failed_init_indices"] = idx
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print(f"[p10] window [{off}:{off+n}] = {idx}")
PYEOF

# ---- per-arm config copies (dose has NO CLI flag: config-only entry) -------
for spec in "dp:0.0" "aty:${ATY_SCALE}" "orb:${ORB_ETA}"; do
  arm=${spec%%:*}; sc=${spec#*:}
  $PY - "configs/eval_tool_hang_entropy.yaml" "$T/cfg_${arm}.yaml" "$sc" "$WPROJ" <<'PYEOF'
import sys, yaml
src, out, sc, proj = sys.argv[1:5]
with open(src) as f:
    cfg = yaml.safe_load(f)
cfg["exploration"]["guidance_scale"] = float(sc)
cfg["wandb"]["project"] = proj
cfg["wandb"]["tags"] = ["p10probe"]
with open(out, "w") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"[p10-cfg] {out}: exploration.guidance_scale={sc} wandb.project={proj}")
PYEOF
done

export MUJOCO_GL=egl TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

# ---- one arm = P shard workers on one GPU + merge, backstop-capped ---------
run_arm() { # name gpu guide vib extra...
  local name=$1 gpu=$2 guide=$3 usevib=$4; shift 4
  local extra=("$@")
  local vibargs=()
  [ "$usevib" = "vib" ] && vibargs=(--vib-ckpt "$VIB")
  echo "[p10] arm=$name gpu=$gpu guide=$guide P=$P extra=${extra[*]:-} (backstop ${BACKSTOP}s)"
  local t0=$(date +%s)
  if [ "${DRY_RUN:-0}" = "1" ]; then
    echo "DRY: timeout -k 60 $BACKSTOP env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu PYTHON=$PY CLEANUP_SHARDS=0 bash soe_scripts/shard_rollout.sh $P $T/$name/log/explore.json $T/$name/success.hdf5 $T/$name/all.hdf5 $CORE -- --config $T/cfg_${name}.yaml --task tool_hang --exp-num $ROUND --base-dp-ckpt $DP --core-hdf5 $CORE ${vibargs[@]+"${vibargs[@]}"} --guide $guide ${extra[@]+"${extra[@]}"} --explore-mode rescue --explore-try-times $TRIES --failed-set-json $T/win.json --n-envs 25 --seed 42 --eval-seed 42 --no-wandb --output-dir $T/$name --output-success $T/$name/success.hdf5 --output-all $T/$name/all.hdf5 > $T/$name.stdout 2>&1"
    return 0
  fi
  mkdir -p "$T/$name/log"
  timeout -k 60 "$BACKSTOP" env CUDA_VISIBLE_DEVICES=$gpu SCOUT_RENDER_GPU=$gpu \
    PYTHON="$PY" CLEANUP_SHARDS=0 \
    bash soe_scripts/shard_rollout.sh "$P" \
      "$T/$name/log/explore.json" "$T/$name/success.hdf5" "$T/$name/all.hdf5" \
      "$CORE" -- \
      --config "$T/cfg_${name}.yaml" --task tool_hang --exp-num "$ROUND" \
      --base-dp-ckpt "$DP" --core-hdf5 "$CORE" ${vibargs[@]+"${vibargs[@]}"} \
      --guide "$guide" ${extra[@]+"${extra[@]}"} \
      --explore-mode rescue --explore-try-times "$TRIES" \
      --failed-set-json "$T/win.json" \
      --n-envs 25 --seed 42 --eval-seed 42 \
      --no-wandb \
      --output-dir "$T/$name" --output-success "$T/$name/success.hdf5" \
      --output-all "$T/$name/all.hdf5" \
      > "$T/$name.stdout" 2>&1
  local rc=$?
  if [ "$rc" = "124" ]; then
    # backstop fired: kill this arm-round's orphaned workers (pattern is
    # unique to these shard argv paths; built here so no ssh line matches it)
    local pat="p10probe/r${ROUND}/${name}"
    pkill -f "$pat" 2>/dev/null && sleep 3
  fi
  local t1=$(date +%s)
  echo "$rc" > "$T/$name.rc"
  echo "[p10] arm=$name rc=$rc wall=$(( (t1-t0)/60 ))m$(( (t1-t0)%60 ))s"
}

echo "[p10] ROUND=$ROUND OFF=$OFF N=$N P=$P aty(scale/cap)=$ATY_SCALE/$ATY_CAP orb(eta/sigma/lam/delta/fbclamp/cap/dimless)=$ORB_ETA/$ORB_SIGMA/$ORB_LAM/$ORB_DELTA/$ORB_FBCLAMP/$ORB_CAP/$ORB_DIMLESS gpus dp/aty/orb=$GPU_DP/$GPU_ATY/$GPU_ORB"
ORB_ARGS=(--atypical-cap "$ORB_CAP" --orbit-lam "$ORB_LAM" \
           --orbit-delta "$ORB_DELTA" --orbit-sigma "$ORB_SIGMA" \
           --orbit-fb-clamp "$ORB_FBCLAMP")
[ "$ORB_DIMLESS" = "1" ] && ORB_ARGS+=(--orbit-eta-dimless)
run_arm orb  "$GPU_ORB" orbit     vib ${ORB_ARGS[@]+"${ORB_ARGS[@]}"} &
P1=$!
run_arm aty  "$GPU_ATY" atypical  vib --atypical-cap "$ATY_CAP" &
P2=$!
run_arm dp   "$GPU_DP"  off       novib &
P3=$!
wait $P1 $P2 $P3
[ "${DRY_RUN:-0}" = "1" ] && { echo "[p10] DRY_RUN done"; exit 0; }

# ---- summary: approx pass@10 per arm + r1 DP reference on this window ------
sleep 2
$PY - "$T" "$ROUND" "$N" "$ATY_SCALE" "$ORB_ETA" "$ORB_SIGMA" "$ROOT/ledger.csv" "$TH" "$ATY_CAP" "$ORB_CAP" "$ORB_DIMLESS" <<'PYEOF'
import csv, glob, json, os, re, sys
T, rnd, N, aty_s, orb_e, orb_sig, ledger, TH = sys.argv[1:9]
aty_cap, orb_cap, orb_dimless = sys.argv[9:12]
r1 = json.load(open(f"{TH}/rollout/DP-exp1/log/tool_hang_DP_explore_exp1.json"))
win = json.load(open(f"{T}/win.json"))["failed_init_indices"]
r1_detail = {d["init"]: d for d in r1["explore_detail"]}
r1_solved = sum(1 for i in win if r1_detail.get(i, {}).get("solved"))
print(f"[p10] r1 DP reference on window: {r1_solved}/{N}")
rows = []
for arm in ("dp", "aty", "orb"):
    row = {"round": rnd, "arm": arm,
           "params": (f"s{aty_s},cap{aty_cap}" if arm == "aty" else
                      f"eta{orb_e},sig{orb_sig},cap{orb_cap},dimless{orb_dimless}"
                      if arm == "orb" else "none"),
           "window": " ".join(map(str, win))}
    rc_f = f"{T}/{arm}.rc"
    row["rc"] = open(rc_f).read().strip() if os.path.exists(rc_f) else "?"
    js = f"{T}/{arm}/log/explore.json"
    if os.path.exists(js):
        d = json.load(open(js))
        row["rescued"] = d.get("exploration_rescued")
        row["n"] = d.get("n_failed")
        row["approx_p10"] = (round(d["exploration_rescued"] / d["n_failed"], 3)
                             if d.get("n_failed") else None)
        row["jerk"] = round(d.get("avg_jerk") or 0, 4)
        row["collected"] = d.get("collected_trajs")
    else:
        row["rescued"] = row["n"] = row["approx_p10"] = row["jerk"] = "NOJSON"
        row["collected"] = ""
    # telemetry tails from shard stdouts (mean_inject; orbit: |fb| / |noise|)
    inject = noise = None
    for p in sorted(glob.glob(f"{T}/{arm}/log/shard*.stdout")):
        txt = open(p, errors="ignore").read()
        m = re.findall(r"mean_inject=([0-9.eE+-]+)", txt)
        if m: inject = m[-1]
        m = re.findall(r"mean\|noise\|/p2row=([0-9.eE+-]+)", txt)
        if m: noise = m[-1]
    row["mean_inject"] = inject or ""
    row["orb_mean_noise"] = noise or ""
    rows.append(row)
    print(f"[p10] {arm}: {row}")
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
                acc[r["arm"]][0] += int(r["rescued"]); acc[r["arm"]][1] += int(r["n"])
            except (TypeError, ValueError):
                pass
    print("[p10] CUMULATIVE approx pass@10: "
          + " ".join(f"{a}={s}/{n}={s/max(n,1):.3f}" for a, (s, n) in acc.items()))
PYEOF
echo "[p10] ROUND $ROUND DONE $(date '+%F %T')"
