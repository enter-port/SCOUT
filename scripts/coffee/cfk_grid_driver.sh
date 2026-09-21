#!/bin/bash
# cfk_grid_driver.sh -- AUTONOMOUS grid orchestrator for COFFEE-MG-p1 on
# the s233 base (2026-09-20, user orders: "grid 范围探针后自动定,不停";
# "eta 取两引导组 pass@5 之和最大,如果有并列你来选 不要汇报不要停").
# Lineage: scripts/coffee_prep/ckp_grid_driver.sh (threading grid machinery +
# the 2026-09-17 recalibration methodology: offline grad probe -> live dose
# probe -> full cells -> peak bracketing), encoded as a deterministic bounded
# loop.
#
# Stages (idempotent by artifacts; safe to re-run after a crash):
#   S1 grad probe  -> TELEMETRY/grad_probe_COFFEE-MG-p1-s233.json
#                     eta0 anchor = median(eta_for_0.1sd) over delta rows
#                     {0.05,0.1,0.25} with live gradients, clamp [2,512]
#   S2 preseed eval -> GRID_s233/PRESEED/ (canonical 100-scene UNGUIDED eval +
#                     frozen failed.json; eval never attaches the planner so
#                     this is identical for every cell; cells skip phase A)
#   S3 dose probes (ATY eta in {eta0/4, eta0, 4*eta0}, ~1/4 failure set x2
#      tries, parallel 3 GPUs) -> mean_inject telemetry; if ALL silent
#      (max mean_inject < 0.05) -> eta0 *= 8
#   S3b placebo cell (DP eval-pk, preseeded) -> unguided pass@5 reference
#   S4 coarse ATY batch: {eta0/2, eta0, 2*eta0, 4*eta0} x kappa {2.5, 10}
#   S5 adaptive loop (<=6 analysis rounds, ATY attempt cap 36):
#        - no row > placebo+0.01 AND injection silent -> eta0*=8, re-coarse (once)
#        - no row > placebo+0.01, injection alive -> kappa fill at peak eta
#        - peak at TOP eta edge and better than same-kappa interior -> extend {2x,4x}
#        - peak at BOTTOM eta edge (>=0.25) -> extend {/4,/2}
#        - interior peak -> kappa fill {2.5,5,10,15,20} at peak eta, then
#          refine {0.7x,1.4x} at kappa_peak, then done
#   S6 eta candidates: TOP-2 DISTINCT ATY etas (per-eta best row by
#      pass@5 -> lower jerk -> lower kappa)
#   S7 ORBIT batch: sigma {0.025,0.05,0.1,0.2} x lambda {0.25,0.5,1.0} at EACH
#      candidate (eta, kappa) -- 24 cells (the sum criterion needs ORBIT
#      evaluated at more than one eta, unlike the coffee_prep driver which
#      ran ORBIT only at the ATY peak)
#   S7.5 eta* by the SUM criterion (user, verbatim: "eta 取两引导组 pass@5
#      之和最大"): eta* = argmax_eta [aty_p5(eta) + orb_p5(eta)];
#      kappa* = the kappa of the per-eta best ATY row at eta*;
#      ties -> higher min(aty,orb) -> LOWER eta (agent-delegated 2026-09-20)
#   S7b if orb_max(eta*) < aty_max(eta*) -> extension sigma {0.0125, 0.3}
#      x lam_best at eta*
#   S8 verdict.json + verdict.txt: sigma*/lam* from ORBIT at eta*
#      (ties: LOWER sigma -- less retrain-data pollution, tp15; lower lam);
#      full candidate table + placebo + n_failed recorded.
#   Caps: ATY attempts <= 36, ORBIT <= 26, wall <= 40h; on cap -> verdict from
#   best-so-far. Runs ON 1022 (GPUs 0-6; GPU7 ECC-banned); cell wave = one
#   GPU each.
#
# usage: tmux new-session -d -s cfk_grid 'bash scripts/coffee/cfk_grid_driver.sh'
set -u
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_20_coffee_mg_p1
S233=$CAMP/COFFEE-MG-p1-s233
GRIDROOT=$CAMP/GRID_s233
PRESEED=$GRIDROOT/PRESEED
TELEMETRY=$CAMP/TELEMETRY
GPUS=${GPUS:-"0 1 2 3 4 5 6"}
NGPU=$(wc -w <<< "$GPUS")
mkdir -p "$GRIDROOT" "$TELEMETRY"
cd "$ROOT" || exit 1

log(){ echo "[$(date '+%F %T')] [driver] $*" | tee -a "$GRIDROOT/driver.log"; }

WGPU=($GPUS)

# filter_done SPECFILE CSV ARM -> drop specs whose CELL already has a valid row
filter_done(){
  $PY - "$2" "$1" "$3" <<'PYEOF'
import sys, csv, os
csv_path, spec_path, arm = sys.argv[1:4]
done = set()
if os.path.isfile(csv_path):
    for r in csv.DictReader(open(csv_path)):
        if r["arm"] == arm and r["rc"] == "0":
            done.add(r["cell"])
lines = [l for l in open(spec_path).read().splitlines()
         if l.strip() and l.split("CELL=")[-1] not in done]
open(spec_path, "w").write("\n".join(lines) + ("\n" if lines else ""))
PYEOF
}

# run_batch SPECFILE KIND -> waves of <= NGPU cells, one GPU each, wait per wave
run_batch(){
  local specfile=$1 kind=$2 logs=$GRIDROOT/batch_logs chunk i gpu
  [ -s "$specfile" ] || { log "batch $kind: empty spec file -- skip"; return 0; }
  mkdir -p "$logs"
  rm -rf "$logs/.chunk.$$"; mkdir -p "$logs/.chunk.$$"
  split -l "$NGPU" "$specfile" "$logs/.chunk.$$/"part_
  for chunk in "$logs/.chunk.$$/"part_*; do
    i=0
    while IFS= read -r spec; do
      [ -z "$spec" ] && continue
      gpu=${WGPU[$i]}
      log "cell start ($kind, gpu$gpu): $spec"
      if [ "$kind" = orbit ]; then
        ( env $spec GPU="$gpu" bash scripts/coffee/cfk_orbit_probe.sh ) \
          > "$logs/${kind}_gpu${gpu}_$$.log" 2>&1 &
      else
        ( env $spec GPU="$gpu" bash scripts/coffee/cfk_kgrid_probe.sh ) \
          > "$logs/${kind}_gpu${gpu}_$$.log" 2>&1 &
      fi
      i=$((i+1))
    done < "$chunk"
    wait
    log "wave done ($kind)"
  done
  rm -rf "$logs/.chunk.$$"
}

# --------------------------- S1: grad probe -------------------------------- #
GRADJSON=$TELEMETRY/grad_probe_COFFEE-MG-p1-s233.json
if [ ! -f "$GRADJSON" ]; then
  log "S1 grad probe start (gpu0)"
  timeout -k 30 3600 env CUDA_VISIBLE_DEVICES=0 SCOUT_RENDER_GPU=0 TMPDIR=/tmp \
    $PY scripts/coffee/cfk_grad_probe.py --seed-data-root "$S233" \
    >> "$GRIDROOT/grad_probe.stdout" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$GRADJSON" ] || { log "FATAL: grad probe rc=$rc"; exit 1; }
fi
log "S1 grad probe done"

# eta0 anchor from the ladder
ETA0=$($PY - "$GRADJSON" <<'PYEOF'
import sys, json, statistics
d = json.load(open(sys.argv[1]))
rows = [r for r in d["ladder"]
        if r["delta_rel"] in (0.05, 0.1, 0.25) and r["n_live"] > 0]
vals = [r["eta_for_0.1sd"] for r in rows if r["eta_for_0.1sd"] > 0]
eta0 = statistics.median(vals) if vals else 64.0
eta0 = min(max(eta0, 2.0), 512.0)
print(f"{eta0:.3g}")
PYEOF
) || { log "FATAL: eta0 parse failed"; exit 1; }
log "S1 eta0 anchor = $ETA0"

# --------------------------- S2: preseed eval ------------------------------ #
DPCKPT=$(ls -t $S233/coffee/train/DP/DP-base/checkpoints/*.ckpt | head -1)
VIBC=$(ls -t $S233/coffee/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$S233/coffee/rollout/coffee_core.hdf5
if [ ! -f "$PRESEED/failed.json" ]; then
  mkdir -p "$PRESEED/log"
  set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
  log "S2 preseed eval start (gpu1, 100 scenes, unguided)"
  timeout -k 60 10800 env CUDA_VISIBLE_DEVICES=1 SCOUT_RENDER_GPU=1 MUJOCO_GL=egl \
    TMPDIR=/tmp PYTHONUNBUFFERED=1 \
    WANDB_DIR=/root/workspace/baojiachun/wandb_runs \
    WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb \
    $PY -m scout.eval.run_rollout \
    --config configs/eval_coffee_entropy.yaml --task coffee --exp-num 1 \
    --base-dp-ckpt "$DPCKPT" --core-hdf5 "$CORE" \
    --vib-ckpt "$VIBC" \
    --guide atypical --atypical-cap 2.5 --guidance-scale 1.0 \
    --seed 42 --eval-seed 42 \
    --explore-mode rescue --eval-only --save-failed-set "$PRESEED/failed.json" \
    --n-envs 25 \
    --wandb-minimal \
    --output-dir "$PRESEED" \
    --output-success "$PRESEED/success.hdf5" \
    --output-all "$PRESEED/all.hdf5" \
    --wandb-name PRESEED-eval \
    --wandb-project COFFEE-MG-GRID \
    > "$PRESEED/preseed.stdout" 2>&1
  rc=$?
  [ $rc -eq 0 ] && [ -f "$PRESEED/failed.json" ] \
    || { log "FATAL: preseed eval rc=$rc (see $PRESEED/preseed.stdout)"; exit 1; }
fi
N_FAILED=$($PY -c "import json;print(len(json.load(open('$PRESEED/failed.json'))['failed_init_indices']))" 2>/dev/null || echo "?")
log "S2 preseed done: n_failed=$N_FAILED"

# --------------------------- S3: dose probes ------------------------------- #
dose_inject(){  # last mean_inject of a probe stdout
  grep -h "guidance-telemetry" "$1" 2>/dev/null | tail -1 \
    | sed -n 's/.*mean_inject=\([0-9.eE+-]*\).*/\1/p'
}
if ! ls "$CAMP"/DOSEPROBE/ATY_eta*_k2.5/probe.stdout >/dev/null 2>&1; then
  E04=$($PY -c "print(f'{$ETA0/4:.3g}')")
  E4=$($PY -c "print(f'{$ETA0*4:.3g}')")
  log "S3 dose probes: eta {$E04, $ETA0, $E4} (parallel gpu 2/3/4)"
  bash scripts/coffee/cfk_dose_probe.sh 2 ATY "$E04" 2 0:4 2.5 > "$GRIDROOT/dose_${E04}.log" 2>&1 &
  P1=$!
  bash scripts/coffee/cfk_dose_probe.sh 3 ATY "$ETA0" 2 0:4 2.5 > "$GRIDROOT/dose_${ETA0}.log" 2>&1 &
  P2=$!
  bash scripts/coffee/cfk_dose_probe.sh 4 ATY "$E4"  2 0:4 2.5 > "$GRIDROOT/dose_${E4}.log" 2>&1 &
  P3=$!
  wait $P1 $P2 $P3
  log "S3 dose probes done"
fi
INJ_MAX=$(for f in "$CAMP"/DOSEPROBE/ATY_eta*_k2.5/probe.stdout; do dose_inject "$f"; done | sort -g | tail -1)
INJ_MAX=${INJ_MAX:-0}
echo "$INJ_MAX" > "$GRIDROOT/.dose_inject_max"
log "S3 max dose-probe mean_inject = $INJ_MAX"

# --------------------------- S3b: placebo cell ----------------------------- #
if ! awk -F, '$1=="DP_placebo" && $6=="0"' "$GRIDROOT/kgrid_results.csv" 2>/dev/null | grep -q .; then
  log "S3b placebo cell (DP, unguided rescue x5)"
  printf 'ARM=DP ETA=0 KAP=2.5 CELL=DP_placebo\n' > "$GRIDROOT/.spec_placebo"
  run_batch "$GRIDROOT/.spec_placebo" aty
  rm -f "$GRIDROOT/.spec_placebo"
fi
PLACEBO=$($PY - "$GRIDROOT/kgrid_results.csv" <<'PYEOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
vals = [float(r["pass5"]) for r in rows if r["arm"] == "DP" and r["pass5"] and r["rc"] == "0"]
print(f"{max(vals):.4f}" if vals else "0.0")
PYEOF
) || PLACEBO=0.0
log "S3b placebo pass@5 = $PLACEBO"

# --------------------------- S4: coarse ATY batch --------------------------- #
if $PY -c "exit(0 if float('$INJ_MAX') < 0.05 else 1)" 2>/dev/null; then
  ETA0=$($PY -c "print(f'{$ETA0*8:.3g}')")
  log "S4 WARNING: dose probes silent (max mean_inject=$INJ_MAX < 0.05) -> eta0 bumped to $ETA0"
fi
E0=$($PY -c "print(f'{$ETA0/2:.3g}')"); E2=$($PY -c "print(f'{$ETA0*2:.3g}')")
E4=$($PY -c "print(f'{$ETA0*4:.3g}')")
{
  for e in "$E0" "$ETA0" "$E2" "$E4"; do
    for k in 2.5 10; do
      echo "ARM=ATY ETA=$e KAP=$k CELL=ATY_eta${e}_k${k}"
    done
  done
} > "$GRIDROOT/.spec_coarse"
filter_done "$GRIDROOT/.spec_coarse" "$GRIDROOT/kgrid_results.csv" ATY
ATY_ATTEMPTS=$(awk -F, '$2=="ATY"' "$GRIDROOT/kgrid_results.csv" 2>/dev/null | wc -l)
log "S4 coarse batch start (cells=$(wc -l < "$GRIDROOT/.spec_coarse" 2>/dev/null || echo 0), ATY attempts so far=$ATY_ATTEMPTS)"
run_batch "$GRIDROOT/.spec_coarse" aty
rm -f "$GRIDROOT/.spec_coarse"

# --------------------------- S5: adaptive loop ------------------------------ #
ROUND=0
RECOARSE_USED=0
while [ "$ROUND" -lt 6 ]; do
  ROUND=$((ROUND+1))
  SPEC=$GRIDROOT/.spec_adapt$ROUND
  STAGE=$($PY - "$GRIDROOT/kgrid_results.csv" "$PLACEBO" "$GRIDROOT/.dose_inject_max" "$SPEC" <<'PYEOF'
import sys, csv, os
csv_path, placebo_s, inj_path, spec_path = sys.argv[1:5]
placebo = float(placebo_s)
inj_max = 0.0
if os.path.isfile(inj_path):
    try: inj_max = float(open(inj_path).read().strip() or 0)
    except ValueError: inj_max = 0.0
rows = list(csv.DictReader(open(csv_path)))
aty = [r for r in rows if r["arm"] == "ATY" and r["rc"] == "0" and r["pass5"] != ""]
for r in aty:
    r["eta"] = float(r["eta"]); r["kappa"] = float(r["kappa"]); r["p5"] = float(r["pass5"])
    r["jerk"] = float(r["avg_jerk"]) if r["avg_jerk"] else 1e9
attempts = sum(1 for r in rows if r["arm"] == "ATY")
if not aty:
    print("FATAL_NO_ATY_ROWS"); sys.exit()
KAPPA_LADDER = [2.5, 5.0, 10.0, 15.0, 20.0]
def spec(eta, kap):
    e = f"{eta:.3g}"; k = f"{kap:g}"
    return f"ARM=ATY ETA={e} KAP={k} CELL=ATY_eta{e}_k{k}"
def emit(cells):
    open(spec_path, "w").write("\n".join(cells) + "\n")
peak = max(aty, key=lambda r: (r["p5"], -r["jerk"], -r["kappa"], -r["eta"]))
etas = sorted({r["eta"] for r in aty})
top, bot = etas[-1], etas[0]
kpk = peak["kappa"]
strong = [r for r in aty if r["p5"] > placebo + 0.01]
# a) nothing beats placebo
if not strong:
    missing_k = [k for k in KAPPA_LADDER
                 if not any(abs(r["kappa"] - k) < 1e-9 and abs(r["eta"] - peak["eta"]) < 1e-9 for r in aty)]
    if missing_k and attempts < 36:
        emit([spec(peak["eta"], k) for k in missing_k])
        print("KAPPA_FILL"); sys.exit()
    if inj_max < 0.05:
        print("RECOARSE"); sys.exit()
    print("KAPPA_DONE"); sys.exit()
# b) top-edge peak -> extend up (bounded by eta < 512 and attempt cap)
if abs(peak["eta"] - top) < 1e-9 and top < 512 and peak["p5"] > placebo + 0.03 and attempts < 36:
    inner = [r for r in aty if r["eta"] < top and abs(r["kappa"] - kpk) < 1e-9]
    if not inner or peak["p5"] > max(r["p5"] for r in inner):
        emit([spec(top * 2.0, kpk), spec(top * 4.0, kpk)])
        print("EXT_UP"); sys.exit()
# c) bottom-edge peak -> extend down (once, bounded by eta > 0.25)
if abs(peak["eta"] - bot) < 1e-9 and bot > 0.25 and peak["p5"] > placebo + 0.03 and attempts < 36:
    inner = [r for r in aty if r["eta"] > bot and abs(r["kappa"] - kpk) < 1e-9]
    if not inner or peak["p5"] > max(r["p5"] for r in inner):
        emit([spec(bot * 0.25, kpk), spec(bot * 0.5, kpk)])
        print("EXT_DOWN"); sys.exit()
# d) interior peak -> kappa fill at peak eta
missing_k = [k for k in KAPPA_LADDER
             if not any(abs(r["kappa"] - k) < 1e-9 and abs(r["eta"] - peak["eta"]) < 1e-9 for r in aty)]
if missing_k and attempts < 36:
    emit([spec(peak["eta"], k) for k in missing_k])
    print("KAPPA_FILL"); sys.exit()
# e) refine eta around the peak at kappa_peak
ref = [e for e in (peak["eta"] * 0.7, peak["eta"] * 1.4)
       if not any(abs(r["eta"] - e) < 1e-9 and abs(r["kappa"] - kpk) < 1e-9 for r in aty)
       and 0.25 < e < 512]
if ref and attempts < 36:
    emit([spec(e, kpk) for e in ref])
    print("REFINE"); sys.exit()
print("KAPPA_DONE")
PYEOF
) || { log "FATAL: analysis parse failed (round $ROUND)"; break; }

  STAGE=$(echo "$STAGE" | tail -1)
  log "S5 analysis round $ROUND -> $STAGE"
  if [ "$STAGE" = "CAP_END" ] || [ "$STAGE" = "KAPPA_DONE" ] || [ "$STAGE" = "FATAL_NO_ATY_ROWS" ]; then
    break
  fi
  if [ "$STAGE" = "RECOARSE" ]; then
    if [ "$RECOARSE_USED" -ge 1 ]; then log "RECOARSE already used -> loop end"; break; fi
    RECOARSE_USED=1
    ETA0=$($PY -c "print(f'{$ETA0*8:.3g}')")
    {
      for e in $($PY -c "print(f'{$ETA0/2:.3g}'); print(f'{$ETA0:.3g}'); print(f'{$ETA0*2:.3g}'); print(f'{$ETA0*4:.3g}')"); do
        for k in 2.5 10; do echo "ARM=ATY ETA=$e KAP=$k CELL=ATY_eta${e}_k${k}"; done
      done
    } > "$SPEC"
    filter_done "$SPEC" "$GRIDROOT/kgrid_results.csv" ATY
    run_batch "$SPEC" aty
    continue
  fi
  if [ -s "$SPEC" ]; then
    filter_done "$SPEC" "$GRIDROOT/kgrid_results.csv" ATY
    run_batch "$SPEC" aty
  else
    log "no cells proposed -> loop end"
    break
  fi
done

# ------------- S6: eta candidates (TOP-2 distinct ATY etas) ----------------- #
ETA_CANDS=$($PY - "$GRIDROOT/kgrid_results.csv" <<'PYEOF'
import sys, csv
rows = list(csv.DictReader(open(sys.argv[1])))
aty = [r for r in rows if r["arm"] == "ATY" and r["rc"] == "0" and r["pass5"] != ""]
for r in aty:
    r["eta"] = float(r["eta"]); r["kappa"] = float(r["kappa"]); r["p5"] = float(r["pass5"])
    r["jerk"] = float(r["avg_jerk"]) if r["avg_jerk"] else 1e9
best = {}
for r in aty:                      # per-eta best row: p5 -> -jerk -> -kappa
    e = r["eta"]
    if e not in best or (r["p5"], -r["jerk"], -r["kappa"]) > \
       (best[e]["p5"], -best[e]["jerk"], -best[e]["kappa"]):
        best[e] = r
top = sorted(best.values(),
             key=lambda r: (r["p5"], -r["jerk"], -r["kappa"], -r["eta"]))[::-1]
cands = top[:2]
print("\n".join(f"{r['eta']:.6g} {r['kappa']:g}" for r in cands))
PYEOF
) || { log "FATAL: eta candidates pick failed"; exit 1; }
[ -n "$ETA_CANDS" ] || { log "FATAL: no ATY eta candidates"; exit 1; }
log "S6 eta candidates (top-2 ATY): $(echo "$ETA_CANDS" | tr '\n' ';')"

# --------------------------- S7: ORBIT batch at BOTH candidates -------------- #
: > "$GRIDROOT/.spec_orbit"
while read -r e k; do
  [ -n "$e" ] || continue
  for s in 0.025 0.05 0.1 0.2; do
    for l in 0.25 0.5 1.0; do
      echo "ETA=$e KAP=$k SIG=$s LAM=$l CELL=ORBIT_e${e}_s${s}_l${l}" >> "$GRIDROOT/.spec_orbit"
    done
  done
done <<< "$ETA_CANDS"
filter_done "$GRIDROOT/.spec_orbit" "$GRIDROOT/ogrid_results.csv" ORBIT
run_batch "$GRIDROOT/.spec_orbit" orbit
rm -f "$GRIDROOT/.spec_orbit"

# ------------- S7.5: eta* by the SUM criterion (user 2026-09-20) ------------- #
PICK=$($PY - "$GRIDROOT/kgrid_results.csv" "$GRIDROOT/ogrid_results.csv" <<'PYEOF'
import sys, csv
aty = [r for r in csv.DictReader(open(sys.argv[1]))
       if r["arm"] == "ATY" and r["rc"] == "0" and r["pass5"] != ""]
orb = [r for r in csv.DictReader(open(sys.argv[2])) if r["rc"] == "0" and r["pass5"] != ""]
for rs in (aty, orb):
    for r in rs:
        r["eta"] = float(r["eta"]); r["kappa"] = float(r["kappa"]); r["p5"] = float(r["pass5"])
        r["jerk"] = float(r["avg_jerk"]) if r.get("avg_jerk") else 1e9
def best(rows):
    return max(rows, key=lambda r: (r["p5"], -r["jerk"], -r["kappa"])) if rows else None
per_eta = []
for e in sorted({r["eta"] for r in aty}):
    a_row = best([r for r in aty if abs(r["eta"] - e) < 1e-9])
    o_row = best([r for r in orb if abs(r["eta"] - e) < 1e-9])
    o_p5 = o_row["p5"] if o_row else 0.0
    per_eta.append({"eta": e, "kappa": a_row["kappa"], "aty": a_row["p5"],
                    "orb": o_p5, "sum": a_row["p5"] + o_p5,
                    "min": min(a_row["p5"], o_p5)})
per_eta.sort(key=lambda c: (-c["sum"], -c["min"], c["eta"]))
c = per_eta[0]
print(f"{c['eta']:.6g} {c['kappa']:g}")
PYEOF
) || { log "FATAL: sum-criterion pick failed"; exit 1; }
ETA_STAR=$(echo "$PICK" | cut -d' ' -f1)
KAPPA_STAR=$(echo "$PICK" | cut -d' ' -f2)
log "S7.5 eta*=$ETA_STAR kappa*=$KAPPA_STAR (max aty+orb pass@5; ties -> higher min -> lower eta)"

# ------------- S7b: ORBIT extension if below ATY at eta* --------------------- #
ORBEXT=$($PY - "$GRIDROOT/kgrid_results.csv" "$GRIDROOT/ogrid_results.csv" "$ETA_STAR" <<'PYEOF'
import sys, csv
aty = [r for r in csv.DictReader(open(sys.argv[1])) if r["arm"] == "ATY" and r["rc"] == "0" and r["pass5"]]
orb = [r for r in csv.DictReader(open(sys.argv[2])) if r["rc"] == "0" and r["pass5"]]
eta_star = float(sys.argv[3])
a_max = max((float(r["pass5"]) for r in aty if abs(float(r["eta"]) - eta_star) < 1e-9), default=0.0)
o_max = max((float(r["pass5"]) for r in orb if abs(float(r["eta"]) - eta_star) < 1e-9), default=0.0)
print("EXTEND" if o_max < a_max else "OK")
PYEOF
) || ORBEXT=OK
if [ "$ORBEXT" = "EXTEND" ]; then
  LAM_BEST=$($PY - "$GRIDROOT/ogrid_results.csv" "$ETA_STAR" <<'PYEOF'
import sys, csv
eta_star = float(sys.argv[2])
rows = [r for r in csv.DictReader(open(sys.argv[1]))
        if r["rc"] == "0" and r["pass5"] and abs(float(r["eta"]) - eta_star) < 1e-9]
print(f"{max(rows, key=lambda r: float(r['pass5']))['lam']:g}" if rows else "0.5")
PYEOF
)
  log "S7b orbit below ATY at eta* -> extension sigma {0.0125, 0.3} x lam $LAM_BEST"
  {
    echo "ETA=$ETA_STAR KAP=$KAPPA_STAR SIG=0.0125 LAM=$LAM_BEST CELL=ORBIT_e${ETA_STAR}_s0.0125_l${LAM_BEST}"
    echo "ETA=$ETA_STAR KAP=$KAPPA_STAR SIG=0.3 LAM=$LAM_BEST CELL=ORBIT_e${ETA_STAR}_s0.3_l${LAM_BEST}"
  } > "$GRIDROOT/.spec_orbit_ext"
  filter_done "$GRIDROOT/.spec_orbit_ext" "$GRIDROOT/ogrid_results.csv" ORBIT
  run_batch "$GRIDROOT/.spec_orbit_ext" orbit
  rm -f "$GRIDROOT/.spec_orbit_ext"
fi

# --------------------------- S8: verdict ------------------------------------ #
# eta*/kappa* come from the S7.5 SUM pick (recomputed here from the csvs as
# the single source of truth, now including any S7b extension cells); S8
# derives sigma*/lam* at eta* and assembles the report.
$PY - "$GRIDROOT/kgrid_results.csv" "$GRIDROOT/ogrid_results.csv" \
     "$GRIDROOT/verdict.json" "$N_FAILED" "$PLACEBO" <<'PYEOF'
import sys, csv, json
kcsv, ocsv, vjson, n_failed_s, placebo_s = sys.argv[1:6]
rows = list(csv.DictReader(open(kcsv)))
aty = [r for r in rows if r["arm"] == "ATY" and r["rc"] == "0" and r["pass5"] != ""]
orb = [r for r in csv.DictReader(open(ocsv)) if r["rc"] == "0" and r["pass5"] != ""]
for rs in (aty, orb):
    for r in rs:
        r["eta"] = float(r["eta"]); r["kappa"] = float(r["kappa"])
        r["p5"] = float(r["pass5"])
        r["jerk"] = float(r["avg_jerk"]) if r.get("avg_jerk") else 1e9
for r in orb:
    r["sigma"] = float(r["sigma"]); r["lam"] = float(r["lam"])
placebo = float(placebo_s)
verdict = {}
if not aty:
    verdict["error"] = "no valid ATY rows"
else:
    def abest(e):
        rs = [r for r in aty if abs(r["eta"] - e) < 1e-9]
        return max(rs, key=lambda r: (r["p5"], -r["jerk"], -r["kappa"]))
    def omax(e):
        rs = [r for r in orb if abs(r["eta"] - e) < 1e-9]
        return max((r["p5"] for r in rs), default=0.0)
    cands = []
    for e in sorted({r["eta"] for r in aty}):
        a = abest(e)
        o = omax(e)
        cands.append({"eta": e, "kappa": a["kappa"], "aty_p5": a["p5"],
                      "orb_p5": o, "sum": a["p5"] + o, "min": min(a["p5"], o)})
    cands.sort(key=lambda c: (-c["sum"], -c["min"], c["eta"]))
    star = cands[0]
    verdict["eta_star"] = star["eta"]
    verdict["kappa_star"] = star["kappa"]
    verdict["aty_max_pass5"] = star["aty_p5"]
    verdict["sum_criterion"] = "eta* = argmax(aty_p5 + orb_p5); ties -> higher min(aty,orb) -> lower eta"
    verdict["candidates"] = cands
    rows_es = [r for r in aty if abs(r["eta"] - star["eta"]) < 1e-9]
    verdict["aty_max_rescued"] = max((int(r["rescued"]) for r in rows_es if r.get("rescued") != ""), default="")
    orb_s = [r for r in orb if abs(r["eta"] - star["eta"]) < 1e-9]
    if orb_s:
        sigmas = sorted({r["sigma"] for r in orb_s})
        def sig_max_p5(s):
            return max(r["p5"] for r in orb_s if abs(r["sigma"] - s) < 1e-9)
        best_sp = max(sig_max_p5(s) for s in sigmas)
        sigma_star = min(s for s in sigmas if sig_max_p5(s) >= best_sp - 0.005)  # tie -> LOWER sigma
        orb_ss = [r for r in orb_s if abs(r["sigma"] - sigma_star) < 1e-9]
        lams = sorted({r["lam"] for r in orb_ss})
        best_lp = max(max(r["p5"] for r in orb_ss if abs(r["lam"] - l) < 1e-9) for l in lams)
        lam_star = min(l for l in lams
                       if max(r["p5"] for r in orb_ss if abs(r["lam"] - l) < 1e-9) >= best_lp - 0.005)
        verdict["sigma_star"] = sigma_star
        verdict["lam_star"] = lam_star
        verdict["orbit_max_pass5"] = max(r["p5"] for r in orb_s)
        verdict["orbit_below_aty"] = verdict["orbit_max_pass5"] < verdict["aty_max_pass5"]
verdict["placebo_pass5"] = placebo
verdict["n_failed"] = n_failed_s
verdict["cells_aty_valid"] = len(aty)
verdict["cells_orbit_valid"] = len(orb)
verdict["low_signal"] = n_failed_s.isdigit() and int(n_failed_s) < 15
verdict["upstream"] = {"kgrid_csv": kcsv, "ogrid_csv": ocsv,
                       "telemetry": "TELEMETRY/grad_probe_COFFEE-MG-p1-s233.json"}
json.dump(verdict, open(vjson, "w"), indent=1)
print(json.dumps(verdict, indent=1))
PYEOF
rc=$?
[ $rc -eq 0 ] && [ -f "$GRIDROOT/verdict.json" ] || {
  log "FATAL: verdict generation failed -- writing error verdict"
  echo '{"error": "verdict generation failed"}' > "$GRIDROOT/verdict.json"
  exit 1
}
{
  echo "=== COFFEE-MG-p1 s233 grid verdict $(date '+%F %T') ==="
  echo "--- kgrid_results.csv ---"; cat "$GRIDROOT/kgrid_results.csv"
  echo "--- ogrid_results.csv ---"; cat "$GRIDROOT/ogrid_results.csv"
  echo "--- verdict.json ---"; cat "$GRIDROOT/verdict.json"
} > "$GRIDROOT/verdict.txt"
log "GRID DONE -- verdict at $GRIDROOT/verdict.json"
