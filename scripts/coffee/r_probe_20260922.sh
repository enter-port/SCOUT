#!/bin/bash
# r_probe_20260922.sh -- cross-task R-invariant probe (user idea, 2026-09-22).
#   R = mean_abs_grad_base * eta0 / mean_abs_action_core
# For each (task, seed): run cfk_param_calib.py n3 with round-vib = base-vib
# (self probe -> phi=1, we only read magabs_base / far_base), on the SAME
# frozen-batch convention as the p3 campaign (B=128, seed0, perm seed7).
# Then a collector computes mean|abs_actions| over each core hdf5 and the
# final R table. READ-ONLY probe: writes only under data/R_probe_20260922/.
#
# usage: bash r_probe_20260922.sh <gpu_id>     # one worker lane
#        bash r_probe_20260922.sh collect      # assemble R table (no GPU)
set -u
ROOT=/root/workspace/baojiachun/scout
cd "$ROOT" || exit 1
OUT=data/R_probe_20260922
mkdir -p "$OUT"

# job table: task|seed|venv|vibcfg|evalcfg|taskdir|root|core|dp|vib|eta0
JOBS=$OUT/jobs.txt
if [ ! -s "$JOBS" ]; then
D=data
newest_vib(){ ls -t "$1"/dyn-base/*/scout_vib.ckpt 2>/dev/null | head -1; }
{
for s in 233 2333 23333; do
  # can: s233 = entropy-era base (core content == orbchain seeded split);
  #      s2333/s23333 = orbchain fresh round0 base
  if [ $s = 233 ]; then CR=$D/2026_8_21_entropy/CAN-entropy-s233/can; CO=$D/2026_9_2_orbchain/ORBIT-s233/can/rollout/can_core.hdf5; else CR=$D/2026_9_2_orbchain/ORBIT-s$s/can; CO=$CR/rollout/can_core.hdf5; fi
  echo "can|$s|.venv|vib_can_exp1.yaml|eval_can_entropy.yaml|$CR|$CO|$CR/train/DP/DP-base/checkpoints/599.ckpt|$(newest_vib $CR/train/dyn)|3.0"
  SR=$D/2026_8_26_entropy/SQUARE-entropy-s$s/square
  echo "square|$s|.venv|vib_square_exp1.yaml|eval_square_entropy.yaml|$SR|$D/2026_8_26_entropy/core_rebuild/square_core_s$s.hdf5|$SR/train/DP/DP-base/checkpoints/599.ckpt|$(newest_vib $SR/train/dyn)|3.0"
  TR=$D/2026_9_5_toolhang/TOOLHANG-s$s/tool_hang
  echo "tool_hang|$s|.venv|vib_tool_hang_exp1.yaml|eval_tool_hang_entropy.yaml|$TR|$TR/rollout/tool_hang_core.hdf5|$TR/train/DP/DP-base/checkpoints/599.ckpt|$(newest_vib $TR/train/dyn)|0.5"
  PR=$D/2026_9_15_transport_p1/TRANSPORT-s$s/transport
  echo "transport|$s|.venv|vib_transport_exp1.yaml|eval_transport_entropy.yaml|$PR|$PR/rollout/transport_core.hdf5|$PR/train/DP/DP-base/checkpoints/599.ckpt|$(newest_vib $PR/train/dyn)|0.5"
  HR=$D/2026_9_16_threading_p1/THREADING-MG-p1-s$s/threading
  echo "threading|$s|.venv|vib_threading_exp1.yaml|eval_threading_entropy.yaml|$HR|$HR/rollout/threading_core.hdf5|$HR/train/DP/DP-base/checkpoints/599.ckpt|$(newest_vib $HR/train/dyn)|2.0"
  FR=$D/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s$s/coffee
  echo "coffee|$s|.venv_mg|vib_coffee_exp1.yaml|eval_coffee_entropy.yaml|$FR|$FR/rollout/coffee_core.hdf5|$FR/train/DP/DP-base/checkpoints/599.ckpt|$(newest_vib $FR/train/dyn)|5.6"
done
} > "$JOBS"
wc -l "$JOBS"
fi

MODE=${1:-}
if [ "$MODE" = collect ]; then
  PYV=/root/workspace/baojiachun/.venv/bin/python
  exec $PYV - "$OUT" <<'EOF'
import json, os, sys, h5py, numpy as np
out = sys.argv[1]
rows = []
for line in open(os.path.join(out, "jobs.txt")):
    line = line.strip()
    if not line: continue
    task, seed, venv, vibcfg, evalcfg, cr, core, dp, vib, eta0 = line.split("|")
    tag = f"{task}_s{seed}"
    pj = os.path.join(out, f"{tag}.json")
    if not os.path.exists(pj):
        rows.append((task, seed, float(eta0), None, None, None)); continue
    d = json.load(open(pj))
    # mean |abs_actions| over the whole core hdf5 (raw action units)
    with h5py.File(core, "r") as f:
        vals = [f[f"data/{k}/abs_actions"][:] for k in f["data"].keys()]
    aa = np.concatenate(vals, 0)
    mabs = float(np.abs(aa).mean())
    mag = d["magabs_base"]
    rows.append((task, seed, float(eta0), mag, mabs,
                 mag * float(eta0) / mabs))
rows.sort(key=lambda r: (r[0], r[1]))
print(f"{'task':<11}{'seed':<7}{'eta0':<7}{'magabs_base':<13}{'meanabs_a_core':<16}{'R':<10}")
for t, s, e, mag, mabs, R in rows:
    if mag is None:
        print(f"{t:<11}{s:<7}{e:<7}{'MISSING':<13}")
    else:
        print(f"{t:<11}{s:<7}{e:<7}{mag:<13.5g}{mabs:<16.5g}{R:<10.5g}")
json.dump([{"task": t, "seed": s, "eta0": e, "magabs_base": m,
            "meanabs_action_core": a, "R": (None if m is None else m*e/a)}
           for t, s, e, m, a, R in rows],
          open(os.path.join(out, "R_table.json"), "w"), indent=1)
print("wrote", os.path.join(out, "R_table.json"))
EOF
fi

GPU=$MODE
PY=/root/workspace/baojiachun/.venv/bin/python
LOCK=$OUT/queue.lock
DONE=$OUT/done.txt
touch "$DONE"
while :; do
  JOB=$(flock "$LOCK" bash -c "grep -vxF -f '$DONE' '$JOBS' | head -1")
  [ -z "$JOB" ] && break
  flock "$LOCK" echo "$JOB" >> "$DONE"
  IFS='|' read -r task seed venv vibcfg evalcfg cr core dp vib eta0 <<< "$JOB"
  tag=${task}_s${seed}
  PYB=/root/workspace/baojiachun/$venv/bin/python
  echo "[$(date '+%T')] GPU$GPU START $tag"
  env CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=$GPU \
    "$PYB" scripts/coffee/cfk_param_calib.py n3 \
      --seed-data-root "$cr/.." --base-vib "$vib" --round-vib "$vib" \
      --dp-ckpt "$dp" --vib-config "configs/$vibcfg" --eval-config "configs/$evalcfg" \
      --task "$task" --core-hdf5 "$core" --batch-size 128 \
      --eta0 "$eta0" --kappa0 2.5 --out "$OUT/$tag.json" \
      > "$OUT/$tag.stdout" 2>&1
  echo "[$(date '+%T')] GPU$GPU rc=$? $tag"
done
echo "[$(date '+%T')] GPU$GPU lane DONE"
