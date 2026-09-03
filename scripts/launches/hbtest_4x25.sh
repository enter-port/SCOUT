#!/bin/bash
# hbtest_4x25.sh -- one-off heartbeat/OOM-visibility test (user 2026-09-02):
# 4 shard workers x 25 envs on the s233 SCOUT round1 trio (DP-base 599 +
# dyn-base + frozen failed set), 20 failed inits x 10 tries = exactly 200
# orbit rescue rollouts. Verifies, WITHOUT touching any chain:
#   (a) [explore-hb] lines stream into each shard stdout;
#   (b) shard_heartbeat.py relays explore_hb/* keys into wandb (dedicated
#       run hbtest-4x25 in project TOOLHANG-9-1-orbit-s233, tagged hbtest);
#   (c) per-worker RSS + system memory over time (heartbeat.log);
#   (d) wall-clock per 200 rollouts -> extrapolate a real 890-traj round.
# usage: GPU=1 bash soe_scripts/hbtest_4x25.sh
set -uo pipefail
GPU=${GPU:?set GPU=<cuda id>}
cd /root/workspace/baojiachun/scout-orbit || exit 1
PY=/root/workspace/baojiachun/.venv/bin/python
TH=data/2026_9_1_toolhang/TOOLHANG-s233/tool_hang
DP=$(ls -t $TH/train/DP/DP-base/checkpoints/*.ckpt | head -1)
VIB=$(ls -t $TH/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$TH/rollout/tool_hang_core.hdf5
FAILED=$TH/rollout/SCOUT-exp1/failed.json
T=data/2026_9_1_toolhang/hbtest_4x25
PROJ=TOOLHANG-9-1-orbit-s233
mkdir -p $T/log
[ -f "$FAILED" ] || { echo "FATAL: no failed.json at $FAILED"; exit 1; }
[ -f "$CORE" ] && [ -n "$DP" ] && [ -n "$VIB" ] || { echo "FATAL: trio missing (dp=$DP vib=$VIB)"; exit 1; }

export MUJOCO_GL=egl SCOUT_RENDER_GPU=$GPU TMPDIR=/tmp PYTHONUNBUFFERED=1
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

# 20 failed inits x 10 tries = 200 rollouts, SAME frozen scene set
$PY - "$FAILED" "$T/failed.json" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
idx = spec["failed_init_indices"][:20]
spec["failed_init_indices"] = idx
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print(f"[hbtest] failed set cut to {len(idx)} inits x10 tries = "
      f"{len(idx) * 10} rollouts (of {sys.argv[1]})")
PYEOF

# dedicated wandb run for the heartbeat relay
RID=$($PY -c "
import wandb
r = wandb.init(project='$PROJ', name='hbtest-4x25', tags=['hbtest', 'non-campaign'])
print(r.id)
r.finish()" 2>/dev/null | tail -1)
if [ ${#RID} -lt 6 ]; then
  echo "[hbtest] WARN: wandb run creation failed (RID='$RID') -- test continues, heartbeat log-only"
  RID=""
fi
echo "[hbtest] dp=$DP"
echo "[hbtest] vib=$VIB"
echo "[hbtest] wandb run id=$RID (project=$PROJ name=hbtest-4x25)"

HB=""
if [ -n "$RID" ]; then
  nohup $PY soe_scripts/shard_heartbeat.py \
    --project "$PROJ" --run-id "$RID" \
    --shard-glob "$T/log/shard*.stdout" --match "$T" \
    --stop-file "$T/all.hdf5" --log-file "$T/heartbeat.log" \
    --interval 30 --max-min 180 > $T/heartbeat.stdout 2>&1 &
  HB=$!
  echo "[hbtest] heartbeat pid=$HB (interval 30s)"
fi

T0=$(date +%s)
PYTHON=$PY CLEANUP_SHARDS=0 bash soe_scripts/shard_rollout.sh 4 \
  "$T/log/tool_hang_SCOUT_explore_hbtest.json" \
  "$T/success.hdf5" "$T/all.hdf5" "$CORE" -- \
  --config configs/eval_tool_hang_entropy.yaml --task tool_hang --exp-num 1 \
  --base-dp-ckpt "$DP" --core-hdf5 "$CORE" --vib-ckpt "$VIB" \
  --guide orbit --atypical-cap 2.5 --orbit-lam 0.5 --orbit-delta 0.25 --orbit-sigma 0.25 \
  --explore-mode rescue --explore-try-times 10 --failed-set-json "$T/failed.json" \
  --n-envs 25 --seed 42 --eval-seed 42 --no-wandb \
  --output-dir "$T" --output-success "$T/success.hdf5" --output-all "$T/all.hdf5"
RC=$?
T1=$(date +%s)
[ -n "$HB" ] && { kill $HB 2>/dev/null; wait $HB 2>/dev/null; }
echo "[hbtest] shard driver rc=$RC wall=$(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s (200 rollouts, 4x25)"
echo "[hbtest] heartbeat.log (RSS curve, per 30s):"
tail -8 $T/heartbeat.log 2>/dev/null
echo "[hbtest] heartbeat.log max RSS per worker:"
awk 'match($0, /rss\[([^\]]*)\]/, a) {print a[1]}' $T/heartbeat.log 2>/dev/null | tr ',' '\n' | sort | tail -4
echo "[hbtest] min system avail:"
grep -o "sys=[0-9.]*/" $T/heartbeat.log 2>/dev/null | sort -t= -k2 -n | head -2
echo "[hbtest] per-shard hb lines (last 2 each):"
for f in $T/log/shard*.stdout; do echo "-- $(basename $f)"; grep -a "explore-hb" "$f" | tail -2; done
echo "[hbtest] merged json summary:"
$PY -c "
import json, glob
p = sorted(glob.glob('$T/log/tool_hang_SCOUT_explore_hbtest.json'))
d = json.load(open(p[0])) if p else {}
print({k: d.get(k) for k in ['pass_at_5', 'exploration_rescued', 'n_failed']})" 2>/dev/null
echo "[hbtest] DONE"
