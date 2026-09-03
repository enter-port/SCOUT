#!/bin/bash
# aty_probe_10env.sh -- atypical (phase-1-only) rescue probe on tool_hang (user 2026-09-03).
# Context: orbit params rescue nothing on tool_hang (r1 phase B: DP arm collecting
# successes, orbit arm 0 collected). Test whether the phase-1 climb alone (guide=atypical,
# cost=-min(KL,kappa), kappa=2.5, guidance_scale=12.0 from the orbit-calibrated config)
# rescues anything. 10 failed inits (first 10 of the r1 frozen failed set) x 10 tries
# = 100 rollouts, n_envs=10. Does NOT touch the running campaign: separate output
# dir, read-only use of the r1 trio (DP-base/dyn-base/core) + failed.json.
# usage: GPU=2 bash soe_scripts/aty_probe_10env.sh
set -uo pipefail
GPU=${GPU:-2}
cd /root/workspace/baojiachun/scout-orbit || exit 1
PY=/root/workspace/baojiachun/.venv/bin/python
TH=data/2026_9_1_toolhang/TOOLHANG-s233/tool_hang
DP=$(ls -t $TH/train/DP/DP-base/checkpoints/*.ckpt | head -1)
VIB=$(ls -t $TH/train/dyn/dyn-base/*/scout_vib.ckpt | head -1)
CORE=$TH/rollout/tool_hang_core.hdf5
FAILED=$TH/rollout/SCOUT-exp1/failed.json
T=data/2026_9_3_atyprobe/probe1_s12
mkdir -p $T/log
[ -f "$FAILED" ] || { echo "[aty-probe] FATAL: no failed.json at $FAILED"; exit 1; }
[ -f "$CORE" ] && [ -n "$DP" ] && [ -n "$VIB" ] || { echo "[aty-probe] FATAL: trio missing (dp=$DP vib=$VIB)"; exit 1; }

export MUJOCO_GL=egl SCOUT_RENDER_GPU=$GPU TMPDIR=/tmp PYTHONUNBUFFERED=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
set -a; . /root/workspace/baojiachun/.secrets/wandb.env; set +a
export WANDB_DIR=/root/workspace/baojiachun/wandb_runs
export WANDB_CACHE_DIR=/root/workspace/baojiachun/.cache/wandb

$PY - "$FAILED" "$T/failed10.json" <<'PYEOF'
import json, sys
spec = json.load(open(sys.argv[1]))
idx = spec["failed_init_indices"][:10]
spec["failed_init_indices"] = idx
json.dump(spec, open(sys.argv[2], "w"), indent=1)
print(f"[aty-probe] failed set cut to {len(idx)} inits x10 tries = {len(idx)*10} rollouts")
print(f"[aty-probe] inits = {idx}")
PYEOF

echo "[aty-probe] dp=$DP"
echo "[aty-probe] vib=$VIB"
echo "[aty-probe] gpu=$GPU guide=atypical kappa=2.5 scale=12.0(config, unpatched)"
T0=$(date +%s)
env CUDA_VISIBLE_DEVICES=$GPU $PY -m scout.eval.run_rollout \
  --config configs/eval_tool_hang_entropy.yaml --task tool_hang --exp-num 1 \
  --base-dp-ckpt "$DP" --vib-ckpt "$VIB" --core-hdf5 "$CORE" \
  --guide atypical --atypical-cap 2.5 \
  --explore-mode rescue --explore-try-times 10 --failed-set-json "$T/failed10.json" \
  --n-envs 10 --seed 42 --eval-seed 42 \
  --wandb-name aty-probe-s12-10env \
  --output-dir "$T" --output-success "$T/success.hdf5" --output-all "$T/all.hdf5" \
  --output-json "$T/log/aty_probe_explore.json" \
  > $T/rollout.stdout 2>&1
RC=$?
T1=$(date +%s)
echo "[aty-probe] rc=$RC wall=$(( (T1-T0)/60 ))m$(( (T1-T0)%60 ))s"
echo "[aty-probe] summary (rescued-of-10 is THE signal):"
$PY - <<PYEOF
import json
d = json.load(open("$T/log/aty_probe_explore.json"))
print({k: d.get(k) for k in ["exploration_rescued", "n_failed", "pass_at_5", "explore_solved"]})
PYEOF
echo "[aty-probe] success.hdf5 demo count:"
$PY -c "
import h5py
try:
    with h5py.File('$T/success.hdf5','r') as f:
        print(len(list(f['data'].keys())) if 'data' in f else 'no data group')
except Exception as e:
    print('ERR', e)"
echo "[aty-probe] guidance telemetry (last 3):"
grep -a "guidance-telemetry" $T/rollout.stdout | tail -3
echo "[aty-probe] hb (last 3):"
grep -a "explore-hb" $T/rollout.stdout | tail -3
echo "[aty-probe] DONE $(date '+%F %T')"
