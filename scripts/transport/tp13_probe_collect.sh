#!/bin/bash
# tp13_probe_collect.sh -- summarize the TRANSPORT-9-13 dose probe (jsons
# authoritative; run any time, partial-safe). Probe arms run run_rollout in
# SINGLE-process rescue mode (not shard_rollout), so their jsons are
# log/{task}_{SCOUT|DP}_rollout_exp0.json -- glob *rollout*.json (review P1
# fix 2026-09-13). Keys per rollout_pipeline/run_rollout: success_rate,
# pass_at_5 (= pass@try_times), exploration_rescued, avg_jerk,
# n_init_states, failed_init_indices, wandb_run_id.
set -u
PDIR=/root/workspace/baojiachun/scout/data/2026_9_13_transport/probe_s233
PY=/root/workspace/baojiachun/.venv/bin/python
cd /root/workspace/baojiachun/scout || exit 1
$PY - "$PDIR" <<'PYEOF'
import sys, json, glob, os
pdir = sys.argv[1]

def latest_json(pat):
    ps = sorted(glob.glob(pat), key=os.path.getmtime)
    return json.load(open(ps[-1])) if ps else None

base = latest_json(os.path.join(pdir, "base", "log", "*rollout*.json"))
if base:
    nfail = len(base.get("failed_init_indices") or [])
    print(f"[base] eval SR={base.get('success_rate')} n={base.get('n_init_states')} "
          f"failed={nfail}  (probe scenes = first 15 of failed)")
else:
    print("[base] no eval json yet")

print(f"{'arm':6} {'rescued':>7} {'pass@5':>7} {'n':>4} {'jerk':>7}  run_id")
for tag in ["s0", "a025", "a05", "a10", "orb"]:
    d = latest_json(os.path.join(pdir, tag, "log", "*rollout*.json"))
    if not d:
        print(f"{tag:6} {'-':>7} {'-':>7} {'-':>4} {'-':>7}  (pending)")
        continue
    rescued = d.get("exploration_rescued")
    p5 = d.get("pass_at_5")
    n = len(d.get("failed_init_indices") or [])
    jerk = d.get("avg_jerk")
    rid = d.get("wandb_run_id", "-")
    jerk_s = f"{jerk:.3f}" if isinstance(jerk, (int, float)) else "-"
    print(f"{tag:6} {str(rescued):>7} {str(p5):>7} {str(n):>4} {jerk_s:>7}  {rid}")
print("\n(dose telemetry mean_inject / orbit-telemetry: wandb project TRANSPORT-9-13-probe)")
PYEOF
