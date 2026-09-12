#!/bin/bash
# One-shot summary of the exploit gate scan (run any time after the chain
# finishes; safe: read-only). Prints the comparison table + gate telemetry.
set -euo pipefail
ROOT=/root/workspace/baojiachun/scout-exploit/data/exploit_sq233_r3/scan
PY=/root/workspace/baojiachun/.venv/bin/python

echo "=== baselines (first-50 scenes, seed 42..91) ==="
echo "  pure DP (r4 json)                 33/50"
echo "  exploit ungated  gst100           25/50   jerk 0.336"
echo
echo "=== scan groups ==="
for d in "$ROOT"/gst50 "$ROOT"/thr087 "$ROOT"/thr154 "$ROOT"/gst50_thr154; do
  tag=$(basename "$d")
  J="$d/log/square_SCOUT_rollout_exp1.json"
  if [ -f "$J" ]; then
    "$PY" -c "
import json
j = json.load(open('$J'))
print(f\"  {('$tag'):16s} solved {j.get('explore_solved')}/50   jerk {round(j.get('avg_jerk') or 0, 4)}\")"
  else
    echo "  $tag   (json not yet written)"
  fi
done
echo
echo "=== gate telemetry (open-rate per arm, from the chain log) ==="
grep -a 'exploit-gate' /tmp/exploit_scan.log | tail -12 || true
echo
echo "=== chain tail ==="
grep -a 'SCAN\|ALL DONE\|Traceback' /tmp/exploit_scan.log | tail -8 || true
