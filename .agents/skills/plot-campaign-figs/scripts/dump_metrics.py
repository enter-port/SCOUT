#!/usr/bin/env python3
"""Dump scalar metrics from a SCOUT campaign's rollout jsons as one JSON list.

Run from the LOCAL repo root (Git Bash), piping this script to the server venv
python over the ssh exec channel (sftp is disabled; this avoids quoting hell):

  ssh -o BatchMode=yes -o ConnectTimeout=20 -p 1022 root@106.14.2.243 \
    "/root/workspace/baojiachun/.venv/bin/python - <DATA_ROOT>" \
    < .agents/skills/plot-campaign-figs/scripts/dump_metrics.py \
    > <figs_dir>/raw_metrics.txt

argv[1] = campaign data root on the server, e.g.
          /root/workspace/baojiachun/scout/data/2026_9_13_transport

Scalar keys are kept verbatim; list values are reduced to "<key>#len" counts.
Glob is **/rollout/*-exp*/log/*.json under the root (probe/smoke dirs that do
not match this layout are naturally excluded). Disambiguate eval vs explore
jsons LOCALLY by keys (eval: success_rate + failed set; explore: pass_at_5),
never by filename alone.
"""
import glob
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "."
pat = os.path.join(root, "**", "rollout", "*-exp*", "log", "*.json")
rows = []
for f in sorted(glob.glob(pat, recursive=True)):
    row = {"file": os.path.relpath(f, root)}
    try:
        with open(f) as fh:
            d = json.load(fh)
    except Exception as e:  # unreadable json: report, keep going
        row["error"] = str(e)
        rows.append(row)
        continue
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (int, float, str, bool)) or v is None:
                row[k] = v
            elif isinstance(v, list):
                row[k + "#len"] = len(v)
    rows.append(row)
print(json.dumps(rows, ensure_ascii=False, indent=1))
