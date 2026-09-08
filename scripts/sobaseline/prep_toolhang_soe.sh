#!/bin/bash
# prep_toolhang_soe.sh -- one-time tool_hang SOE dataset build (2026-09-07).
# Mirrors prep_square_soe.sh with these deltas:
#   * --n 40: SCOUT's toolhang core is FORTY demos (split_core.py
#     default_rng(seed).choice(200, 40, replace=False) sorted -- the exact
#     same rng call make_core_soe uses, so the selection is identical).
#   * --scout-core points at the TOOLHANG-9-5-orbit-s233 campaign core for
#     the count/length/total cross-check (selection identity proof).
#   * The SOE core is then REBUILT from SCOUT's own abs_actions values via
#     make_core_soe_from_scout.py (rotvec->6d math), making the 40-demo
#     training data VALUE-IDENTICAL to TOOLHANG-9-5-orbit-s233. Reason: the
#     two converter lineages diverge on each demo's FIRST action (controller
#     goal-sync after reset_to; pos<=~5cm at step 0 only, mid-demo identical,
#     same pattern present in the historical can datasets). The
#     replay-converted 200-demo file from step 1 is kept for provenance.
#   * Final gate: template normalize bounds must cover SCOUT's abs value
#     range; plus an informational (non-failing) cross-converter report.
# NOTE: make_core_soe's mask key stays "core_20" even for n=40 (hardcoded
# name); round_soe.sh k=0 reads filter key core_20 -- name is cosmetic.
set -eu
VENV=/root/workspace/baojiachun/.venv_soe/bin/python
OUT=/root/workspace/baojiachun/soe_data/datasets/tool_hang
SRC=/root/workspace/baojiachun/scout/data/robomimic/tool_hang/ph/image_v141.hdf5
SCOUTABS=/root/workspace/baojiachun/scout/data/robomimic/tool_hang/ph/image_v141_abs.hdf5
SCOUTCORE=/root/workspace/baojiachun/scout-th95/data/2026_9_5_toolhang/TOOLHANG-s233/tool_hang/rollout/tool_hang_core.hdf5
TEMPLATE=/root/workspace/baojiachun/SOE/simulation/config_template/tool_hang_soe.json
CORE=$OUT/tool_hang_core_soe_s233.hdf5
mkdir -p "$OUT"

# 1) SOE-native conversion + selection-identity cross-check (count/lens/total)
"$VENV" /root/workspace/baojiachun/SOE_scripts_2/make_core_soe.py \
    --src "$SRC" --out-dir "$OUT" --task tool_hang --seeds 233 --n 40 \
    --scout-core "$SCOUTCORE"

# 2) rebuild the core value-identical to SCOUT's (asserts internally)
"$VENV" /root/workspace/baojiachun/SOE_scripts_2/make_core_soe_from_scout.py \
    --scout-core "$SCOUTCORE" --out "$CORE" --n 40

# 3) bounds gate + informational cross-converter report
"$VENV" - "$SCOUTABS" "$OUT/image_v141_abs_6drot.hdf5" "$TEMPLATE" <<'PYEOF'
import sys, json
import h5py
import numpy as np

scout_abs, soe_abs6d, template = sys.argv[1:4]

mins = np.full(3, 1e9); maxs = np.full(3, -1e9)
with h5py.File(scout_abs, "r") as f:
    for k in f["data"]:
        aa = f["data"][k]["abs_actions"][:, :3]
        mins = np.minimum(mins, aa.min(0)); maxs = np.maximum(maxs, aa.max(0))
with open(template) as f:
    cfg = json.load(f)
tmin = np.array(cfg["dataset"]["params"]["normalize_params"]["min_val"][:3])
tmax = np.array(cfg["dataset"]["params"]["normalize_params"]["max_val"][:3])
print("SCOUT abs pos range: %s .. %s" % (np.round(mins, 4), np.round(maxs, 4)))
print("template bounds    : %s .. %s" % (tmin, tmax))
assert np.all(mins >= tmin - 1e-4) and np.all(maxs <= tmax + 1e-4), \
    "template normalize bounds do not cover SCOUT's abs value range"

# informational only: where the two converter lineages disagree (expected:
# step 0 of each demo, ~0.15% of steps; same pattern as historical can)
d0 = dmid = 0.0
with h5py.File(soe_abs6d, "r") as f1, h5py.File(scout_abs, "r") as f2:
    for k in f2["data"]:
        a1 = f1["data"][k]["actions"][:3, :3]
        a2 = f2["data"][k]["abs_actions"][:3, :3]
        d0 = max(d0, float(np.abs(a1[0] - a2[0]).max()))
with h5py.File(soe_abs6d, "r") as f1, h5py.File(scout_abs, "r") as f2:
    for k in f2["data"]:
        a1 = f1["data"][k]["actions"][:, :3]
        a2 = f2["data"][k]["abs_actions"][:, :3]
        dmid = max(dmid, float(np.abs(a1 - a2).max()))
print("cross-converter max pos diff: first-step %.4f / overall %.4f "
      "(expected small; core is SCOUT-derived regardless)" % (d0, dmid))
print("BOUNDS_AND_REPORT_OK")
PYEOF
echo "PREP_TOOLHANG_SOE_DONE"
