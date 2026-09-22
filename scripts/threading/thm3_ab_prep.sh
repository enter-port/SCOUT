#!/bin/bash
# thm3_ab_prep.sh -- stage-1 data prep for the THM3 A/B undertraining study
# (user plan 2026-09-23, autonomous session). Builds the DEDUPED success-side
# file for the B arms (DP retrain data: core20 + one success per solved
# scene, from s233 ATY r1) and VERIFIES the A2 dyn data (the existing
# 73-demo deduped all_accum from the 09-23 dedup experiment -- its zarr +
# featbank caches are valid for E_s=DP-ATY-exp1, so we reuse, not rebuild).
# Outputs under data/2026_9_22_threading_mg_p2/PROBE_AB/ (outside the
# per-seed trees so chain accum globs never see them). Idempotent.
# usage: bash scripts/threading/thm3_ab_prep.sh
set -u
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
CAMP=$ROOT/data/2026_9_22_threading_mg_p2
S=$CAMP/THREADING-MG-p2-s233/threading
EXP=$CAMP/PROBE_AB
mkdir -p "$EXP"
cd "$ROOT" || exit 1
CORE=$S/rollout/threading_core.hdf5
SUCC1=$S/rollout/ATY-exp1/success.hdf5
ESBASE=$S/train/DP/DP-base/checkpoints/599.ckpt
ESATY1=$S/train/DP/DP-ATY-exp1/checkpoints/299.ckpt
DEDUP73=$CAMP/PROBE_DEDUP/all_accum_dedup_r1.hdf5
for f in "$CORE" "$SUCC1" "$ESBASE" "$ESATY1" "$DEDUP73"; do
  [ -f "$f" ] || { echo "[prep] FATAL missing $f"; exit 1; }
done
ndemos(){ $PY -c "import h5py,sys; print(len(h5py.File(sys.argv[1],'r')['data'].keys()))" "$1"; }

# 1) dedup the r1 SUCCESS file: ONE success per scene (lowest id), core kept
DEDUP_S=$EXP/succ_dedup_r1.hdf5
ACCUM_S=$EXP/succ_accum_dedup_r1.hdf5
if [ -f "$ACCUM_S" ]; then
  N=$(ndemos "$ACCUM_S")
  if [ "$N" = "43" ]; then echo "[prep] $ACCUM_S exists with $N demos -- skip rebuild"; else
    echo "[prep] WARN existing $ACCUM_S has $N demos != 43 -- rebuilding"; fi
fi
if [ ! -f "$ACCUM_S" ] || [ "$N" != "43" ]; then
$PY - "$SUCC1" "$DEDUP_S" <<'PYEOF'
import sys, hashlib, shutil
import h5py, numpy as np
succ_path, out_path = sys.argv[1:3]
def h0(g):
    st = g["states"][0] if "states" in g else g["obs/robot0_eef_pos"][0]
    return hashlib.md5(np.ascontiguousarray(st).tobytes()).hexdigest()
def did(k): return int(k.split("_")[-1])
fs = h5py.File(succ_path, "r")
expl = [d for d in sorted(fs["data"].keys(), key=did) if did(d) >= 20]
groups = {}
for d in expl:
    groups.setdefault(h0(fs[f"data/{d}"]), []).append(d)
keep = {min(dd, key=did) for dd in groups.values()}   # all are successes
drop = [d for d in expl if d not in keep]
print(f"[prep] succ dedup: scenes={len(groups)} kept={len(keep)} "
      f"dropped={len(drop)} raw={len(expl)}")
fs.close()
shutil.copyfile(succ_path, out_path)
with h5py.File(out_path, "r+") as fo:
    for d in drop:
        del fo[f"data/{d}"]
    fo.attrs["total"] = len(keep) + 20
print(f"[prep] wrote {out_path}")
PYEOF
[ -f "$DEDUP_S" ] || { echo "[prep] FATAL dedup write failed"; exit 1; }
$PY - "$CORE" "$DEDUP_S" "$ACCUM_S" <<'PYEOF'
import sys, os
sys.path.insert(0, os.getcwd())
from scout.eval.hdf5_writer import merge_accumulated_hdf5
print("[prep] accum:", merge_accumulated_hdf5(sys.argv[1], [sys.argv[2]], sys.argv[3]))
PYEOF
fi
N=$(ndemos "$ACCUM_S")
echo "[prep] B-arm DP data: $ACCUM_S ($N demos; expect 43 = core20 + 23 scenes)"
[ "$N" = "43" ] || { echo "[prep] FATAL demo count $N != 43"; exit 1; }

# 2) A-arm dyn data: verify only (caches alongside are valid -- do not touch)
N=$(ndemos "$DEDUP73")
echo "[prep] A-arm dyn data: $DEDUP73 ($N demos; expect 73)"
[ "$N" = "73" ] || { echo "[prep] FATAL A2 data count $N != 73"; exit 1; }
echo "[prep] ALL DONE"
