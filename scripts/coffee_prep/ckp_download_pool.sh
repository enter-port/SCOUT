#!/bin/bash
# ckp_download_pool.sh -- idempotent MimicGen coffee_preparation_d0 pool
# download for COFFEEPREP-MG-p1 (2026-09-20). The registry entry lives in
# mimicgen's DATASET_REGISTRY[core][coffee_preparation_d0] (url
# core/coffee_preparation_d0.hdf5, horizon 800, 1000 mg-generated demos).
# huggingface.co is BLOCKED from the server; hf-mirror.com is reachable
# (verified 200 on 2026-09-20), so we pull the resolve URL from the mirror.
# Threading pool precedent: same host family, 1.7G for threading_d0;
# coffee_preparation_d0 is 7187662878 bytes (HEAD-checked 2026-09-20).
# usage: bash scripts/coffee_prep/ckp_download_pool.sh
set -euo pipefail
ROOT=/root/workspace/baojiachun/scout
MGDIR=$ROOT/data/robomimic/coffee_prep/mg
POOL=$MGDIR/coffee_preparation_d0.hdf5
URL="https://hf-mirror.com/datasets/amandlek/mimicgen_datasets/resolve/main/core/coffee_preparation_d0.hdf5"
EXPECT_BYTES=7187662878

mkdir -p "$MGDIR"
if [ -f "$POOL" ]; then
  SZ=$(stat -c%s "$POOL")
  if [ "$SZ" -ge "$EXPECT_BYTES" ]; then
    echo "[pool] exists and complete ($SZ bytes) -- verify + skip"
  else
    echo "[pool] incomplete ($SZ < $EXPECT_BYTES) -- resuming"
    curl -sL --retry 8 --retry-delay 10 -C - -o "$POOL" "$URL"
  fi
else
  echo "[pool] downloading from hf-mirror ($EXPECT_BYTES expected)"
  curl -sL --retry 8 --retry-delay 10 -C - -o "$POOL" "$URL"
fi

SZ=$(stat -c%s "$POOL")
[ "$SZ" -ge "$EXPECT_BYTES" ] || { echo "[pool] FATAL: size $SZ < $EXPECT_BYTES after download"; exit 1; }

# structure sanity: 1000 demos, env Coffee_Prep_D0, pre-rendered 84x84 2 cams
/root/workspace/baojiachun/.venv_mg/bin/python - "$POOL" <<'PYEOF'
import sys, json
import h5py
f = h5py.File(sys.argv[1], "r")
demos = [k for k in f["data"].keys() if k.startswith("demo")]
assert len(demos) == 1000, f"expected 1000 demos, got {len(demos)}"
ea = json.loads(f["data"].attrs["env_args"])
assert ea["env_name"] == "CoffeePreparation_D0", ea["env_name"]
d = f["data"]["demo_0"]
for k in ("agentview_image", "robot0_eye_in_hand_image"):
    assert d[f"obs/{k}"].shape[1:] == (84, 84, 3), (k, d[f"obs/{k}"].shape)
for k in ("robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"):
    assert k in d["obs"], k
assert d["actions"].shape[-1] == 7, d["actions"].shape
f.close()
print(f"[pool-verify] OK: 1000 demos, env={ea['env_name']}, obs 84x84 2cams, actions 7-dim OSC_POSE")
PYEOF
echo "[pool] DONE -> $POOL ($(stat -c%s "$POOL") bytes)"
