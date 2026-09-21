#!/bin/bash
# cfk_prep_data.sh -- per-seed core20 materialization for COFFEE-MG-p1
# (2026-09-20). COPY of scripts/coffee_prep/ckp_prep_data.sh with these
# deltas ONLY: task = coffee (MimicGen coffee_d0 pool, env_name Coffee_D0,
# 400 mg-generated demos; paths 2026_9_20_coffee_mg_p1). Everything else
# identical: MimicGen pool (obs PRE-RENDERED 84x84 agentview_image +
# robot0_eye_in_hand_image HWC uint8, states, actions (T,7) OSC_POSE) ->
#   1) seeded split (TSEED) to 20 demos via scripts/analysis/split_core.py
#   2) robomimic_dataset_conversion adds abs_actions (7-dim axis-angle) by
#      replaying states in Coffee_D0 (no rendering; no EGL needed)
#   3) rollout/coffee_core.hdf5 in the campaign root + structure verify
#   4) ONE-TIME double-seed reset reproducibility probe (the fixed-100-scene
#      eval protocol rests on np.random.seed(42+i)+reset(); robosuite 1.4.1
#      placement samplers draw from the numpy GLOBAL rng -- verify bitwise).
# Runs under .venv_mg (mimicgen must be importable for env creation; the
# robomimic env_robosuite hook imports it).
# usage: bash scripts/coffee/cfk_prep_data.sh <SEED>
set -euo pipefail
SEED=${1:?usage: cfk_prep_data.sh <SEED>}
ROOT=/root/workspace/baojiachun/scout
PY=/root/workspace/baojiachun/.venv_mg/bin/python
POOL=$ROOT/data/robomimic/coffee/mg/coffee_d0.hdf5
DST=$ROOT/data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s$SEED/coffee
CORE=$DST/rollout/coffee_core.hdf5
STAGE=$DST/rollout/.stage_core_noabs_s$SEED.hdf5

[ -f "$POOL" ] || { echo "[prep] FATAL: pool missing: $POOL (run cfk_download_pool.sh)"; exit 1; }
[ -f "$CORE" ] && { echo "[prep] core exists -- skip ($CORE)"; exit 0; }

# CPFS disk-write probe (df untrustworthy, 2026-09-10 lesson)
dd if=/dev/zero of=$ROOT/data/.cfk_prep_probe bs=1M count=512 oflag=direct 2>/dev/null \
  || dd if=/dev/zero of=$ROOT/data/.cfk_prep_probe bs=1M count=512
rm -f $ROOT/data/.cfk_prep_probe
echo "[prep] disk write probe OK"
mkdir -p "$DST/rollout"

echo "[prep] s$SEED step1: seeded split 20 of 1000 (rng $SEED)"
"$PY" scripts/analysis/split_core.py "$POOL" "$STAGE" 20 "$SEED"

echo "[prep] s$SEED step2: abs_actions conversion (state replay in Coffee_D0)"
(cd "$ROOT" && "$PY" -m diffusion_policy.scripts.robomimic_dataset_conversion \
  -i "$STAGE" -o "$CORE" -n 4)

echo "[prep] s$SEED step3: verify + one-time reset-reproducibility probe"
"$PY" - "$CORE" "$POOL" <<'PYEOF'
import sys, json
import numpy as np
import h5py

core, pool = sys.argv[1], sys.argv[2]
f = h5py.File(core, "r")
demos = sorted([k for k in f["data"].keys()], key=lambda k: int(k.split("_")[-1]))
assert len(demos) == 20, f"expected 20 demos, got {len(demos)}"
d = f["data"][demos[0]]
assert d["abs_actions"].shape[-1] == 7, d["abs_actions"].shape
assert d["actions"].shape == d["abs_actions"].shape
for k in ("agentview_image", "robot0_eye_in_hand_image"):
    assert d[f"obs/{k}"].shape[1:] == (84, 84, 3), (k, d[f"obs/{k}"].shape)
env_args = json.loads(f["data"].attrs["env_args"])
assert env_args["env_name"] == "Coffee_D0", env_args["env_name"]
f.close()
print(f"[prep-verify] core OK: 20 demos, abs_actions 7-dim, obs 84x84x2cams, env={env_args['env_name']}")

# fixed-scene protocol probe: same seed -> bitwise-same init state; different
# seed -> different. Fresh env per draw, exactly like collect_initial_states.
# (ObsUtils init = same thing make_robomimic_env_factory does before env
# creation; without it get_observation hits an empty modality mapping.)
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils
ObsUtils.initialize_obs_modality_mapping_from_dict({
    "rgb": ["agentview_image", "robot0_eye_in_hand_image"],
    "low_dim": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"],
})

meta = FileUtils.get_env_metadata_from_dataset(pool)
meta["env_kwargs"]["use_object_obs"] = False

def state_after(seed):
    env = EnvUtils.create_env_from_metadata(
        env_meta=meta, render=False, render_offscreen=False, use_image_obs=False)
    np.random.seed(seed)
    env.reset()
    s = np.array(env.env.sim.get_state().flatten())
    del env          # EnvRobosuite has no close(); no render context here
    return s

s42a, s42b, s43 = state_after(42), state_after(42), state_after(43)
assert np.array_equal(s42a, s42b), "seed 42 NOT reproducible -- fixed-scene protocol BROKEN"
assert not np.array_equal(s42a, s43), "seed 43 identical to 42 -- scenes would degenerate"
print("[prep-probe] reset reproducibility OK: seed42 bitwise-equal across fresh envs, seed43 differs")
PYEOF

rm -f "$STAGE"
echo "[prep] s$SEED DONE -> $CORE ($(stat -c%s "$CORE") bytes)"
