"""EGL-index -> physical-GPU mapping probe (2026-08-22).

For each EGL device index k: create ONE robomimic env with
MUJOCO_EGL_DEVICE_ID=k (CUDA_VISIBLE_DEVICES unset -> robosuite's
egl_context picks all_devices[k] verbatim), step it a few times, then read
per-GPU memory deltas from nvidia-smi IN-PROCESS (context is freed at exit,
so deltas must be taken while alive). The GPU whose VRAM grew is the physical
device EGL[k] actually renders on.

Usage: MUJOCO_EGL_DEVICE_ID=<k> python soe_scripts/egl_map_probe.py
"""
import os
import subprocess
import sys
import time

import numpy as np


def gpu_mem():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits"]).decode()
    return [int(line.split(",")[1]) for line in out.strip().splitlines()]


def main():
    k = os.environ["MUJOCO_EGL_DEVICE_ID"]
    sys.path.insert(0, os.getcwd())
    import yaml
    from omegaconf import OmegaConf
    from scout.eval.rollout import make_robomimic_env_factory

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    before = gpu_mem()
    core = "data/2026_8_21/CAN-exp1-2333/can/rollout/can_core.hdf5"
    with open("configs/base_dp_can_image.yaml") as f:
        lpb = OmegaConf.create(yaml.safe_load(f))
    fac = make_robomimic_env_factory(
        dataset_path=core,
        shape_meta=OmegaConf.to_container(lpb.shape_meta, resolve=True),
        n_obs_steps=2, abs_action=True, render_obs_key="agentview_image")
    env = fac()
    obs = env.reset(seed=7)
    rng = np.random.default_rng(0)
    for _ in range(10):
        obs, _, _, _ = env.step(rng.uniform(-0.3, 0.3, size=10))
    time.sleep(2)                       # let nvidia-smi accounting settle
    after = gpu_mem()
    delta = [a - b for a, b in zip(after, before)]
    grew = [i for i, d in enumerate(delta) if d > 50]
    live = float(np.asarray(obs["agentview_image"], dtype=np.float32).std())
    print(f"EGL[{k}] memdelta_mi={delta} -> physical_gpu={grew} "
          f"obs_std={live:.3f}", flush=True)


if __name__ == "__main__":
    main()
