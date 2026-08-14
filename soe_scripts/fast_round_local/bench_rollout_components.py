"""Component benchmark for the square rollout (policy vs env cost split).

Times, on the real square ckpts:
  1. predict_action (unguided) and predict_action_dyn_guided at B=50 and B=100
     -- per-call wall time (the guided loop does UNet fwd + 50x UNet bwd).
  2. env.step per-tick cost for 50 and 100 robomimic envs (EGL offscreen
     render 2 views + sim), the CPU side of the vec rollout.

Run on the server:  MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 python bench_rollout_components.py
"""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
import torch

sys.path.insert(0, "/root/workspace/baojiachun/scout")
os.chdir("/root/workspace/baojiachun/scout")

from scout.eval.factories import (load_cfg, make_lpb_dp_factory,
                                  make_scout_vib_factory, make_default_env_factory)
from scout.eval.rollout import make_action_bridge, make_obs_adapter
from scout.guidance.planner import ScoutPlanner

cfg = load_cfg("configs/eval_square.yaml")
dev = torch.device("cuda")

t0 = time.time()
dp = make_lpb_dp_factory(dev)("data/square/train/DP-square-base/checkpoints/160.ckpt")
vib = make_scout_vib_factory(cfg, dev)("data/square/train/square-SCOUT/scout_vib.ckpt")
planner = ScoutPlanner(vib, bridge=make_action_bridge(dp),
                       obs_adapter=make_obs_adapter(cfg.eval.view_names,
                                                    cfg.eval.proprio_keys), z=None)
dp.initialize_scout_planner(planner, 50, 5.0)
print(f"[load] ckpts in {time.time()-t0:.0f}s")

def mkobs(B):
    g = torch.Generator().manual_seed(0)
    return {
        "agentview_image": (torch.rand(B, 2, 3, 84, 84, generator=g) * 255).float(),
        "robot0_eye_in_hand_image": (torch.rand(B, 2, 3, 84, 84, generator=g) * 255).float(),
        "robot0_eef_pos": torch.randn(B, 2, 3, generator=g),
        "robot0_eef_quat": torch.randn(B, 2, 4, generator=g),
        "robot0_gripper_qpos": torch.randn(B, 2, 2, generator=g),
    }

for B in (50, 100):
    obs = mkobs(B)
    for name, fn, n in (("unguided", dp.predict_action, 5),
                        ("GUIDED  ", dp.predict_action_dyn_guided, 3)):
        for _ in range(2):
            fn(obs)
        torch.cuda.synchronize(); t = time.time()
        for _ in range(n):
            fn(obs)
        torch.cuda.synchronize()
        ms = (time.time() - t) / n * 1000
        print(f"[policy] {name} B={B}: {ms:7.0f} ms/call  ({ms/B:.1f} ms/row)")

envf = make_default_env_factory(cfg)
envs = []
t = time.time()
for i in range(100):
    envs.append(envf())
    if i + 1 in (50, 100):
        print(f"[env] created {i+1} envs in {time.time()-t:.0f}s")
act = np.zeros(7, dtype=np.float32)

def tick(es):
    for e in es:
        e.step(act)

for label, es in (("50", envs[:50]), ("100", envs)):
    for e in es:
        e.reset_to(e.get_state())
    for _ in range(10):
        tick(es)
    t = time.time(); T = 100
    for _ in range(T):
        tick(es)
    dt = (time.time() - t) / T
    print(f"[env] n={len(es)}: {dt*1000:6.0f} ms/tick  ({dt/len(es)*1000:.2f} ms/env-step)")
print("bench done")
