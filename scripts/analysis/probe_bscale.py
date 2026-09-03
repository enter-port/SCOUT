"""B-invariance probe: does the guided action for row 0 depend on batch size?

Mirrors the production path exactly (same factories, same planner attach,
same predict_action_dyn_guided call as rollout_vec._replan). One REAL obs
frame from the chain's core hdf5 is replicated to B rows; z and the torch
seed are fixed, so row 0's noise stream is identical for every B. If the
1/B fix is airtight, row 0's guided action must match across B up to
float noise (~1e-6).

Run on the server:  PROBE_GPU=0 .venv/bin/python probe_bscale.py
"""
import os, sys, time

os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("PROBE_GPU", "0")
os.environ.setdefault("MUJOCO_GL", "egl")

import torch
import h5py
import numpy as np

sys.path.insert(0, "/root/workspace/baojiachun/scout")
from scout.eval.factories import load_cfg, make_lpb_dp_factory, make_scout_vib_factory
from scout.eval.rollout import make_obs_adapter, make_action_bridge
from scout.guidance.planner import ScoutPlanner

SEED_DIR = os.environ.get(
    "SEED_DIR",
    "/root/workspace/baojiachun/scout/data/2026_8_21/CAN-exp1-23333/can")
DPC = os.path.join(SEED_DIR, "train/DP/DP-SCOUT-exp3/checkpoints/99.ckpt")
VIB = os.path.join(SEED_DIR, "train/dyn/dyn-SCOUT-exp3/20260823-002449/scout_vib.ckpt")
CFG = "/root/workspace/baojiachun/scout/configs/eval_can_exp1.yaml"
CORE = os.path.join(SEED_DIR, "rollout/can_core.hdf5")

t0 = time.time()
cfg = load_cfg(CFG)
# mirror run_rollout.py:158 -- E_s is built from the ROLLOUT DP ckpt
cfg.vib.ckpt_path = VIB
cfg.vib.base_dp_ckpt = DPC
dev = torch.device("cuda:0")
print("loading DP ...", flush=True)
dp = make_lpb_dp_factory(dev)(DPC)
print("loading VIB ...", flush=True)
vib = make_scout_vib_factory(cfg, dev)(VIB)
bridge = make_action_bridge(dp)
obs_adapter = make_obs_adapter(list(cfg.eval.view_names), list(cfg.eval.proprio_keys))
planner = ScoutPlanner(vib, bridge=bridge, obs_adapter=obs_adapter, z=None)
dp.initialize_scout_planner(
    planner,
    int(cfg.exploration.guidance_start_timestep),
    float(cfg.exploration.guidance_scale))
print(f"models ready {time.time()-t0:.0f}s  guidance_scale="
      f"{cfg.exploration.guidance_scale}  start_t="
      f"{cfg.exploration.guidance_start_timestep}", flush=True)

# ---- one real obs window (2 frames, t=9..10 of demo_0) ----
To = 2
t = 10
with h5py.File(CORE, "r") as f:
    d = f["data/demo_0/obs"]

    def img(k):
        a = d[k][t - To + 1:t + 1]                     # (To,H,W,C) uint8
        return np.transpose(a, (0, 3, 1, 2)).astype(np.float32) / 255.0

    obs1 = {k: img(k) for k in cfg.eval.view_names}
    for k in cfg.eval.proprio_keys:
        obs1[k] = d[k][t - To + 1:t + 1].astype(np.float32)


def batchify(B):
    return {k: torch.from_numpy(np.repeat(v[None], B, axis=0)).to(dev)
            for k, v in obs1.items()}


torch.manual_seed(7)
z1 = torch.randn(1, int(vib.style_dim), device=dev)

BATCHES = [1, 4, 12, 25, 50]
outs = {}
for B in BATCHES:
    ob = batchify(B)
    planner.set_z(z1.repeat(B, 1))
    torch.manual_seed(1234)   # same init-noise stream -> row 0 identical across B
    r = dp.predict_action_dyn_guided(ob)
    outs[B] = r["action"][0].detach().cpu()
    print(f"B={B:3d} done {time.time()-t0:.0f}s", flush=True)

ref = outs[1]
print("\n=== row-0 guided action vs B=1 reference ===")
for B in BATCHES:
    diff = (outs[B] - ref).abs()
    print(f"B={B:3d}  max|d|={diff.max().item():.3e}  "
          f"rms={diff.pow(2).mean().sqrt().item():.3e}")

# guidance bite: guided vs unguided row 0 (B=12)
ob = batchify(12)
torch.manual_seed(1234)
with torch.no_grad():
    ug = dp.predict_action(ob)["action"][0].detach().cpu()
bite = (outs[12] - ug).abs()
print(f"\nguidance bite (B=12): max|guided-unguided|={bite.max().item():.4f} "
      f"rms={bite.pow(2).mean().sqrt().item():.4f} "
      f"(action abs-mean={ref.abs().mean().item():.4f})")
print("PROBE DONE")
