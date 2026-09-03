"""Dose calibration for the CAN exploit matrix (unit conversion of the
square champion eta=250).

The square champion eta=250 was calibrated against the |grad_a cost| scale
of the SQUARE dyn-SCOUT-exp3 VIB (calib_exploit.py recipe). Verbatim 250
on the CAN per-round VIBs overdoses catastrophically (08-30 evidence:
explore_solved 0-1/100 vs chain 0.68-0.86, jerk 2.3-5.2 vs chain ~0.25,
identical across threshold regimes 2.35/1.60/0.02 -> dose, not gate).
This probe measures |grad_a cost|/row with the IDENTICAL recipe (tiny
bank = query frames, identity bridge, visual slice, core-demo expert
chunks) on the square reference config and on every can (seed, round)
config, then writes eta_r{N}.txt = 250 * g_sq / g_can -- i.e. the same
effective inject magnitude eta*|grad| as the champion.

Usage: can_dose_probe.py            # square ref + all 18 can configs
       can_dose_probe.py SQUARE     # reference only
       can_dose_probe.py 233 3      # single can config
"""
import os
import sys

sys.path.insert(0, "/root/workspace/baojiachun/scout-exploit")

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from scout.eval.factories import make_scout_vib_factory
from scout.eval.rollout import make_obs_adapter

REPO = "/root/workspace/baojiachun/scout-exploit"
SQ = ("/root/workspace/baojiachun/scout-entropy/data/"
      "2026_8_26_entropy/SQUARE-entropy-s233/square")
ENT = "/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy"
ETA_SQ = 250.0

TS = {
    233: {1: "20260825-074749", 2: "20260825-122009", 3: "20260825-185709",
          4: "20260825-230040", 5: "20260826-030203", 6: "20260826-074559"},
    2333: {1: "20260825-084315", 2: "20260825-182721", 3: "20260825-221728",
           4: "20260826-024512", 5: "20260826-082627", 6: "20260826-150122"},
    23333: {1: "20260825-071516", 2: "20260825-102750", 3: "20260825-133347",
            4: "20260825-181256", 5: "20260825-215517", 6: "20260826-015139"},
}


def pin_s_bar(planner, s_bar):
    planner._cached_s_bar_t = s_bar
    planner._cached_obs_id = None


def measure(cfg_path, dp, vibp, core, dev):
    cfg = OmegaConf.load(cfg_path)
    cfg.vib.base_dp_ckpt = dp
    vib = make_scout_vib_factory(cfg, dev)(vibp).eval()
    views = list(cfg.eval.view_names)
    props = list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)
    from diffusion_policy.dataset.robomimic_replay_image_dataset import (
        _convert_actions,
    )
    from diffusion_policy.model.common.rotation_transformer import (
        RotationTransformer,
    )
    rt = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    with h5py.File(core, "r") as f:
        d = f["data/demo_0"]
        T = d["actions"].shape[0]
        fr = np.arange(0, T - 8, 40)[:6]
        obs = {v: torch.from_numpy(
                   np.ascontiguousarray(d[f"obs/{v}"][:][fr])
               ).permute(0, 3, 1, 2).float().unsqueeze(1) for v in views}
        for k in props:
            obs[k] = torch.from_numpy(
                np.ascontiguousarray(d[f"obs/{k}"][:][fr])
            ).float().unsqueeze(1)
        acts7 = d["abs_actions"][:]
    es = adapter(obs)
    es = {"visual": {v: x.to(dev) for v, x in es["visual"].items()},
          "proprio": es["proprio"].to(dev)}
    with torch.no_grad():
        s_bar = vib.encode(es)
    ach = np.stack([_convert_actions(acts7[t:t + 8].astype(np.float64),
                                     True, rt).reshape(-1) for t in fr])
    ach = torch.from_numpy(ach).float().to(dev)
    from scout.guidance.exploit_costs import ExploitCostPlanner
    from scout.normalizer import IdentityBridge
    pl = ExploitCostPlanner(vib, state_bank=s_bar.detach().clone(),
                            bridge=IdentityBridge(), latent="visual")
    pin_s_bar(pl, s_bar)
    a = ach.clone().requires_grad_(True)
    loss = pl.compute_loss(a.unsqueeze(1), None, reduction="sum")
    g = torch.autograd.grad(loss, a)[0]
    return float(g.norm(dim=-1).mean()), float(loss) / ach.shape[0]


def main():
    dev = torch.device("cuda")
    args = sys.argv[1:]
    if args and args[0] != "SQUARE":
        seed, rnd = int(args[0]), int(args[1])
        base = f"{ENT}/CAN-entropy-s{seed}/can"
        g, c = measure(f"{REPO}/configs/eval_can_entropy.yaml",
                       f"{base}/train/DP/DP-SCOUT-exp{rnd}/checkpoints/299.ckpt",
                       f"{base}/train/dyn/dyn-SCOUT-exp{rnd}/{TS[seed][rnd]}/scout_vib.ckpt",
                       f"{base}/rollout/can_core.hdf5", dev)
        print(f"s{seed} r{rnd}: |grad|/row={g:.4f} cost/row={c:.4f}")
        return
    g_sq, c_sq = measure(f"{REPO}/configs/eval_square_entropy.yaml",
                         SQ + "/train/DP/DP-SCOUT-exp3/checkpoints/299.ckpt",
                         SQ + "/train/dyn/dyn-SCOUT-exp3/20260827-145511/scout_vib.ckpt",
                         SQ + "/rollout/square_core.hdf5", dev)
    print(f"SQUARE-ref: |grad|/row={g_sq:.4f} cost/row={c_sq:.4f} "
          f"(champion eta=250)", flush=True)
    if args and args[0] == "SQUARE":
        return
    for seed in (233, 2333, 23333):
        base = f"{ENT}/CAN-entropy-s{seed}/can"
        out_root = f"{REPO}/data/exploit_can_matrix/s{seed}"
        for rnd in range(1, 7):
            g, c = measure(f"{REPO}/configs/eval_can_entropy.yaml",
                           f"{base}/train/DP/DP-SCOUT-exp{rnd}/checkpoints/299.ckpt",
                           f"{base}/train/dyn/dyn-SCOUT-exp{rnd}/{TS[seed][rnd]}/scout_vib.ckpt",
                           f"{base}/rollout/can_core.hdf5", dev)
            eta = ETA_SQ * g_sq / g
            with open(f"{out_root}/eta_r{rnd}.txt", "w") as f:
                f.write(f"{eta:.2f}\n")
            print(f"s{seed} r{rnd}: |grad|/row={g:.4f} cost/row={c:.4f} "
                  f"-> eta={eta:.2f}", flush=True)


if __name__ == "__main__":
    main()
