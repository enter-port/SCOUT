"""Calibrate the exploit OOD-gate threshold: NN-distance distribution of the
states the policy ACTUALLY visits (last run's all.hdf5 trajectories) to the
r2 bank, in the eye slice. Quantiles -> threshold candidates.
"""
import sys

sys.path.insert(0, "/root/workspace/baojiachun/scout-exploit")

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from scout.eval.factories import make_scout_vib_factory
from scout.eval.rollout import make_obs_adapter

BASE = ("/root/workspace/baojiachun/scout-entropy/data/"
        "2026_8_26_entropy/SQUARE-entropy-s233/square")
DP = BASE + "/train/DP/DP-SCOUT-exp3/checkpoints/299.ckpt"
VIB = BASE + "/train/dyn/dyn-SCOUT-exp3/20260827-145511/scout_vib.ckpt"
RUN = ("/root/workspace/baojiachun/scout-exploit/data/"
       "exploit_sq233_r3/onetry/exploit/all.hdf5")
PER_VIEW = 512
SL = slice(PER_VIEW, 2 * PER_VIEW)     # eye (matches --exploit-latent eye)


def main():
    cfg = OmegaConf.load("/root/workspace/baojiachun/scout-exploit/"
                         "configs/eval_square_entropy.yaml")
    cfg.vib.base_dp_ckpt = DP
    dev = torch.device("cuda")
    vib = make_scout_vib_factory(cfg, dev)(VIB).eval()

    views = list(cfg.eval.view_names)
    props = list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)

    # bank = r2 success_accum frames (the rollout bank, eye-sliced)
    from scout.guidance.exploit_costs import build_expert_state_bank
    bank_full = build_expert_state_bank(
        BASE + "/rollout/SCOUT-exp2/success_accum.hdf5", vib,
        view_names=views, proprio_keys=props, device=dev, stride=1)
    bank = bank_full[:, SL]

    # sample states from the policy's own trajectories (stride 10)
    dists = []
    with h5py.File(RUN, "r") as f:
        demos = sorted(f["data"].keys())[20:]        # skip core prefix
        for k in demos:
            g = f[f"data/{k}"]
            T = g["actions"].shape[0]
            fr = np.arange(0, T, 10)
            if len(fr) == 0:
                continue
            obs = {v: torch.from_numpy(
                       np.ascontiguousarray(g[f"obs/{v}"][:][fr])
                   ).permute(0, 3, 1, 2).float().unsqueeze(1) for v in views}
            for p_ in props:
                obs[p_] = torch.from_numpy(
                    np.ascontiguousarray(g[f"obs/{p_}"][:][fr])
                ).float().unsqueeze(1)
            es = adapter(obs)
            es = {"visual": {v: x.to(dev) for v, x in es["visual"].items()},
                  "proprio": es["proprio"].to(dev)}
            with torch.no_grad():
                s = vib.encode(es)[:, SL]
            for c0 in range(0, bank.shape[0], 8192):
                d = torch.cdist(s, bank[c0:c0 + 8192], p=2).min(-1).values
                dists.append(d.cpu())
    d = torch.cat(dists).numpy()
    print(f"n_states={len(d)}  eye-slice NN distance to r2 bank:")
    for q in (10, 25, 50, 75, 90, 95):
        print(f"  p{q:02d} = {np.percentile(d, q):.4f}")
    print(f"  mean = {d.mean():.4f}  std = {d.std():.4f}")


if __name__ == "__main__":
    main()
