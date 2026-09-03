"""Visual-slice gate calibration against a CORE-only bank (20 expert demos).
CPU variant -- run while GPUs are busy. States sampled from the champion run
(vis250t171 trajectories). Output: quantiles -> thr for a core-bank arm."""
import sys

sys.path.insert(0, "/root/workspace/baojiachun/scout-exploit")

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf

from scout.eval.factories import make_scout_vib_factory
from scout.eval.rollout import make_obs_adapter
from scout.guidance.exploit_costs import build_expert_state_bank

BASE = ("/root/workspace/baojiachun/scout-entropy/data/"
        "2026_8_26_entropy/SQUARE-entropy-s233/square")
DP = BASE + "/train/DP/DP-SCOUT-exp3/checkpoints/299.ckpt"
VIB = BASE + "/train/dyn/dyn-SCOUT-exp3/20260827-145511/scout_vib.ckpt"
CORE = BASE + "/rollout/square_core.hdf5"
RUN = ("/root/workspace/baojiachun/scout-exploit/data/"
       "exploit_sq233_r3/sweep/vis250t171/all.hdf5")
SL = slice(0, 1024)
STRIDE = 5


def main():
    cfg = OmegaConf.load("/root/workspace/baojiachun/scout-exploit/"
                         "configs/eval_square_entropy.yaml")
    cfg.vib.base_dp_ckpt = DP
    dev = torch.device("cpu")
    torch.set_num_threads(16)
    vib = make_scout_vib_factory(cfg, dev)(VIB).eval()
    views = list(cfg.eval.view_names)
    props = list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)

    bank = build_expert_state_bank(
        CORE, vib, view_names=views, proprio_keys=props,
        device=dev, stride=1, batch=32)[:, SL]
    print(f"core bank: {tuple(bank.shape)}", flush=True)

    def nn(q):
        best = None
        for c0 in range(0, bank.shape[0], 4096):
            d = torch.cdist(q, bank[c0:c0 + 4096], p=2).min(-1).values
            best = d if best is None else torch.minimum(best, d)
        return best

    dists = []
    with h5py.File(RUN, "r") as f:
        for k in sorted(f["data"].keys(),
                        key=lambda s: int(s.split("_")[1])):
            if int(k.split("_")[1]) < 20:
                continue
            g = f["data/" + k]
            T = g["actions"].shape[0]
            fr = np.arange(0, T, STRIDE)
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
            es = {"visual": {v: x for v, x in es["visual"].items()},
                  "proprio": es["proprio"]}
            with torch.no_grad():
                s = vib.encode(es)[:, SL]
            dists.append(nn(s))
            print(".", end="", flush=True)
    d = torch.cat(dists).numpy()
    print(f"\nn_states={len(d)}  visual-dist to CORE bank:")
    for q in (10, 25, 50, 75, 90, 95):
        print(f"  p{q:02d} = {np.percentile(d, q):.3f}")
    print(f"  mean = {d.mean():.3f}  std = {d.std():.3f}")


if __name__ == "__main__":
    main()
