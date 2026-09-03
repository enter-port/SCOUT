"""Gate forensics v2: per-scene VISUAL-slice (0:1024) NN distance traces from
the vis400t154 run, with frac-above thresholds 1.71 / 1.9 / 2.24 / 2.6.
Question: do the vis-CRACK scenes {60,71,72,82,83,98} open the gate at higher
distances than the vis-DROP scenes {68,85,87,88,89}? -> pick a separating thr.
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
       "exploit_sq233_r3/sweep/vis400t171/all.hdf5")
CACHE = ("/root/workspace/baojiachun/scout-exploit/data/"
         "exploit_sq233_r3/bank_r2accum_full.pt")
SL = slice(0, 1024)
STRIDE = 5
THRS = (1.71, 1.9, 2.24, 2.6)

CRACKS = {60, 71, 72, 82, 83, 98}
DROPS = {68, 85, 87, 88, 89}
PIPEFAIL = {3, 51, 61, 80}


def main():
    cfg = OmegaConf.load("/root/workspace/baojiachun/scout-exploit/"
                         "configs/eval_square_entropy.yaml")
    cfg.vib.base_dp_ckpt = DP
    dev = torch.device("cuda")
    vib = make_scout_vib_factory(cfg, dev)(VIB).eval()
    views = list(cfg.eval.view_names)
    props = list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)

    bank = torch.load(CACHE, map_location=dev)[:, SL]

    def nn(q):
        best = None
        for c0 in range(0, bank.shape[0], 8192):
            d = torch.cdist(q, bank[c0:c0 + 8192], p=2).min(-1).values
            best = d if best is None else torch.minimum(best, d)
        return best

    rows = []
    with h5py.File(RUN, "r") as f:
        for k in sorted(f["data"].keys(),
                        key=lambda s: int(s.split("_")[1])):
            i = int(k.split("_")[1]) - 20
            if i < 0:
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
            es = {"visual": {v: x.to(dev) for v, x in es["visual"].items()},
                  "proprio": es["proprio"].to(dev)}
            with torch.no_grad():
                s = vib.encode(es)[:, SL]
            rows.append((i, nn(s).cpu().numpy()))

    hdr = "  ".join(f"f>{t}" for t in THRS)
    print(f"=== per-scene visual-dist: p50 p90 max  {hdr} ===")
    for i, d in rows:
        tag = ("CRACK" if i in CRACKS else
               "DROP " if i in DROPS else
               "PFAIL" if i in PIPEFAIL else "     ")
        fracs = "  ".join(f"{(d > t).mean():.3f}" for t in THRS)
        print(f"  {tag} scene {i:3d}: {np.percentile(d,50):5.2f} "
              f"{np.percentile(d,90):5.2f} {d.max():5.2f}  {fracs}",
              flush=True)


if __name__ == "__main__":
    main()
