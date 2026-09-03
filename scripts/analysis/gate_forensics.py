"""Gate forensics: per-scene eye-slice NN distance traces from a finished run.

Answers: (a) in the persistent loser scenes {3,51,61,71,80} did the gate ever
OPEN (dist>thr) -- i.e. is the damage attributable to guided chunks, and would
a higher thr have kept them closed? (b) do winner scenes still exceed a higher
thr (would tightening kill the wins)? (c) agentview-slice distance quantiles
for a possible agentview arm. Run on a free GPU (~15 min, bank encode).
"""
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
RUN = ("/root/workspace/baojiachun/scout-exploit/data/"
       "exploit_sq233_r3/sweep/e500t154/all.hdf5")
PER = 512
EYE = slice(PER, 2 * PER)
AGT = slice(0, PER)
STRIDE = 5


def main():
    cfg = OmegaConf.load("/root/workspace/baojiachun/scout-exploit/"
                         "configs/eval_square_entropy.yaml")
    cfg.vib.base_dp_ckpt = DP
    dev = torch.device("cuda")
    vib = make_scout_vib_factory(cfg, dev)(VIB).eval()
    views = list(cfg.eval.view_names)
    props = list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)

    import os
    cache = ("/root/workspace/baojiachun/scout-exploit/data/"
             "exploit_sq233_r3/bank_r2accum_full.pt")
    if os.path.exists(cache):
        bank_full = torch.load(cache, map_location=dev)
        print(f"[cache] bank loaded {tuple(bank_full.shape)}", flush=True)
    else:
        bank_full = build_expert_state_bank(
            BASE + "/rollout/SCOUT-exp2/success_accum.hdf5", vib,
            view_names=views, proprio_keys=props, device=dev, stride=1)
        torch.save(bank_full.cpu(), cache)
        bank_full = bank_full.to(dev)

    bank_eye = bank_full[:, EYE]
    bank_agt = bank_full[:, AGT]

    def nn_dist(q):   # q: (B, 512) on dev, bank pre-sliced
        best = None
        for c0 in range(0, bank_eye.shape[0], 8192):
            d = torch.cdist(q, bank_eye[c0:c0 + 8192], p=2).min(-1).values
            best = d if best is None else torch.minimum(best, d)
        return best

    def nn_dist_agt(q):
        best = None
        for c0 in range(0, bank_agt.shape[0], 8192):
            d = torch.cdist(q, bank_agt[c0:c0 + 8192], p=2).min(-1).values
            best = d if best is None else torch.minimum(best, d)
        return best

    losers = {3, 51, 61, 71, 80}
    winners = {13, 30, 52, 75, 87, 88, 89}
    rows = []
    agt_all = []
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
                sb = vib.encode(es)
            de = nn_dist(sb[:, EYE]).cpu().numpy()
            da = nn_dist_agt(sb[:, AGT]).cpu().numpy()
            agt_all.append(da)
            rows.append((i, de, da))
    agt = np.concatenate(agt_all)
    print("\n=== agentview-slice NN-dist quantiles (all visited states) ===")
    for q in (10, 25, 50, 75, 90, 95):
        print(f"  p{q:02d} = {np.percentile(agt, q):.3f}")

    print("\n=== per-scene eye-dist (stride %d) : p50 p90 max  frac>1.54 "
          "frac>2.0 frac>2.4 | agent p50 p90 ===" % STRIDE)
    for i, de, da in rows:
        tag = ("LOSE" if i in losers else
               "WIN " if i in winners else "    ")
        print(f"  {tag} scene {i:3d}: {np.percentile(de,50):5.2f} "
              f"{np.percentile(de,90):5.2f} {de.max():5.2f}  "
              f"{(de > 1.54).mean():.3f} {(de > 2.0).mean():.3f} "
              f"{(de > 2.4).mean():.3f} | {np.percentile(da,50):5.2f} "
              f"{np.percentile(da,90):5.2f}", flush=True)


if __name__ == "__main__":
    main()
