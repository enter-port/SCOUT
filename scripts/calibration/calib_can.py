"""CAN exploit-matrix gate-threshold calibration (square-champion transfer).

Square champion vis250t171: thr = p75 of visual-slice (0:1024) NN distance
from reference states to the bank (re-verified 08-30 on the square cache:
p75 = 1.714 -> 1.71). Per (seed, round) here:

  bank      = training data of DP-SCOUT-exp{N} = core + success_1..N
              (appended parts). Built incrementally from per-source-file
              encode caches; each success.hdf5 already contains a core
              copy, so the assembled tensor holds duplicate core frames --
              harmless: NN-min distance is invariant to duplicates.
  VIB / E_s = dyn-SCOUT-exp{N} with cfg.vib.base_dp_ckpt pointed at
              DP-SCOUT-exp{N}/299.ckpt -- mirrors the rollout invocation.
  reference = SCOUT-exp{N}/all.hdf5 (that round's rescue trajectories,
              stride 5, all demos; chain all.hdf5 numbering starts at 0,
              no core demos inside).

Usage: calib_can.py SEED ROUND VIB_CKPT
Env:    CALIB_RULE=p75 (default) | p50 | p90
Writes: <out>/thr_r{N}.txt (two decimals) + prints the quantile table.
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
from scout.guidance.exploit_costs import build_expert_state_bank

REPO = "/root/workspace/baojiachun/scout-exploit"
ENT = "/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy"
VIS = slice(0, 1024)
STRIDE = 5
RULE = os.environ.get("CALIB_RULE", "p75")
QMAP = {"p50": 50, "p75": 75, "p90": 90}


def file_bank(path, vib, views, props, dev, cache):
    if os.path.exists(cache):
        return torch.load(cache, map_location=dev)
    b = build_expert_state_bank(path, vib, view_names=views,
                                proprio_keys=props, device=dev, stride=1)
    torch.save(b.cpu(), cache)
    print(f"[bank] {os.path.basename(path)}: {tuple(b.shape)}", flush=True)
    return b


def main():
    seed, rnd, vibp = sys.argv[1], int(sys.argv[2]), sys.argv[3]
    base = f"{ENT}/CAN-entropy-s{seed}/can"
    out_root = f"{REPO}/data/exploit_can_matrix/s{seed}"
    os.makedirs(out_root, exist_ok=True)
    dp = f"{base}/train/DP/DP-SCOUT-exp{rnd}/checkpoints/299.ckpt"

    cfg = OmegaConf.load(REPO + "/configs/eval_can_entropy.yaml")
    cfg.vib.base_dp_ckpt = dp
    dev = torch.device("cuda")
    vib = make_scout_vib_factory(cfg, dev)(vibp).eval()
    views = list(cfg.eval.view_names)
    props = list(cfg.eval.proprio_keys)
    adapter = make_obs_adapter(views, props)

    parts = [file_bank(f"{base}/rollout/can_core.hdf5", vib, views, props,
                       dev, f"{out_root}/fb_core.pt")]
    for k in range(1, rnd + 1):
        parts.append(file_bank(f"{base}/rollout/SCOUT-exp{k}/success.hdf5",
                               vib, views, props, dev,
                               f"{out_root}/fb_succ{k}.pt"))
    bank_full = torch.cat(parts, dim=0)
    bc = f"{out_root}/bank_r{rnd}_full.pt"
    if not os.path.exists(bc):
        torch.save(bank_full.cpu(), bc)
    bank_vis = bank_full[:, VIS].to(dev).contiguous()
    print(f"[bank] r{rnd} assembled {tuple(bank_full.shape)}", flush=True)

    def nn(q):
        best = None
        for c0 in range(0, bank_vis.shape[0], 8192):
            d = torch.cdist(q, bank_vis[c0:c0 + 8192], p=2).min(-1).values
            best = d if best is None else torch.minimum(best, d)
        return best

    dists = []
    with h5py.File(f"{base}/rollout/SCOUT-exp{rnd}/all.hdf5", "r") as f:
        for k in sorted(f["data"].keys(), key=lambda s: int(s.split("_")[1])):
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
                s = vib.encode(es)[:, VIS]
            dists.append(nn(s).cpu())
    d = torch.cat(dists).numpy()
    print(f"s{seed} r{rnd}: n_states={len(d)} visual NN dist to r{rnd} bank:")
    for q in (5, 10, 25, 50, 75, 90, 95):
        print(f"  p{q:02d} = {np.percentile(d, q):.3f}")
    print(f"  mean = {d.mean():.3f} std = {d.std():.3f} "
          f"frac<0.1 = {(d < 0.1).mean():.3f} frac<0.3 = {(d < 0.3).mean():.3f}",
          flush=True)
    thr = float(np.percentile(d, QMAP[RULE]))
    with open(f"{out_root}/thr_r{rnd}.txt", "w") as f:
        f.write(f"{thr:.2f}\n")
    print(f"[thr] s{seed} r{rnd} rule={RULE} -> {thr:.2f} "
          f"(written thr_r{rnd}.txt)", flush=True)


if __name__ == "__main__":
    main()
