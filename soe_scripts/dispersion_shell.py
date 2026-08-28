"""Per-scene retry-code dispersion: 方案三 (atypical) vs 方案A (shell).

The motivation experiment for 方案A: SCOUT retries share the J^T*Lambda*J
escape direction (narrow cone) while SOE sprays.  This script QUANTIFIES the
retry spread actually executed on each failed scene:

  1. group a run's rescue demos (all.hdf5 beyond the 20 core demos) by their
     initial-state fingerprint (rescue retries start from the SAME scene
     init, so states[0] identifies the scene -- no index bookkeeping needed);
  2. per demo, encode every action chunk (s_bar_t, chunk) with the frozen
     VIB encoder and average the means -> one trajectory-level behavior
     summary mu_bar (16-d);
  3. per scene group (>=2 retries): mean pairwise L2 of mu_bar and the
     participation ratio PR = (tr S)^2 / tr S^2 of the group's centered
     second moment -- PR ~= 1 means all retries moved along ONE direction
     (cone), PR ~= min(n,16) means isotropic spray.

Run (server):
  python soe_scripts/dispersion_shell.py --config configs/eval_can_entropy.yaml \
      --vib-ckpt <dyn-base>/scout_vib.ckpt --n-core 20 \
      --run plan3_env50=data/.../PROBE-base-pass10/all.hdf5 \
      --run planA_eta0p35=data/.../PROBE-shellA-pass10-eta0p35/all.hdf5
"""
from __future__ import annotations

import argparse
import hashlib

import numpy as np
import torch


def _fingerprint(state0: np.ndarray) -> str:
    return hashlib.md5(np.asarray(state0).tobytes()).hexdigest()[:12]


def encode_run(all_hdf5: str, scout_vib, adapter, view_names, proprio_keys,
               device, n_core: int, batch: int = 64, stride_chunks: int = 1,
               action_key: str = "abs_actions"):
    """-> dict fp -> list of per-demo mu_bar (np.ndarray [16])."""
    import h5py

    from scout.guidance.expert_bank import _default_aa_to_6d

    vib_enc = scout_vib.vib_enc
    chunk_dim = int(vib_enc.action_dim)          # 80 = 8 steps x 10
    n_steps = chunk_dim // 10
    groups: dict = {}
    with h5py.File(all_hdf5, "r") as f:
        demos = sorted((k for k in f["data"].keys() if k.startswith("demo")),
                       key=lambda s: int(s.split("_")[1]))
        for demo in demos[int(n_core):]:
            g = f[f"data/{demo}"]
            if "obs" not in g or g[action_key].shape[0] < n_steps:
                continue                            # eval-only trajs (no obs)
            T = g[action_key].shape[0]
            fp = _fingerprint(g["states"][0])
            imgs = {v: g[f"obs/{v}"][:] for v in view_names}
            props = {k: g[f"obs/{k}"][:] for k in proprio_keys}
            acts = _default_aa_to_6d(g[action_key][:]).astype(np.float32)
            ts = list(range(0, T - n_steps + 1, n_steps * stride_chunks))
            mus = []
            for s in range(0, len(ts), batch):
                tb = ts[s:s + batch]
                obs_dict = {
                    **{v: torch.from_numpy(np.ascontiguousarray(imgs[v][tb]))
                       .permute(0, 3, 1, 2).float().unsqueeze(1)
                       for v in view_names},
                    **{k: torch.from_numpy(np.ascontiguousarray(props[k][tb]))
                       .float().unsqueeze(1) for k in proprio_keys},
                }
                obs_es = adapter(obs_dict)
                obs_es = {"visual": {v: x.to(device)
                                     for v, x in obs_es["visual"].items()},
                          "proprio": obs_es["proprio"].to(device)}
                a_flat = np.stack([acts[t:t + n_steps].reshape(-1) for t in tb])
                a_flat = torch.from_numpy(a_flat).float().to(device)
                with torch.no_grad():
                    s_bar = scout_vib.encode(obs_es)
                    mu, _ = vib_enc(s_bar, a_flat)
                mus.append(mu.detach().float().cpu().numpy())
            groups.setdefault(fp, []).append(np.concatenate(mus).mean(axis=0))
    return groups


def group_stats(mus: list) -> dict:
    X = np.stack(mus)                              # (n,16)
    n = len(X)
    d2 = [np.linalg.norm(X[i] - X[j])
          for i in range(n) for j in range(i + 1, n)]
    Xc = X - X.mean(axis=0, keepdims=True)
    S = Xc.T @ Xc / max(n - 1, 1)
    tr = float(np.trace(S))
    pr = float(tr ** 2 / max(float(np.sum(S * S)), 1e-12)) if tr > 0 else 0.0
    return {"n": n, "mean_pair_dist": float(np.mean(d2)) if d2 else 0.0,
            "pr": pr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--vib-ckpt", required=True)
    p.add_argument("--base-dp-ckpt", default=None,
                   help="override cfg.vib.base_dp_ckpt (E_s source)")
    p.add_argument("--n-core", type=int, default=20)
    p.add_argument("--run", action="append", required=True,
                   help="name=all.hdf5 path (repeatable)")
    p.add_argument("--cuda", default="5")
    args = p.parse_args()

    import os
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda)
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(args.config)
    if args.base_dp_ckpt:
        cfg.vib.base_dp_ckpt = args.base_dp_ckpt
    device = torch.device("cuda")
    from scout.eval.factories import make_scout_vib_factory
    from scout.eval.rollout import make_obs_adapter

    vib = make_scout_vib_factory(cfg, device)(args.vib_ckpt)
    adapter = make_obs_adapter(cfg.eval.view_names, cfg.eval.proprio_keys)
    view_names, proprio_keys = list(cfg.eval.view_names), list(cfg.eval.proprio_keys)

    for spec in args.run:
        name, path = spec.split("=", 1)
        groups = encode_run(path, vib, adapter, view_names, proprio_keys,
                            device, args.n_core)
        multi = {fp: g for fp, g in groups.items() if len(g) >= 2}
        stats = [group_stats(g) for g in multi.values()]
        if not stats:
            print(f"[dispersion] {name}: no multi-retry scene groups")
            continue
        ns = [s["n"] for s in stats]
        dists = [s["mean_pair_dist"] for s in stats]
        prs = [s["pr"] for s in stats]
        print(f"[dispersion] {name}: {len(groups)} scene groups "
              f"({len(multi)} with >=2 retries); retries/group median "
              f"{int(np.median(ns))}; mean pairwise mu_bar dist "
              f"{np.mean(dists):.4f} (median {np.median(dists):.4f}); "
              f"participation ratio mean {np.mean(prs):.2f} / median "
              f"{np.median(prs):.2f}  [1 = single-direction cone, "
              f"n = isotropic]")


if __name__ == "__main__":
    main()
