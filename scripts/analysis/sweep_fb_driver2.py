#!/usr/bin/env python
"""In-process fb/beta/lambda sweep on the collapsed setting (can r1 all_accum).

The cold start (31GB feature bank + zarr + E_s) costs ~5 min per PROCESS,
which blew the per-experiment 5-min cap. So: load the dataset/bank ONCE,
then run each parameter combo inside the same process (fresh seed + fresh
model init per combo, ~2-3 min of training each -> every EXPERIMENT stays
under 5 min). Structure untouched: only beta / free_bits / failure_weight /
lr / weight_decay vary.

Usage: python driver2.py <gpu>   (from repo root)
"""
import contextlib
import copy
import io
import os
import re
import sys
import time

import numpy as np
import torch
import yaml

REPO = "/root/workspace/baojiachun/scout"
os.chdir(REPO)
sys.path.insert(0, REPO)

from easydict import EasyDict                                   # noqa: E402
from torch.utils.data import DataLoader, Subset                 # noqa: E402
from scout.train_vib import (build_E_s, make_dataloader,        # noqa: E402
                             set_seed, train_one_beta)
from scout.feat_cache import (CachedFeatureTransitionDataset,   # noqa: E402
                              get_or_build_bank)

BASE_CFG = f"{REPO}/data/experiment2/can/train/dyn/dyn-SCOUT-exp1/config.yaml"
OUT = "/root/workspace/baojiachun/diag_vib/sweep_fb"
GPU = sys.argv[1] if len(sys.argv) > 1 else "4"
EP = 70          # collapse was visible by ep ~42 at 300ep-cosine; 70 is enough

RUNS = [
    # name, free_bits, beta, lam, lr, wd
    ("base_fb0005_b1e-4_l5", 0.005, 1.0e-4, 5, 1e-3, 1e-6),
    ("fb002",               0.02,  1.0e-4, 5, 1e-3, 1e-6),
    ("fb005",               0.05,  1.0e-4, 5, 1e-3, 1e-6),
    ("fb010",               0.10,  1.0e-4, 5, 1e-3, 1e-6),
    ("b3e-5",               0.005, 3.0e-5, 5, 1e-3, 1e-6),
    ("b1e-5",               0.005, 1.0e-5, 5, 1e-3, 1e-6),
    ("lam1",                0.005, 1.0e-4, 1, 1e-3, 1e-6),
    ("fb002_b3e-5",         0.02,  3.0e-5, 5, 1e-3, 1e-6),
    ("wd0",                 0.005, 1.0e-4, 5, 1e-3, 0.0),
    ("lr3e-4",              0.005, 1.0e-4, 5, 3e-4, 1e-6),
]


class _WeightedView(torch.utils.data.Dataset):
    def __init__(self, base, w):
        self.base, self.w = base, w

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        return (*self.base[i], self.w[i])


def main():
    cfg = EasyDict(yaml.safe_load(open(BASE_CFG)))
    device = torch.device("cuda")
    set_seed(cfg.seed)

    t0 = time.time()
    loader, ds = make_dataloader(cfg)               # hdf5 -> zarr (cached zip)
    E_s = build_E_s(cfg)
    banks = get_or_build_bank(ds, E_s, cfg, device) # cache hit (~31GB load)
    ds.imgs = {}
    train_ds = CachedFeatureTransitionDataset(ds, banks, train=True)
    val_ds = CachedFeatureTransitionDataset(ds, banks, train=False)
    print(f"[sweep] data ready in {time.time()-t0:.0f}s "
          f"({len(train_ds)} anchors)", flush=True)

    # episode-level split (same as run())
    n_eps = len(train_ds.episode_start_indices)
    n_val = max(1, int(round(n_eps * float(cfg.val_ratio))))
    cutoff = int(train_ds.episode_start_indices[n_eps - n_val])
    anchors = train_ds.valid_anchor_indices
    train_idx = np.where(anchors < cutoff)[0]
    val_idx = np.where(anchors >= cutoff)[0]
    val_loader = (DataLoader(Subset(val_ds, val_idx), batch_size=int(cfg.batch_size),
                             shuffle=False, num_workers=0)
                  if len(val_idx) > 0 else None)

    # per-anchor weights for lam=5 (failures upweighted) + unweighted view
    import h5py
    def _dn(k):
        m = re.search(r"(\d+)$", k)
        return int(m.group(1)) if m else 0
    with h5py.File(cfg.dataset.zarr_path, "r") as f:
        demos = sorted([k for k in f["data"].keys() if k.startswith("demo")], key=_dn)
        succ = [bool(np.asarray(f["data"][k]["success"]).ravel()[0])
                if "success" in f["data"][k] else True for k in demos]
    ep_w5 = np.array([5.0 if not s else 1.0 for s in succ], dtype=np.float32)
    a_ep = np.searchsorted(ds.episode_ends, train_ds.valid_anchor_indices, side="right")
    view5 = _WeightedView(train_ds, ep_w5[a_ep])
    print(f"[sweep] {int((~np.array(succ)).sum())}/{len(succ)} failure episodes", flush=True)

    for name, fb, beta, lam, lr, wd in RUNS:
        set_seed(cfg.seed)                          # identical init across combos
        view = view5 if lam == 5 else train_ds
        train_loader = DataLoader(Subset(view, train_idx), batch_size=int(cfg.batch_size),
                                  shuffle=True, num_workers=0, drop_last=True)
        rcfg = EasyDict(copy.deepcopy(dict(cfg)))
        rcfg.free_bits = fb
        rcfg.num_epochs = EP
        rcfg.steps_per_epoch = 100
        rcfg.optimizer.params.lr = lr
        rcfg.optimizer.params.weight_decay = wd
        # screen under CONSTANT peak lr: the 300ep cosine keeps lr>=0.8*peak
        # for the first ~60 epochs (where the production collapse happened);
        # a 70ep cosine decays too fast and froze the drift (baseline at 70ep
        # stayed alive=0.19 -- a false negative). Constant-peak is harsher
        # than production's first 60 epochs -> survival here implies
        # survival under the real schedule.
        rcfg.optimizer.lr_scheduler = "none"
        out_dir = f"{OUT}/{name}"
        os.makedirs(out_dir, exist_ok=True)
        buf = io.StringIO()
        t1 = time.time()
        try:
            with contextlib.redirect_stdout(buf):
                s = train_one_beta(rcfg, train_loader, val_loader, ds, cfg.model.E_s,
                                   ds.action_dim, beta, device, out_dir,
                                   train_t=None, val_t=None, val_every=10_000,
                                   wandb_run=None, E_s=E_s, feats_mode=True)
            err = ""
        except Exception as e:                      # noqa: BLE001
            s, err = None, repr(e)[:120]
        dt = time.time() - t1
        log = buf.getvalue()
        with open(f"{OUT}/{name}.log", "w") as f:
            f.write(log)
        alive = re.findall(r"relu_alive ([0-9.]+)", log)
        done = re.search(r"=== done \| .*latent_mse=([0-9.e+-]+) kl=([0-9.e+-]+) "
                         r"\|μ\|=([0-9.e+-]+)", log)
        g = re.search(r"\[guidance-check\] \|dNLL/da\| on a real batch = ([0-9.e+-]+)", log)
        row = "\t".join([
            name, f"fb={fb}", f"b={beta:g}", f"lam={lam}", f"lr={lr:g}", f"wd={wd:g}",
            f"{dt:.0f}s",
            f"alive0={alive[1] if len(alive) > 1 else '?'}",
            f"aliveL={alive[-1] if alive else '?'}",
            f"kl={done.group(2) if done else '?'}",
            f"mu={done.group(3) if done else '?'}",
            f"mse={done.group(1) if done else '?'}",
            f"grad={g.group(1) if g else '?'}",
            err])
        print("[sweep] " + row, flush=True)
        with open(f"{OUT}/summary2.tsv", "a") as f:
            f.write(row + "\n")
    print("[sweep] ALL DONE", flush=True)


if __name__ == "__main__":
    main()
