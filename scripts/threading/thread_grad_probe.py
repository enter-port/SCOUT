"""Offline KL-cost gradient telemetry for THREADING-MG-p1 (2026-09-17).

Why: the s233 grid (eta 0.1..2.0) showed a DOSE-INVARIANT response --
ATY rescued 16->16, jerk 0.1673->0.1686 (+0.8% over 20x), statistically
identical to the unguided DP arm (jerk 0.1677). Either (a) the VIB
posterior separates so strongly that KL(q_a||q_a0) > kappa=2.5 for every
candidate a, so the climb's cap mask (kl <= kappa) zeroes EVERY gradient
and eta multiplies nothing, or (b) the posterior is action-insensitive
and the raw gradient scale is just tiny. Both look identical in pass@5;
this probe separates them offline, no env needed.

Method (direct adaptation of scripts/analysis/diag_grad_decompose.py
part_b, 2026-08-21): load the s233 dyn-base VIB through the SAME factory
the rollout uses; draw one (obs, a-chunk) batch from the SAME core hdf5
via the SAME vib dataloader; anchor a0 = the demo chunk (= the unguided-
intent analog); perturb a = a0 + delta_rel * std(a) * eps for a ladder of
delta_rel; report the KL distribution vs kappa, the cap-hit fraction, and
the per-row |dKL/da| gradient norms (the exact quantity eta multiplies
inside guided_step/_climb_gradient, including the cap mask).

Run (server):
  cd /root/workspace/baojiachun/scout && CUDA_VISIBLE_DEVICES=0 \
    /root/workspace/baojiachun/.venv_mg/bin/python \
    scripts/threading/thread_grad_probe.py --seed-data-root \
    data/2026_9_16_threading_p1/THREADING-MG-p1-s233
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml
from easydict import EasyDict

CAP = 2.5          # atypical cap kappa in flight during the grid
CAPS_CTX = [5.0, 10.0]   # extra caps for context (if a raise is needed)
DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]  # frac of std(a)
N_EPS = 3          # perturbation draws per delta (seeded)


def newest_vib(d):
    return sorted(glob.glob(os.path.join(d, "*", "scout_vib.ckpt")),
                  key=os.path.getmtime)[-1]


def newest_ckpt(d):
    return sorted(glob.glob(os.path.join(d, "checkpoints", "*.ckpt")),
                  key=os.path.getmtime)[-1]


def kl_rows(mu, logvar, mu0, logvar0):
    """Exact copy of scout.guidance.entropy_costs._kl_rows math (both lists
    full here): KL(q(z|a) || q(z|a0)) per row."""
    var, var0 = torch.exp(logvar), torch.exp(logvar0)
    return 0.5 * (((mu - mu0) ** 2 / var0)
                  + (var / var0) - 1.0 - (logvar - logvar0)).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-data-root", required=True,
                    help="THREADING-MG-p1-s<seed> dir (uses its dyn-base + DP-base + core)")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sdr = args.seed_data_root
    task_dir = os.path.join(sdr, "threading")
    core = os.path.join(task_dir, "rollout", "threading_core.hdf5")
    vib_ckpt = newest_vib(os.path.join(task_dir, "train", "dyn", "dyn-base"))
    dp_ckpt = newest_ckpt(os.path.join(task_dir, "train", "DP", "DP-base"))
    print(f"[probe] core={core}\n[probe] vib={vib_ckpt}\n[probe] es={dp_ckpt}")

    dev = torch.device("cuda")

    # -- one batch through the SAME training-side pipeline (images -> E_s) -- #
    from scout.train_vib import make_dataloader, _slice_transition
    from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet

    vib_cfg = yaml.safe_load(open("configs/vib_threading_exp1.yaml"))
    vib_cfg["dataset"]["zarr_path"] = core
    vib_cfg["dataset"]["num_workers"] = 0
    vib_cfg["dataset"]["feature_cache"] = False
    vib_cfg["batch_size"] = args.batch_size
    torch.manual_seed(0)
    loader, _ds = make_dataloader(EasyDict(vib_cfg))
    batch = next(iter(loader))
    t_crop = get_eval_crop_transform_resnet(84, 76)
    obs_t, a_t, _, _ = _slice_transition(batch, dev, t_crop)
    print(f"[probe] batch B={a_t.shape[0]} a_dim={a_t.shape[1]}")

    # -- the model the rollout loads (factory path, incl. E_s guards) ------- #
    from scout.eval.factories import load_cfg, make_scout_vib_factory
    ecfg = load_cfg("configs/eval_threading_entropy.yaml")
    ecfg.vib.ckpt_path = vib_ckpt
    ecfg.vib.base_dp_ckpt = dp_ckpt
    model = make_scout_vib_factory(ecfg, dev)(vib_ckpt)
    model.eval()

    with torch.no_grad():
        s_bar = model.encode(obs_t)
        mu0, lv0 = model.vib_enc(s_bar, a_t)
        sig0 = torch.exp(0.5 * lv0)
        kl_prior = (0.5 * (mu0 ** 2 + torch.exp(lv0) - 1.0 - lv0)).sum(1).mean()
    print(f"[context] |mu0|={float(mu0.norm(dim=1).mean()):.4f} "
          f"sig0 mean/min/max={float(sig0.mean()):.4f}/{float(sig0.min()):.4f}/"
          f"{float(sig0.max()):.4f} KL(q0||N(0,I))={float(kl_prior):.3f}")

    a_std = a_t.std(dim=0)                       # per-dim action std (B,)-(a_dim,)
    a_std_mean = float(a_std.mean())
    print(f"[context] std(a) per-dim mean={a_std_mean:.5f}")

    # sanity: delta=0 -> KL=0, grad ~= 0
    a_g = a_t.clone().requires_grad_(True)
    mu, lv = model.vib_enc(s_bar.detach(), a_g)
    kl = kl_rows(mu, lv, mu0, lv0)
    g0 = torch.autograd.grad(kl.sum(), a_g)[0].flatten(1).norm(dim=1)
    print(f"[sanity] delta=0: KL mean={float(kl.mean()):.2e} "
          f"max={float(kl.max()):.2e} row|g| mean={float(g0.mean()):.2e}")

    results = {"context": {
        "B": int(a_t.shape[0]), "a_dim": int(a_t.shape[1]),
        "mu0_norm": float(mu0.norm(dim=1).mean()),
        "sig0_mean": float(sig0.mean()),
        "kl_prior": float(kl_prior), "std_a_mean": a_std_mean,
        "sanity_kl_mean": float(kl.mean()), "sanity_rowgrad_mean": float(g0.mean()),
    }, "ladder": []}

    hdr = ("%8s %8s %8s %8s %8s %9s %9s %9s %9s %9s" %
           ("delta", "KLmean", "KLq50", "KLq90", "cap_hit", "|g|all", "|g|live",
           "n_live", "eta*.1sd", "eta*.5sd"))
    print(hdr)
    for d_rel in DELTAS:
        kl_all, g_all, g_live, cap_hits, rows_n = [], [], [], 0, 0
        for zi in range(N_EPS):
            gen = torch.Generator(device="cpu").manual_seed(1000 + zi)
            eps = torch.randn(a_t.shape, generator=gen).to(dev)
            a_p = (a_t + d_rel * a_std.unsqueeze(0) * eps)
            a_g = a_p.clone().requires_grad_(True)
            mu, lv = model.vib_enc(s_bar.detach(), a_g)
            kl = kl_rows(mu, lv, mu0, lv0)
            g = torch.autograd.grad(kl.sum(), a_g)[0].flatten(1).norm(dim=1)
            kld = kl.detach()
            live = (kld <= CAP).cpu()
            kl_all.append(kld.cpu()); g_all.append(g.detach().cpu())
            g_live.append(g.detach().cpu()[live]); cap_hits += int((~live).sum())
            rows_n += int(kld.shape[0])
        klc = torch.cat(kl_all).numpy()
        gac = torch.cat(g_all).numpy()
        gl = torch.cat(g_live).numpy() if len(g_live) else np.array([0.0])
        gl_mean = float(gl.mean()) if len(gl) else 0.0
        # eta that would inject ~0.1x / 0.5x of std(a) per guided step at
        # sqrt(1-abar)~0.5 (mid guided window, DDPM100 gst50):
        # disp = eta * sqrt(1-abar) * g  =>  eta = target / (0.5 * g)
        eta01 = 0.1 * a_std_mean / max(gl_mean, 1e-12) / 0.5
        eta05 = 0.5 * a_std_mean / max(gl_mean, 1e-12) / 0.5
        print(("%8g %8.4f %8.4f %8.4f %8s %9.3e %9.3e %9d %9.3g %9.3g" %
               (d_rel, float(klc.mean()), float(np.median(klc)),
                float(np.quantile(klc, 0.9)),
                "%.0f%%" % (100.0 * cap_hits / rows_n),
                float(gac.mean()), gl_mean, int(gl.shape[0]), eta01, eta05)))
        results["ladder"].append({
            "delta_rel": d_rel, "kl_mean": float(klc.mean()),
            "kl_q50": float(np.median(klc)),
            "kl_q90": float(np.quantile(klc, 0.9)),
            "cap_hit_frac": cap_hits / rows_n,
            "cap5_hit_frac": float((klc > 5.0).mean()),
            "cap10_hit_frac": float((klc > 10.0).mean()),
            "rowgrad_all_mean": float(gac.mean()),
            "rowgrad_live_mean": gl_mean,
            "n_live": int(gl.shape[0]),
            "eta_for_0.1sd": eta01, "eta_for_0.5sd": eta05,
        })

    # far-field context: another demo's chunk as candidate (crossed pairs)
    perm = torch.randperm(a_t.shape[0], generator=torch.Generator().manual_seed(7)).to(dev)
    a_g = a_t[perm].clone().requires_grad_(True)
    mu, lv = model.vib_enc(s_bar.detach(), a_g)
    kl = kl_rows(mu, lv, mu0, lv0)
    g = torch.autograd.grad(kl.sum(), a_g)[0].flatten(1).norm(dim=1)
    kld = kl.detach()
    print(("[far-field] crossed-demo chunks: KL mean/q50=%.3f/%.3f cap_hit=%.0f%% "
           "row|g| live-mean=%.3e" %
           (float(kld.mean()), float(kld.median()),
            100.0 * float((kld > CAP).float().mean()),
            float(g[kld <= CAP].mean()) if int((kld <= CAP).sum()) else 0.0)))
    results["far_field"] = {
        "kl_mean": float(kld.mean()), "kl_q50": float(kld.median()),
        "cap_hit_frac": float((kld > CAP).float().mean()),
    }

    out = args.out or os.path.join(os.path.dirname(sdr.rstrip("/")),
                                   "TELEMETRY", "grad_probe_%s.json" %
                                   os.path.basename(sdr.rstrip("/")))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[probe] saved -> {out}")


if __name__ == "__main__":
    main()
