"""Per-round KL-cost gradient telemetry for the ATY-decline investigation.

COPY of scripts/coffee/cfk_grad_probe.py (grid S1) with ONLY the ckpt
selection parametrized: --vib-ckpt / --dp-ckpt are explicit, so any
(DP-exp{k}, dyn-exp{k}) field pair can be probed on the SAME frozen core
probe batch (torch.manual_seed(0), num_workers=0 -> identical batch across
runs). All ladder math, CAP, DELTAS, N_EPS, eta formulas are UNTOUCHED so
numbers are directly comparable with grad_probe_COFFEE-MG-p1-s233.json
(the dyn-base anchor behind eta*=5.6).

Run (server), one invocation per ckpt pair, each well under 5 min:
  cd /root/workspace/baojiachun/scout && CUDA_VISIBLE_DEVICES=0 \
    /root/workspace/baojiachun/.venv_mg/bin/python \
    scripts/coffee/cfk_grad_probe_arm.py --seed-data-root \
    data/2026_9_20_coffee_mg_p1/COFFEE-MG-p1-s233 \
    --vib-ckpt <.../dyn-ATY-exp1/<ts>/scout_vib.ckpt> \
    --dp-ckpt  <.../DP/DP-ATY-exp1/checkpoints/299.ckpt> --tag aty1_pair
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")

import numpy as np
import torch
import yaml
from easydict import EasyDict

CAP = 2.5
DELTAS = [0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
N_EPS = 3


def kl_rows(mu, logvar, mu0, logvar0):
    var, var0 = torch.exp(logvar), torch.exp(logvar0)
    return 0.5 * (((mu - mu0) ** 2 / var0)
                  + (var / var0) - 1.0 - (logvar - logvar0)).sum(dim=-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-data-root", required=True)
    ap.add_argument("--vib-config", default="configs/vib_coffee_exp1.yaml")
    ap.add_argument("--eval-config", default="configs/eval_coffee_entropy.yaml")
    ap.add_argument("--task", default="coffee")
    ap.add_argument("--core-name", default="coffee_core.hdf5")
    ap.add_argument("--vib-ckpt", required=True)
    ap.add_argument("--dp-ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--data-hdf5", default=None,
                    help="override probe batch source (e.g. an all_accum.hdf5)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    sdr = args.seed_data_root
    task_dir = os.path.join(sdr, args.task)
    core = (args.data_hdf5 or os.path.join(task_dir, "rollout", args.core_name))
    vib_ckpt, dp_ckpt = args.vib_ckpt, args.dp_ckpt
    for p in (core, vib_ckpt, dp_ckpt):
        assert os.path.exists(p), f"missing {p}"
    print(f"[probe] core={core}\n[probe] vib={vib_ckpt}\n[probe] es={dp_ckpt}")

    dev = torch.device("cuda")

    from scout.train_vib import make_dataloader, _slice_transition
    from dyn_model.datasets.img_transforms import get_eval_crop_transform_resnet

    vib_cfg = yaml.safe_load(open(args.vib_config))
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

    from scout.eval.factories import load_cfg, make_scout_vib_factory
    ecfg = load_cfg(args.eval_config)
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

    a_std = a_t.std(dim=0)
    a_std_mean = float(a_std.mean())
    print(f"[context] std(a) per-dim mean={a_std_mean:.5f}")

    a_g = a_t.clone().requires_grad_(True)
    mu, lv = model.vib_enc(s_bar.detach(), a_g)
    kl = kl_rows(mu, lv, mu0, lv0)
    g0 = torch.autograd.grad(kl.sum(), a_g)[0].flatten(1).norm(dim=1)
    print(f"[sanity] delta=0: KL mean={float(kl.mean()):.2e} "
          f"max={float(kl.max()):.2e} row|g| mean={float(g0.mean()):.2e}")

    results = {"tag": args.tag, "vib_ckpt": vib_ckpt, "dp_ckpt": dp_ckpt,
               "context": {
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
                                   "TELEMETRY", "grad_probe_arm_%s.json" % args.tag)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"[probe] saved -> {out}")


if __name__ == "__main__":
    main()
