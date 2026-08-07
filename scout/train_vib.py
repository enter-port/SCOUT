"""E1: VIB joint training + β scan + life/death sensitivity
(scout_impl_plan.md Task 3.4, scout_design.md §3/§5).

For each β in ``cfg.betas`` train a fresh :class:`ScoutVIB` on transitions from
a :class:`TransitionSource` (default :class:`RobomimicLowdimSource`), logging
AE / dyn / KL and μ mean/std. After training, compute
:func:`sensitivity_ratio` and save ckpt + loss PNG. Finally plot sensitivity
vs β to pick the largest β whose sensitivity ≥ ~0.3 (design §5; the
make-or-break knob per §7 risk #1).

Usage:
    python -m scout.train_vib --config configs/vib_lift_lowdim.yaml

Sampling is ``source.sample(batch)`` (no DataLoader): each "epoch" is
``steps_per_epoch`` random batches -- this matches the online-buffer contract
of :class:`TransitionSource` (the self-improvement loop reuses the same path).
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from easydict import EasyDict

from scout.data.transition_source import TransitionSource
from scout.diagnose import sensitivity_ratio
from scout.model.scout_vib import ScoutVIB


# --------------------------------------------------------------------------- #
# utils
# --------------------------------------------------------------------------- #
def to_plain(obj):
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def set_seed(seed):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def plot_curves(history, keys, title, path):
    fig, ax = plt.subplots(figsize=(6, 4))
    for k in keys:
        if history.get(k):
            ax.plot(history[k], label=k, linewidth=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("value"); ax.set_title(title); ax.legend()
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------- #
# per-β training
# --------------------------------------------------------------------------- #
def mu_stats(model, source, batch_size, device):
    """Sample a batch, return (mu.mean over dims, mu.std over batch mean, sigma_mu)."""
    b = source.sample(min(batch_size, len(source)))
    S_t = b["S_t"].to(device); A_t = b["A_t"].to(device)
    with torch.no_grad():
        mu, _ = model.vib_enc(model.ae.encode(S_t), A_t)
    mu_mean = float(mu.mean())                  # grand mean (≈0 if KL working)
    sigma_mu = float(mu.std(dim=0).mean())      # mean per-dim std -> μ scale
    mu_abs_mean = float(mu.abs().mean())
    return S_t, A_t, mu_mean, mu_abs_mean, sigma_mu


def train_one_beta(cfg, source, state_dim, action_dim, beta, sigma_a, device, beta_dir):
    model = ScoutVIB(
        state_dim, action_dim,
        modality=cfg.model.modality,
        s_latent_dim=cfg.model.s_latent_dim,
        style_dim=cfg.model.style_dim,
        hidden_dim=cfg.model.hidden_dim,
        beta=beta,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), **cfg.optimizer.params)

    steps_per_epoch = int(cfg.steps_per_epoch)
    num_epochs = int(cfg.num_epochs)
    log_every_batch = max(1, steps_per_epoch // 5)
    history = {"ae": [], "dyn": [], "kl": [], "mu_abs": []}

    model.train()
    for epoch in range(num_epochs):
        ep = {"ae": 0.0, "dyn": 0.0, "kl": 0.0, "mu_abs": 0.0, "n": 0}
        for it in range(steps_per_epoch):
            b = source.sample(cfg.batch_size)
            S_t = b["S_t"].to(device); A_t = b["A_t"].to(device); S_tp1 = b["S_tp1"].to(device)
            out = model(S_t, A_t, S_tp1)
            opt.zero_grad(); out["loss"].backward(); opt.step()

            if (it + 1) % log_every_batch == 0 or it == steps_per_epoch - 1:
                ep["ae"] += out["ae"].item(); ep["dyn"] += out["dyn"].item()
                ep["kl"] += out["kl"].item()
                ep["mu_abs"] += out["mu"].detach().abs().mean().item()
                ep["n"] += 1
        n = max(1, ep["n"])
        history["ae"].append(ep["ae"] / n); history["dyn"].append(ep["dyn"] / n)
        history["kl"].append(ep["kl"] / n); history["mu_abs"].append(ep["mu_abs"] / n)
        print(f"  [β={beta:g}] epoch {epoch:4d} | ae {history['ae'][-1]:.4f} "
              f"dyn {history['dyn'][-1]:.4f} kl {history['kl'][-1]:.4f} "
              f"|μ| {history['mu_abs'][-1]:.4f}")

    # post-train diagnostics
    S_t, A_t, mu_mean, mu_abs_mean, sigma_mu = mu_stats(model, source, cfg.batch_size, device)
    sr = sensitivity_ratio(model, S_t, A_t, sigma_a, sigma_mu)

    torch.save({"state_dict": model.state_dict(), "beta": beta,
                "sensitivity": sr, "sigma_mu": sigma_mu, "sigma_a": sigma_a},
               os.path.join(beta_dir, "scout_vib.ckpt"))
    plot_curves(history, ["ae", "dyn", "kl"], f"VIB losses (β={beta:g})",
                os.path.join(beta_dir, "losses.png"))
    plot_curves(history, ["mu_abs"], f"|μ| (β={beta:g})",
                os.path.join(beta_dir, "mu.png"))

    return {"beta": beta,
            "ae": history["ae"][-1], "dyn": history["dyn"][-1], "kl": history["kl"][-1],
            "mu_abs": mu_abs_mean, "sigma_mu": sigma_mu, "sensitivity": sr}


def run(cfg, source=None):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    if source is None:
        from scout.data.robomimic_lowdim import RobomimicLowdimSource   # lazy: keeps dummy path hermetic
        source = RobomimicLowdimSource(cfg.dataset.path, mask_key=cfg.dataset.train_filter_key)
    state_dim = source.state_dim
    action_dim = source.action_dim
    print(f"source: len={len(source)} state_dim={state_dim} action_dim={action_dim}")

    sigma_a = float(source.stats()["A_t"].std.mean())
    print(f"sigma_a (mean per-dim action std) = {sigma_a:.4f}")

    run_root = os.path.join(cfg.save_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs(run_root, exist_ok=True)
    with open(os.path.join(run_root, "config.yaml"), "w") as f:
        yaml.safe_dump(to_plain(cfg), f, default_flow_style=False)

    summaries = []
    for beta in cfg.betas:
        beta_dir = os.path.join(run_root, f"beta_{beta:g}")
        os.makedirs(beta_dir, exist_ok=True)
        print(f"\n=== training β={beta:g} ===")
        s = train_one_beta(cfg, source, state_dim, action_dim, beta, sigma_a, device, beta_dir)
        summaries.append(s)
        print(f"=== β={beta:g} done | sensitivity={s['sensitivity']:.4f} "
              f"sigma_mu={s['sigma_mu']:.4f} (sigma_a/sigma_mu={sigma_a/s['sigma_mu']:.4f}) ===")

    # sensitivity vs beta
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([s["beta"] for s in summaries],
            [s["sensitivity"] for s in summaries], "o-", linewidth=1.2)
    ax.axhline(0.3, color="r", linestyle="--", linewidth=0.8, label="~0.3 threshold")
    ax.set_xscale("log"); ax.set_xlabel("β"); ax.set_ylabel("sensitivity ratio")
    ax.set_title("VIB life/death: sensitivity vs β"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(run_root, "sensitivity_vs_beta.png"), dpi=120)
    plt.close(fig)

    with open(os.path.join(run_root, "summary.yaml"), "w") as f:
        yaml.safe_dump(to_plain(summaries), f, default_flow_style=False)
    print(f"\nsummaries:\n{yaml.safe_dump(to_plain(summaries), default_flow_style=False)}")
    print(f"run_root: {run_root}")
    return run_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    print(dict(cfg))
    run(cfg)
