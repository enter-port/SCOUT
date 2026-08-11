"""Step 1: VIB dynamics joint training, single β (stage1_plan.md Step 1,
scout_design.md §3).

Trains ONE :class:`ScoutVIB` (β = ``cfg.beta``, default 1e-3) on transitions
drawn from LPB's :class:`RobomimicImageDynamicsModelDataset` (image + proprio),
logging latent MSE / KL / |μ| and saving ckpt + loss PNGs. **No β scan, no
life/death diagnostic** -- per the revised stage-1 plan (single β, run straight
into Step 2). If Step 4 (base vs explore) is NO-GO, the first suspect is
β too large -> guidance no-op; re-run :func:`scout.diagnose.sensitivity_ratio`
then (it lives in its own module, not here).

Loss = latent MSE + β·KL (latent-level target = ``E_s(S_{t+1}).detach()``;
no AE / no reconstruction / no state decoder; scout_design.md §3). The frozen
base-DP ResNet inside E_s has ``requires_grad=False`` -- single-chain, single
``backward``, no gradient isolation. Only ``vib_enc`` / ``D_s`` /
``proprio_embed`` update.

Usage (real run, needs pytorch3d/robomimic + a base-DP ckpt -- training env):
    python -m scout.train_vib --config configs/vib_lift_image.yaml

For an environment-agnostic forward/backward smoke test (no dataset, no ckpt):
    python -m scout.train_vib --dummy
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict

try:
    import wandb
except ImportError:
    wandb = None

from scout.diagnose import sensitivity_ratio
from scout.model.encoder import StateEncoder
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
# E_s construction
# --------------------------------------------------------------------------- #
def build_E_s(cfg) -> StateEncoder:
    """Build the LPB-style E_s (frozen base-DP ResNet + trained proprio Conv1d).

    Uses :meth:`StateEncoder.from_base_dp_ckpt` -- lazy-imports
    ``dyn_model.models.resnet_encoder`` so this trainer imports cleanly in
    environments without the LPB stack.
    """
    d = cfg.model.E_s
    return StateEncoder.from_base_dp_ckpt(
        base_dp_ckpt=d.base_dp_ckpt,
        view_names=list(d.view_names),
        proprio_dim=int(d.proprio_dim),
        proprio_emb_dim=int(getattr(d, "proprio_emb_dim", 64)),
    )


# --------------------------------------------------------------------------- #
# LPB dataset -> (obs_t, a_t, obs_tp1) windows
# --------------------------------------------------------------------------- #
def make_dataloader(cfg):
    """Build a DataLoader over LPB ``RobomimicImageDynamicsModelDataset``.

    Lazy-imported (pytorch3d/robomimic not available on Windows). Returns
    ``(loader, dataset)``. Uses ``num_hist=num_pred=1, frameskip=1`` so each item
    is a clean 2-frame window ``(obs, act, state)`` we slice into a transition.
    """
    from torch.utils.data import DataLoader

    from dyn_model.datasets.robomimic_dset import RobomimicImageDynamicsModelDataset

    d = cfg.dataset
    ds = RobomimicImageDynamicsModelDataset(
        zarr_path=d.zarr_path,
        num_hist=1,
        num_pred=1,
        frameskip=int(getattr(d, "frameskip", 1)),
        view_names=list(d.view_names),
        abs_action=bool(getattr(d, "abs_action", False)),
        use_crop=bool(getattr(d, "use_crop", False)),
        train=True,
        shape_obs=dict(d.shape_obs),
        original_img_size=int(getattr(d, "original_img_size", 140)),
        cropped_img_size=int(getattr(d, "cropped_img_size", 128)),
        action_dim=int(d.action_dim),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.batch_size),
        shuffle=True,
        num_workers=int(getattr(d, "num_workers", 4)),
        drop_last=True,
    )
    return loader, ds


def _slice_transition(batch, device, img_transform=None):
    """LPD batch ``(obs, act, state)`` -> ``(obs_t, a_t, obs_tp1)`` on `device`.

    Batch shapes (num_hist=num_pred=1, frameskip=fs):
      obs['visual'][v]: (B, 2, 3, H, W)  obs['proprio']: (B, 2, P)   # frames t, t+fs
      act:              (B, 2*fs, step_dim)                          # per-step actions
    Slices frame 0 -> t, frame 1 -> t+fs. The action fed to VIB_enc is the first
    ``fs`` per-step actions flattened (B, fs*step_dim) -- the chunk that moves
    s_t -> s_{t+fs}. frameskip=1 -> single action; frameskip=8 -> 80-dim chunk,
    matching the base DP's n_action_steps=8 chunk so train/test mu(s,a) agree.
    """
    obs, act, _state = batch
    visual = obs["visual"]
    proprio = obs["proprio"]
    if img_transform is not None:
        # per-sample crop: same random offset across the window's frames
        # (mirrors LPB dyn_model/train.py:438 and base DP CropRandomizer). Apply
        # BEFORE slicing t/t+1 so both frames of a sample share the crop offset.
        visual = {v: torch.stack([img_transform(im) for im in vimg])
                  for v, vimg in visual.items()}
    obs_t = {
        "visual": {v: img[:, 0:1].to(device) for v, img in visual.items()},
        "proprio": proprio[:, 0:1].to(device),
    }
    obs_tp1 = {
        "visual": {v: img[:, 1:2].to(device) for v, img in visual.items()},
        "proprio": proprio[:, 1:2].to(device),
    }
    # action chunk for the transition: first `fs` per-step actions flattened
    # (fs = frameskip; act is (B, 2*fs, step_dim) since num_hist=num_pred=1).
    fs = act.shape[1] // 2
    a_t = act[:, :fs].reshape(act.shape[0], -1).to(device)
    return obs_t, a_t, obs_tp1


# --------------------------------------------------------------------------- #
# VIB training (single β; no diagnostic -- stage1_plan.md Step 1)
# --------------------------------------------------------------------------- #
def train_one_beta(cfg, loader, ds, E_s_cfg, action_dim, beta,
                   device, out_dir, img_transform=None, wandb_run=None):
    E_s = build_E_s(E_s_cfg)
    model = ScoutVIB(
        action_dim=action_dim,
        E_s=E_s,
        style_dim=cfg.model.style_dim,
        hidden_dim=cfg.model.hidden_dim,
        beta=beta,
    ).to(device)
    # only trainable params: vib_enc / D_s / proprio_embed (ResNet frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, **cfg.optimizer.params)

    steps_per_epoch = int(cfg.steps_per_epoch)
    num_epochs = int(cfg.num_epochs)
    log_every_batch = max(1, steps_per_epoch // 5)
    history = {"latent_mse": [], "kl": [], "mu_abs": []}

    model.train()
    for epoch in range(num_epochs):
        ep = {"latent_mse": 0.0, "kl": 0.0, "mu_abs": 0.0, "n": 0}
        for it, batch in enumerate(loader):
            obs_t, A_t, obs_tp1 = _slice_transition(batch, device, img_transform)
            out = model(obs_t, A_t, obs_tp1)
            opt.zero_grad(); out["loss"].backward(); opt.step()

            if (it + 1) % log_every_batch == 0 or it == steps_per_epoch - 1:
                ep["latent_mse"] += out["latent_mse"].item()
                ep["kl"] += out["kl"].item()
                ep["mu_abs"] += out["mu"].detach().abs().mean().item()
                ep["n"] += 1
            if it + 1 >= steps_per_epoch:
                break
        n = max(1, ep["n"])
        history["latent_mse"].append(ep["latent_mse"] / n)
        history["kl"].append(ep["kl"] / n)
        history["mu_abs"].append(ep["mu_abs"] / n)
        print(f"  [β={beta:g}] epoch {epoch:4d} | latent_mse {history['latent_mse'][-1]:.4f} "
              f"kl {history['kl'][-1]:.4f} |μ| {history['mu_abs'][-1]:.4f}")
        if wandb_run is not None:
            wandb_run.log({"latent_mse": history["latent_mse"][-1],
                           "kl": history["kl"][-1],
                           "mu_abs": history["mu_abs"][-1]}, step=epoch)

    # save ckpt (single-β flow; no diagnostic -- stage1_plan.md Step 1)
    torch.save({"state_dict": model.state_dict(), "beta": beta},
               os.path.join(out_dir, "scout_vib.ckpt"))
    plot_curves(history, ["latent_mse", "kl"], f"VIB losses (β={beta:g})",
                os.path.join(out_dir, "losses.png"))
    plot_curves(history, ["mu_abs"], f"|μ| (β={beta:g})",
                os.path.join(out_dir, "mu.png"))

    return {"beta": beta,
            "latent_mse": history["latent_mse"][-1],
            "kl": history["kl"][-1],
            "mu_abs": history["mu_abs"][-1]}


def run(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    loader, ds = make_dataloader(cfg)
    action_dim = ds.action_dim
    print(f"dataset: len={len(ds)} action_dim={action_dim} proprio_dim={ds.proprio_dim} "
          f"views={ds.view_names}")
    beta = float(cfg.get("beta", 1.0e-3))
    print(f"beta = {beta:g}  (single-β flow; no scan, no diagnostic)")

    # per-sample image transform (matches base DP crop). LPB applies it in the
    # train loop (dyn_model/train.py:438), not the dataset, so SCOUT wires it
    # here. None => feed full image (ResNet adaptive-pool is size-agnostic).
    from dyn_model.datasets.img_transforms import get_train_crop_transform_resnet
    if bool(getattr(cfg.dataset, "use_crop", False)):
        img_transform = get_train_crop_transform_resnet(
            int(cfg.dataset.original_img_size), int(cfg.dataset.cropped_img_size))
        print(f"img_transform: RandomCrop({cfg.dataset.cropped_img_size}) "
              f"on {cfg.dataset.original_img_size} (per-sample, matches base DP)")
    else:
        img_transform = None
        print("img_transform: none (full image)")

    run_root = os.path.join(cfg.save_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    os.makedirs(run_root, exist_ok=True)
    with open(os.path.join(run_root, "config.yaml"), "w") as f:
        yaml.safe_dump(to_plain(cfg), f, default_flow_style=False)

    # wandb (optional). Key isolation: the launch sources baojiachun/.secrets/wandb.env
    # (WANDB_API_KEY / WANDB_CONFIG_DIR / WANDB_CACHE_DIR); nothing touches /root/.netrc.
    wandb_run = None
    if wandb is not None and bool(getattr(cfg, "use_wandb", True)):
        wcfg = cfg.get("wandb", {}) or {}
        wandb_run = wandb.init(
            project=wcfg.get("project", "scout-dynamics"),
            name=wcfg.get("name"), config=to_plain(cfg),
            dir=cfg.save_dir, tags=list(wcfg.get("tags", ["step1", "vib"])))
        print(f"wandb: project={wcfg.get('project', 'scout-dynamics')} name={wcfg.get('name')}")
    else:
        print("wandb: disabled")

    print(f"\n=== training β={beta:g} ===")
    s = train_one_beta(cfg, loader, ds, cfg, action_dim, beta, device, run_root,
                       img_transform=img_transform, wandb_run=wandb_run)
    print(f"=== done | β={beta:g} | latent_mse={s['latent_mse']:.4f} "
          f"kl={s['kl']:.4f} |μ|={s['mu_abs']:.4f} ===")

    with open(os.path.join(run_root, "summary.yaml"), "w") as f:
        yaml.safe_dump(to_plain(s), f, default_flow_style=False)
    if wandb_run is not None:
        wandb_run.log({"final/latent_mse": s["latent_mse"], "final/kl": s["kl"],
                       "final/mu_abs": s["mu_abs"]})
        wandb_run.finish()
    print(f"run_root: {run_root}")
    return run_root


# --------------------------------------------------------------------------- #
# dummy verification path (no dataset, no ckpt; hermetic)
# --------------------------------------------------------------------------- #
class _MockResNetEncoder(nn.Module):
    """Tiny stand-in for LPB :class:`ResNetEncoder` -- same duck-typed interface
    (``emb_dim=512``, ``forward({view: (B,T,3,H,W)}) -> {view: (B,T,1,512)}``)
    so :class:`StateEncoder` can be exercised without a base-DP ckpt. Parameters
    exist so the freeze check is meaningful.
    """

    def __init__(self, view_names):
        super().__init__()
        self.view_names = list(view_names)
        self.emb_dim = 512
        self.proj = nn.Conv2d(3, 512, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        from einops import rearrange
        out = {}
        for v in self.view_names:
            imgs = x[v]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t ... -> (b t) ...")
            feat = self.avgpool(self.proj(imgs))     # (B*T, 512, 1, 1)
            feat = feat.flatten(1).unsqueeze(1)      # (B*T, 1, 512)
            out[v] = rearrange(feat, "(b t) p d -> b t p d", b=b)
        return out


def dummy_run(view_names=("agentview", "robot0_eye_in_hand"),
              proprio_dim=10, action_dim=10, batch_size=8, seed=233,
              beta=1.0e-3):
    """Forward + backward smoke test with random {image, proprio} windows.

    Verifies:
      - StateEncoder fuses ResNet (frozen) + proprio -> s̄ of expected dim;
      - ScoutVIB.forward returns a loss dict (latent_mse + β·KL);
      - backward only touches vib_enc / D_s / proprio_embed (ResNet grads None);
      - sensitivity_ratio is computable.
    """
    set_seed(seed)
    device = torch.device("cpu")
    torch.manual_seed(seed)

    E_s = StateEncoder(
        resnet_encoder=_MockResNetEncoder(view_names),
        view_names=list(view_names),
        proprio_dim=proprio_dim,
        proprio_emb_dim=64,
    )
    model = ScoutVIB(action_dim=action_dim, E_s=E_s, beta=beta).to(device)

    def rand_obs(B):
        return {
            "visual": {v: torch.randn(B, 1, 3, 128, 128) for v in view_names},
            "proprio": torch.randn(B, 1, proprio_dim),
        }

    obs_t = rand_obs(batch_size); obs_tp1 = rand_obs(batch_size)
    A_t = torch.randn(batch_size, action_dim)

    # s_bar dim check
    s_bar = model.encode(obs_t)
    expected_s_bar = 512 * len(view_names) + 64
    assert s_bar.shape == (batch_size, expected_s_bar), \
        f"s_bar shape {s_bar.shape} != ({batch_size}, {expected_s_bar})"

    out = model(obs_t, A_t, obs_tp1)
    assert set(out) == {"loss", "latent_mse", "kl", "mu", "logvar"}
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-3)
    opt.zero_grad(); out["loss"].backward(); opt.step()

    # gradient isolation: ResNet must have NO grad; proprio_embed / vib_enc / D_s must.
    resnet_grads = [p.grad for p in model.E_s.resnet.parameters()]
    assert all(g is None for g in resnet_grads), \
        "ResNet should be frozen but got a gradient (anchor broken)"
    proprio_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.E_s.proprio_embed.parameters())
    vib_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.vib_enc.parameters())
    ds_has_grad = any(
        p.grad is not None and p.grad.abs().sum().item() > 0
        for p in model.D_s.parameters())
    assert proprio_has_grad and vib_has_grad and ds_has_grad, \
        "expected grads on proprio_embed / vib_enc / D_s"

    # sensitivity_ratio (sigma_a / sigma_mu from the dummy batch).
    sigma_a = float(A_t.std(dim=0).mean())
    with torch.no_grad():
        s_bar_t = model.encode(obs_t)
        mu_, _ = model.vib_enc(s_bar_t, A_t)
        sigma_mu = float(mu_.std(dim=0).mean())
    sr = sensitivity_ratio(model, obs_t, A_t, sigma_a, sigma_mu)

    print(f"[dummy] s_bar_dim = {expected_s_bar} (={512}*{len(view_names)} + 64)")
    print(f"[dummy] loss={out['loss'].item():.4f} "
          f"latent_mse={out['latent_mse'].item():.4f} kl={out['kl'].item():.4f}")
    print(f"[dummy] ResNet frozen (no grad): OK")
    print(f"[dummy] grads on proprio/vib_enc/D_s: OK")
    print(f"[dummy] sensitivity_ratio = {sr:.4f}  "
          f"(sigma_a={sigma_a:.4f}, sigma_mu={sigma_mu:.4f})")
    return {"s_bar_dim": expected_s_bar, "loss": float(out["loss"].item()),
            "latent_mse": float(out["latent_mse"].item()),
            "kl": float(out["kl"].item()), "sensitivity": sr}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="YAML config path (real run)")
    parser.add_argument("--dummy", action="store_true",
                        help="hermetic forward/backward smoke test (no dataset/ckpt)")
    args = parser.parse_args()
    if args.dummy:
        dummy_run()
    elif args.config:
        with open(args.config, "r") as f:
            cfg = EasyDict(yaml.safe_load(f))
        print(dict(cfg))
        run(cfg)
    else:
        parser.error("provide --config <yaml> or --dummy")
