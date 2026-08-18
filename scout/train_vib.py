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
import math
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
    obs, act, _state = batch[:3]
    w = batch[3] if len(batch) > 3 else None   # failure weight (train only)
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
    return obs_t, a_t, obs_tp1, w


def _slice_transition_feats(batch, device):
    """Feature-cache variant of :func:`_slice_transition`.

    ``batch`` comes from
    :class:`scout.feat_cache.CachedFeatureTransitionDataset` -- ``obs`` carries
    PRECOMPUTED frozen-ResNet features (``obs["visual_feat"][v]``:
    (B, T, 512)) instead of raw images, so no crop/transform happens here.
    Returns ``(feat_obs_t, a_t, feat_obs_tp1)`` in the layout
    :meth:`ScoutVIB.forward_feats` consumes. Identical anchor/action slicing
    to the live path.
    """
    obs, act, _state = batch[:3]
    w = batch[3] if len(batch) > 3 else None   # failure weight (train only)
    vf = obs["visual_feat"]
    proprio = obs["proprio"]
    obs_t = {
        "visual_feat": {v: f[:, 0:1].to(device) for v, f in vf.items()},
        "proprio": proprio[:, 0:1].to(device),
    }
    obs_tp1 = {
        "visual_feat": {v: f[:, 1:2].to(device) for v, f in vf.items()},
        "proprio": proprio[:, 1:2].to(device),
    }
    fs = act.shape[1] // 2
    a_t = act[:, :fs].reshape(act.shape[0], -1).to(device)
    return obs_t, a_t, obs_tp1, w


@torch.no_grad()
def eval_val_mse(model, val_loader, device, val_img_transform, feats_mode=False):
    """Mean latent_mse over the val split (model.eval; restores train() after).

    Held-out demos -> distinguishes trivial-prediction (val~train~0) from
    overfitting (val >> train). Uses the center-crop val transform (no random aug)
    -- on the feats path the center offsets are baked into the val dataset.
    """
    if val_loader is None:
        return None
    model.eval()
    mses = []
    for b in val_loader:
        if feats_mode:
            obs_t, A_t, obs_tp1, _w = _slice_transition_feats(b, device)
            out = model.forward_feats(obs_t, A_t, obs_tp1)
        else:
            obs_t, A_t, obs_tp1, _w = _slice_transition(b, device, val_img_transform)
            out = model(obs_t, A_t, obs_tp1)
        mses.append(out["latent_mse"].item())
    model.train()
    return float(np.mean(mses)) if mses else None


# --------------------------------------------------------------------------- #
# VIB training (single β; no diagnostic -- stage1_plan.md Step 1)
# --------------------------------------------------------------------------- #
def train_one_beta(cfg, train_loader, val_loader, ds, E_s_cfg, action_dim, beta,
                   device, out_dir, train_t=None, val_t=None, val_every=20,
                   wandb_run=None, E_s=None, feats_mode=False,
                   metric_prefix=""):
    if E_s is None:
        E_s = build_E_s(E_s_cfg)
    model = ScoutVIB(
        action_dim=action_dim,
        E_s=E_s,
        style_dim=cfg.model.style_dim,
        hidden_dim=cfg.model.hidden_dim,
        beta=beta,
        free_bits=float(cfg.get("free_bits", 0.0)),
    ).to(device)
    # only trainable params: vib_enc / D_s / proprio_embed (ResNet frozen).
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, **cfg.optimizer.params)

    steps_per_epoch = int(cfg.steps_per_epoch)
    num_epochs = int(cfg.num_epochs)
    # LR schedule (user 2026-08-17): linear warmup -> cosine decay to 5% peak.
    total_steps = steps_per_epoch * num_epochs
    _ocfg = cfg.get("optimizer", {}) or {}
    warmup_steps = max(1, int(_ocfg.get("warmup_epochs", 5)) * steps_per_epoch)
    scheduler = None
    if str(_ocfg.get("lr_scheduler", "cosine")) == "cosine":
        def _lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * t))
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda)

    def _liveness(s_bar_p, a_p):
        """First-layer ReLU alive fraction + posterior KL on a REAL batch.

        2026-08-17 postmortem: the dead-ReLU encoder was a silent constant
        function for an entire run (kl 3e-6 nats, guidance gradient exactly
        0). These two numbers make that failure loud instead of silent.
        """
        with torch.no_grad():
            x = model.vib_enc.in_norm(torch.cat([s_bar_p, a_p], dim=-1))
            pre = model.vib_enc.net.encoder[0](x)
            alive = (pre > 0).float().mean().item()
            mu, logvar = model.vib_enc(s_bar_p, a_p)
            kl = 0.5 * (mu.pow(2) + logvar.exp() - 1.0 - logvar).sum(1).mean().item()
        return alive, kl

    history = {"latent_mse": [], "kl": [], "mu_abs": [], "val_mse": []}
    probe = None   # (s_bar, A) of the first real batch, for liveness checks

    model.train()
    for epoch in range(num_epochs):
        # accumulate metrics as GPU tensors; .item() once per EPOCH (3 per-step
        # GPU syncs were ~half the step time on the cached-features path,
        # where the compute itself is microseconds).
        ep = {"latent_mse": None, "kl": None, "mu_abs": None, "n": 0}
        for it, batch in enumerate(train_loader):
            if feats_mode:
                obs_t, A_t, obs_tp1, w = _slice_transition_feats(batch, device)
                out = model.forward_feats(obs_t, A_t, obs_tp1)
            else:
                obs_t, A_t, obs_tp1, w = _slice_transition(batch, device, train_t)
                out = model(obs_t, A_t, obs_tp1)
            if probe is None:
                with torch.no_grad():
                    s_p = (model.encode_from_feats(obs_t) if feats_mode
                           else model.encode(obs_t))
                    probe = (s_p.detach(), A_t.detach())
            # weighted reconstruction + free-bits KL (failure_weight / free_bits;
            # w=None or fw=1 or fb=0 reproduces the old objective exactly)
            if w is not None:
                w = w.to(device)
                mse_loss = (w * out["latent_mse_per"]).sum() / w.sum()
            else:
                mse_loss = out["latent_mse"]
            loss = mse_loss + beta * out["kl_fb"]
            opt.zero_grad(); loss.backward(); opt.step()
            if scheduler is not None:
                scheduler.step()

            # accumulate over EVERY batch (sampling every-N skipped small-data
            # epochs entirely -- lift has 27 batches/epoch < old log_every_batch=40,
            # so it reported a bogus train loss of 0).
            lm = out["latent_mse"].detach()
            kl = out["kl"].detach()
            ma = out["mu"].detach().abs().mean()
            ep["latent_mse"] = lm if ep["latent_mse"] is None else ep["latent_mse"] + lm
            ep["kl"] = kl if ep["kl"] is None else ep["kl"] + kl
            ep["mu_abs"] = ma if ep["mu_abs"] is None else ep["mu_abs"] + ma
            ep["n"] += 1
            if it + 1 >= steps_per_epoch:
                break
        n = max(1, ep["n"])
        history["latent_mse"].append(ep["latent_mse"].item() / n)
        history["kl"].append(ep["kl"].item() / n)
        history["mu_abs"].append(ep["mu_abs"].item() / n)

        # liveness DIAGNOSTICS (2026-08-17 postmortem). User 2026-08-18:
        # sentinels CANCELLED -- the aborts fired on fb=0.005 explore-data
        # retrains (can r1: alive 0.0079) and would kill legitimate sparse
        # solutions; free_bits alone now carries the collapse support. The
        # numbers are still printed every epoch so degradation is visible.
        alive, kl_probe = _liveness(*probe)
        if alive < 0.01:
            print(f"  [liveness] relu_alive {alive:.3f} < 0.01 (was: ABORT; now: observe-only)")
        elif alive < 0.10:
            print(f"  [liveness] WARNING: relu_alive {alive:.3f} is low (sparse)")
        if epoch >= 10 and history["kl"][-1] < 0.01:
            print(f"  [liveness] kl {history['kl'][-1]:.4f} < 0.01 (was: ABORT; now: observe-only)")
        if epoch % 10 == 0:
            print(f"  [liveness] relu_alive {alive:.2f} kl_probe {kl_probe:.3f}")

        # val on held-out demos: trivial-prediction (val~train) vs overfit (val>>train)
        val_mse = None
        do_val = val_loader is not None and (
            epoch % max(1, val_every) == 0 or epoch == num_epochs - 1)
        if do_val:
            val_mse = eval_val_mse(model, val_loader, device, val_t,
                                   feats_mode=feats_mode)
            history["val_mse"].append(val_mse)
        print(f"  [β={beta:g}] epoch {epoch:4d} | latent_mse {history['latent_mse'][-1]:.4f} "
              f"kl {history['kl'][-1]:.4f} |μ| {history['mu_abs'][-1]:.4f}"
              + (f" | val_mse {val_mse:.4f}" if val_mse is not None else ""))
        if wandb_run is not None:
            if metric_prefix:
                # experiment2 round-run section: dyn/{latent_mse,kl,lr,epoch}.
                # NO explicit step=epoch here: the resumed round-run's global
                # counter sits at the DP stage's ~5x10^4 steps and wandb 0.28
                # drops any explicit step below it (observed 2026-08-18:
                # every dyn/* row discarded with "monotonically increasing"
                # warnings). Auto-increment keeps the counter monotonic; the
                # charts' x-axis is dyn/epoch via define_metric(step_metric).
                log_d = {metric_prefix + "latent_mse": history["latent_mse"][-1],
                         metric_prefix + "kl": history["kl"][-1],
                         metric_prefix + "lr": opt.param_groups[0]["lr"],
                         metric_prefix + "epoch": epoch}
                wandb_run.log(log_d)
            else:
                log_d = {"latent_mse": history["latent_mse"][-1],
                         "kl": history["kl"][-1],
                         "mu_abs": history["mu_abs"][-1]}
                if val_mse is not None:
                    log_d["val_mse"] = val_mse
                wandb_run.log(log_d, step=epoch)

    # save ckpt (single-β flow; no diagnostic -- stage1_plan.md Step 1).
    # view_names + proprio key order ride along so the eval-side factory can
    # assert the E_s fusion order matches training (state_dict keys alone
    # cannot catch a same-set-different-order permutation).
    proprio_keys = [k for k, m in dict(cfg.dataset.shape_obs).items()
                    if m.get("type") != "rgb" and "image" not in k]
    torch.save({"state_dict": model.state_dict(), "beta": beta,
                "view_names": list(ds.view_names),
                "proprio_keys": proprio_keys},
               os.path.join(out_dir, "scout_vib.ckpt"))
    plot_curves(history, ["latent_mse", "kl"], f"VIB losses (β={beta:g})",
                os.path.join(out_dir, "losses.png"))
    plot_curves(history, ["val_mse"], f"val latent_mse (β={beta:g})",
                os.path.join(out_dir, "val_mse.png"))
    plot_curves(history, ["mu_abs"], f"|μ| (β={beta:g})",
                os.path.join(out_dir, "mu.png"))

    # guidance-gradient liveness on a REAL batch (2026-08-17 postmortem): the
    # rollout guidance was silently an exact no-op when the encoder was a
    # dead-ReLU constant. The bridge used at rollout is affine, so nonzero
    # here <=> nonzero through the real guidance path.
    s_p, a_p = probe
    a_g = a_p.clone().requires_grad_(True)
    mu_g, logvar_g = model.vib_enc(s_p, a_g)
    z_g = torch.randn_like(mu_g)
    nll_g = 0.5 * ((z_g - mu_g).pow(2) / logvar_g.exp() + logvar_g).sum(1).mean()
    grad_g, = torch.autograd.grad(nll_g, a_g)
    g_norm = grad_g.norm().item()
    print(f"[guidance-check] |dNLL/da| on a real batch = {g_norm:.3e}")
    if not (g_norm > 0.0):
        print("[guidance-check] WARNING: d NLL/d a is exactly zero -- "
              "guidance would be a no-op (was: ABORT; now: observe-only, user 2026-08-18)")

    return {"beta": beta,
            "latent_mse": history["latent_mse"][-1],
            "kl": history["kl"][-1],
            "mu_abs": history["mu_abs"][-1],
            "val_mse": history["val_mse"][-1] if history["val_mse"] else None}


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

    # Build E_s ONCE (train_one_beta reuses it; the feature bank reuses its
    # frozen ResNet). Costs one base-DP ckpt load either way.
    E_s = build_E_s(cfg)

    # Optional frozen-ResNet feature cache (scout/feat_cache.py): precompute the
    # ResNet output for EVERY (frame, view, 76x76-crop offset) once, then train
    # on cached features. Same objective + same RandomCrop distribution (81
    # offsets, per-frame draws) -- turns the CPU-bound ~4min/epoch pipeline
    # into a few-minutes total run. val uses centre offsets (CenterCrop).
    feats_mode = bool(getattr(cfg.dataset, "feature_cache", False))
    train_ds, val_ds = ds, ds
    if feats_mode:
        from scout.feat_cache import (
            CachedFeatureTransitionDataset, get_or_build_bank)
        banks = get_or_build_bank(ds, E_s, cfg, device)
        ds.imgs = {}   # free raw in-RAM images (states/actions/anchors stay)
        train_ds = CachedFeatureTransitionDataset(ds, banks, train=True)
        val_ds = CachedFeatureTransitionDataset(ds, banks, train=False)
        print(f"feature_cache: ON -- {len(train_ds)} train anchors, "
              f"train=uniform-81-offsets / val=centre-offset")

    # train/val image transforms (random crop train, center crop val).
    from dyn_model.datasets.img_transforms import (
        get_train_crop_transform_resnet, get_eval_crop_transform_resnet)
    train_t = val_t = None
    if (not feats_mode) and bool(getattr(cfg.dataset, "use_crop", False)):
        train_t = get_train_crop_transform_resnet(
            int(cfg.dataset.original_img_size), int(cfg.dataset.cropped_img_size))
        val_t = get_eval_crop_transform_resnet(
            int(cfg.dataset.original_img_size), int(cfg.dataset.cropped_img_size))
        print(f"img_transform: RandomCrop({cfg.dataset.cropped_img_size}) train / "
              f"CenterCrop val, on {cfg.dataset.original_img_size}")
    elif feats_mode:
        print("img_transform: baked into the feature bank "
              "(uniform 81 offsets train / centre val)")
    else:
        print("img_transform: none (full image)")

    # episode-level train/val split: hold out the last val_ratio of demos for val
    # (anchors whose frame index >= the first val episode's start belong to val).
    from torch.utils.data import DataLoader, Subset
    val_ratio = float(getattr(cfg, "val_ratio", 0.1))
    n_eps = len(train_ds.episode_start_indices)
    n_val_eps = max(1, int(round(n_eps * val_ratio))) if n_eps > 1 else 0
    if 0 < n_val_eps < n_eps:
        cutoff = int(train_ds.episode_start_indices[n_eps - n_val_eps])
        anchors = train_ds.valid_anchor_indices
        train_idx = np.where(anchors < cutoff)[0]
        val_idx = np.where(anchors >= cutoff)[0]
    else:
        train_idx = np.arange(len(train_ds))
        val_idx = np.array([], dtype=np.int64)
    # failure up-weighting (user 2026-08-17): weight the reconstruction of
    # every anchor from a NON-success trajectory by cfg.failure_weight
    # (success/core stay 1.0; val stays unweighted -- honest generalization).
    fw = float(cfg.get("failure_weight", 1.0))
    train_view = train_ds
    if fw != 1.0:
        import h5py
        def _dn(k):
            import re as _re
            m = _re.search(r"(\d+)$", k)
            return int(m.group(1)) if m else 0
        with h5py.File(cfg.dataset.zarr_path, "r") as f:
            demos = sorted([k for k in f["data"].keys() if k.startswith("demo")], key=_dn)
            succ = [bool(np.asarray(f["data"][k]["success"]).ravel()[0])
                    if "success" in f["data"][k] else True for k in demos]
        ep_w = np.array([fw if not sc else 1.0 for sc in succ], dtype=np.float32)
        anchors_ep = np.searchsorted(ds.episode_ends, train_ds.valid_anchor_indices, side="right")
        w_all = ep_w[anchors_ep]
        n_fail = int((~np.array(succ)).sum())
        class _WeightedView(torch.utils.data.Dataset):
            def __init__(self, base, w):
                self.base, self.w = base, w
            def __len__(self):
                return len(self.base)
            def __getitem__(self, i):
                return (*self.base[i], self.w[i])
        train_view = _WeightedView(train_ds, w_all)
        print(f"failure_weight: {fw:g} -- {n_fail}/{len(succ)} failure episodes upweighted "
              f"(mean anchor weight {w_all.mean():.2f})")
    nw = int(getattr(cfg.dataset, "num_workers", 0))
    train_loader = DataLoader(Subset(train_view, train_idx), batch_size=int(cfg.batch_size),
                              shuffle=True, num_workers=nw, drop_last=True)
    val_loader = (DataLoader(Subset(val_ds, val_idx), batch_size=int(cfg.batch_size),
                             shuffle=False, num_workers=nw)
                  if len(val_idx) > 0 else None)
    print(f"split: {len(train_idx)} train / {len(val_idx)} val anchors "
          f"({n_val_eps}/{n_eps} demos held out for val)")

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
        # opt-in metric prefix (experiment2: dyn/* section of a shared
        # round-run; also honors WANDB_RUN_ID resume from the driver)
        metric_prefix = str(wcfg.get("metric_prefix", "") or "")
        if metric_prefix:
            wandb.define_metric(metric_prefix + "epoch", hidden=True)
            wandb.define_metric(metric_prefix + "*",
                                step_metric=metric_prefix + "epoch")
        print(f"wandb: project={wcfg.get('project', 'scout-dynamics')} "
              f"name={wcfg.get('name')} prefix={metric_prefix or None}")
    else:
        print("wandb: disabled")
        metric_prefix = ""

    val_every = int(getattr(cfg, "val_every", 20))
    print(f"\n=== training β={beta:g} (feats_mode={feats_mode}) ===")
    s = train_one_beta(cfg, train_loader, val_loader, ds, cfg, action_dim, beta, device,
                       run_root, train_t=train_t, val_t=val_t, val_every=val_every,
                       wandb_run=wandb_run, E_s=E_s, feats_mode=feats_mode,
                       metric_prefix=metric_prefix)
    print(f"=== done | β={beta:g} | latent_mse={s['latent_mse']:.4f} "
          f"kl={s['kl']:.4f} |μ|={s['mu_abs']:.4f} ===")

    with open(os.path.join(run_root, "summary.yaml"), "w") as f:
        yaml.safe_dump(to_plain(s), f, default_flow_style=False)
    if wandb_run is not None:
        # /final = cross-stage summary of the shared round-run (rollout adds
        # eval/explore, DP retrain adds dp_train_loss); mu_abs/val_mse stay as
        # dyn-only collapse/overfit diagnostics alongside the renamed pair
        wandb_run.log({"final/dyn_mse_loss": s["latent_mse"], "final/dyn_kl_loss": s["kl"],
                       "final/mu_abs": s["mu_abs"], "final/val_mse": s.get("val_mse")})
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
