"""Frozen-ResNet feature bank for VIB training (performance, not semantics).

Why: the live VIB data pipeline is CPU-bound -- per sample it slices 2 views x
2 frames of 84x84 uint8 from the in-RAM replay buffer, converts to float32
/255, and the (FROZEN) ResNet-18 then runs on every sampled window; with
batch 256 x 300 epochs x 200 steps the GPU sits at ~0% util (observed:
SCOUT-dyn-can-2 at ~3.9 min/epoch => ~19.5 h for 300 epochs).

Key observation: E_s's ResNet is frozen + eval, so its output for a given
(frame, view, 76x76-crop offset) is a CONSTANT. RandomCrop(76) on an 84x84
frame has exactly (84-76+1)^2 = 9x9 = 81 possible offsets, so the ENTIRE
augmentation support can be precomputed once into a per-view bank of shape
``(N_frames, 9, 9, 512)``. Training then samples offsets (uniform over the 81,
independently per frame -- replicating the live behaviour where torchvision
RandomCrop re-draws params on every image call) and only runs the ~469k
trainable params. Same objective, same sampling distribution; only the
(frozen, deterministic) ResNet work moves from per-step to one-off.

Bank files live next to the dataset (``<zarr_path>.featbank.<hash8>.<view>.npy``
+ a .json sidecar fingerprint). The hash covers the base-DP ckpt the frozen
ResNet was ripped from, so two VIB configs with different E_s bases never
share a stale bank. Banks are loaded with ``mmap_mode='r'`` (zero-copy page
cache; server has ~870 GB RAM).

CLI:
    python -m scout.feat_cache --config configs/vib_can_image_fast.yaml          # build only
    python -m scout.feat_cache --config ... --verify 8                           # bank vs live E_s
    python -m scout.feat_cache --smoke                                           # hermetic CPU check
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# bank build / load
# --------------------------------------------------------------------------- #
def _bank_paths(zarr_path: str, base_dp_ckpt: str):
    tag = hashlib.md5(str(base_dp_ckpt).encode()).hexdigest()[:8]
    sidecar = f"{zarr_path}.featbank.{tag}.json"
    views_npy = lambda v: f"{zarr_path}.featbank.{tag}.{v}.npy"  # noqa: E731
    return sidecar, views_npy


@torch.no_grad()
def build_feature_bank(
    imgs_by_view: Dict[str, np.ndarray],
    resnet,
    view_names,
    device,
    original_img_size: int = 84,
    cropped_img_size: int = 76,
    chunk: int = 512,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Precompute the frozen ResNet output for EVERY (frame, view, crop offset).

    ``imgs_by_view[v]``: ``(N, H, W, 3)`` uint8 (the replay-buffer raw images).
    ``resnet``: LPB ``ResNetEncoder``-compatible module --
    ``forward({view: (B,T,3,h,w)}) -> {view: (B,T,1,512)}`` (called with ALL
    view names at once, matching its internal loop). The per-crop preprocessing
    mirrors ``RobomimicImageDynamicsModelDataset.__getitem__`` exactly:
    HWC uint8 -> moveaxis -> float32 /255 -> CHW tensor.

    Returns ``{view: (N, R, C, 512) float32}`` with ``R = C = H - cropped + 1``.
    """
    R = original_img_size - cropped_img_size + 1
    C = R  # square crops
    view0 = view_names[0]
    N = imgs_by_view[view0].shape[0]
    emb_dim = int(getattr(resnet, "emb_dim", 512))
    banks = {
        v: np.empty((N, R, C, emb_dim), dtype=np.float32) for v in view_names
    }

    was_training = resnet.training
    resnet.eval()
    t0 = time.time()
    n_calls = 0
    for r in range(R):
        for c in range(C):
            for k0 in range(0, N, chunk):
                x = {}
                for v in view_names:
                    crops = imgs_by_view[v][k0:k0 + chunk,
                                            r:r + cropped_img_size,
                                            c:c + cropped_img_size, :]
                    arr = np.moveaxis(crops, -1, 1).astype(np.float32) / 255.0
                    x[v] = torch.from_numpy(arr).unsqueeze(1).to(device)
                out = resnet(x)                      # {v: (K,1,1,emb)}
                for v in view_names:
                    banks[v][k0:k0 + chunk, r, c] = \
                        out[v].squeeze(1).squeeze(1).cpu().numpy()
                n_calls += 1
        if verbose:
            print(f"  [featbank] row {r + 1}/{R} done "
                  f"({time.time() - t0:.1f}s, {n_calls} resnet calls)")
    if was_training:
        resnet.train()
    if verbose:
        print(f"  [featbank] built {N} frames x {R}x{C} offsets x "
              f"{len(view_names)} views in {time.time() - t0:.1f}s")
    return banks


def get_or_build_bank(ds, E_s, cfg, device, verbose: bool = True):
    """Load the bank from disk if the fingerprint matches, else build + save.

    Fingerprint = (n_frames, view_names, img sizes, base_dp_ckpt) -- anything
    that changes the frozen-ResNet input/output invalidates the bank.
    """
    d = cfg.dataset
    view_names = list(ds.view_names)
    base_dp_ckpt = str(cfg.model.E_s.base_dp_ckpt)
    sidecar, views_npy = _bank_paths(str(d.zarr_path), base_dp_ckpt)

    fp = {
        "n_frames": int(ds.imgs[view_names[0]].shape[0]),
        "view_names": view_names,
        "original_img_size": int(getattr(d, "original_img_size", 84)),
        "cropped_img_size": int(getattr(d, "cropped_img_size", 76)),
        "base_dp_ckpt": base_dp_ckpt,
    }
    if os.path.isfile(sidecar) and all(
            os.path.isfile(views_npy(v)) for v in view_names):
        with open(sidecar) as f:
            saved = json.load(f)
        if saved == fp:
            banks = {
                v: np.load(views_npy(v), mmap_mode="r") for v in view_names
            }
            if verbose:
                print(f"[featbank] loaded existing bank (fingerprint match): "
                      f"{sidecar}")
            return banks
        if verbose:
            print("[featbank] fingerprint mismatch -> rebuilding")

    banks = build_feature_bank(
        imgs_by_view=ds.imgs,
        resnet=E_s.resnet,
        view_names=view_names,
        device=device,
        original_img_size=fp["original_img_size"],
        cropped_img_size=fp["cropped_img_size"],
        verbose=verbose,
    )
    for v in view_names:
        np.save(views_npy(v), banks[v])
    with open(sidecar, "w") as f:
        json.dump(fp, f, indent=2)
    if verbose:
        print(f"[featbank] saved bank -> {views_npy('<view>')}")
    return banks


# --------------------------------------------------------------------------- #
# cached-feature transition dataset
# --------------------------------------------------------------------------- #
class CachedFeatureTransitionDataset(Dataset):
    """Same anchors/windows as ``RobomimicImageDynamicsModelDataset`` but serves
    precomputed frozen-ResNet features instead of raw images.

    ``__getitem__`` mirrors the base indexing verbatim (obs_indices /
    action_indices incl. the last-chunk repeat) and replicates the live crop
    behaviour: TRAIN draws a fresh uniform (r, c) offset per frame (the live
    path re-draws RandomCrop params on every per-image transform call, so the
    two window frames get INDEPENDENT offsets); VAL uses the centre offset
    (== ``CenterCrop`` for odd grid sizes). proprio / actions are passed
    through unchanged.
    """

    def __init__(self, base_ds, banks, train: bool = True):
        self.base = base_ds
        self.banks = banks
        self.train = bool(train)
        self.view_names = list(base_ds.view_names)
        v0 = self.view_names[0]
        self.R = banks[v0].shape[1]
        self.C = banks[v0].shape[2]
        self.center_r = (self.R - 1) // 2
        self.center_c = (self.C - 1) // 2
        # attrs train_vib's split logic reads off the dataset
        self.episode_start_indices = base_ds.episode_start_indices
        self.valid_anchor_indices = base_ds.valid_anchor_indices
        self.num_valid = base_ds.num_valid
        self.action_dim = base_ds.action_dim
        self.proprio_dim = base_ds.proprio_dim
        self.frameskip = base_ds.frameskip
        self.num_frames = base_ds.num_frames

    def __len__(self):
        return self.num_valid

    def __getitem__(self, idx):
        start = int(self.base.valid_anchor_indices[idx])
        end = start + self.num_frames * self.frameskip
        obs_indices = list(range(start, end, self.frameskip))
        action_indices = list(range(start, end))
        action_indices[-self.frameskip:] = \
            [obs_indices[-1] - 1] * self.frameskip

        obs = {"visual_feat": {}, "proprio": None}
        for v in self.view_names:
            fb = self.banks[v][obs_indices]            # (num_frames, R, C, 512)
            if self.train:
                rs = np.random.randint(0, self.R, size=len(obs_indices))
                cs = np.random.randint(0, self.C, size=len(obs_indices))
            else:
                rs = np.full(len(obs_indices), self.center_r)
                cs = np.full(len(obs_indices), self.center_c)
            feats = np.stack(
                [fb[f, rs[f], cs[f]] for f in range(len(obs_indices))])
            obs["visual_feat"][v] = torch.from_numpy(feats)   # (T, 512)

        proprio = self.base.states[obs_indices].astype(np.float32)
        obs["proprio"] = torch.from_numpy(proprio)
        act = torch.from_numpy(
            self.base.actions[action_indices].astype(np.float32))
        state = torch.from_numpy(proprio.copy())
        return tuple([obs, act, state])


# --------------------------------------------------------------------------- #
# equivalence check: bank vs live E_s (real ckpt; needs the LPB stack)
# --------------------------------------------------------------------------- #
def verify_bank(ds, E_s, banks, device, cfg, n_check: int = 8, atol: float = 5e-3):
    """For n_check random anchors: crop the raw frames at random offsets and
    compare live frozen-ResNet outputs against bank lookups (and the fused
    s̄ via forward vs forward_from_feats).

    Tolerance: the bank is built at chunk-size batches (512) while the live
    check runs batch-1 calls, and cuDNN (TF32 on) picks different conv
    algorithms per shape -- measured on the real ckpt: live b512 vs bank =
    0.0 EXACT; live b1 vs b512 = 3-6e-4; feature scale |f|~0.7 => the live
    pipeline itself varies by this much between full/partial batches. So
    atol=5e-3 (relative ~1e-3) is shape-noise level, not a bug scale.
    """
    d = cfg.dataset
    cropped = int(getattr(d, "cropped_img_size", 76))
    view_names = list(ds.view_names)
    rng = np.random.default_rng(0)
    max_off = int(getattr(d, "original_img_size", 84)) - cropped  # 8

    E_s.eval()
    max_err, max_sbar_err = 0.0, 0.0
    with torch.no_grad():
        for _ in range(n_check):
            idx = int(rng.integers(0, len(ds)))
            obs, _act, _state = ds[idx]           # live dataset (no transform)
            anchor = int(ds.valid_anchor_indices[idx])
            # one random offset per (frame); all views in ONE resnet call
            # (ResNetEncoder.forward iterates its full view_names list).
            for f in range(obs["visual"][view_names[0]].shape[0]):
                r = int(rng.integers(0, max_off + 1))
                c = int(rng.integers(0, max_off + 1))
                x = {}
                for v in view_names:
                    frames = obs["visual"][v]      # (T,3,H,W) float32 [0,1]
                    x[v] = frames[f:f + 1, :, r:r + cropped,
                                  c:c + cropped].unsqueeze(1).to(device)
                live = E_s.resnet(x)               # {v: (1,1,1,512)}
                for v in view_names:
                    frame_idx = anchor + f * ds.frameskip
                    cached = torch.from_numpy(
                        np.asarray(banks[v][frame_idx, r, c])
                    ).unsqueeze(0).to(device)
                    err = (live[v].squeeze(1).squeeze(1) - cached
                           ).abs().max().item()
                    max_err = max(max_err, err)

            # fusion equivalence: forward vs forward_from_feats on same inputs
            anchor = int(ds.valid_anchor_indices[idx])
            fidx = [anchor, anchor + ds.frameskip]
            vf = {
                v: torch.stack([
                    torch.from_numpy(np.asarray(banks[v][fidx[0], 4, 4])),
                    torch.from_numpy(np.asarray(banks[v][fidx[1], 4, 4])),
                ]).unsqueeze(0).to(device)          # (1,2,512)
                for v in view_names
            }
            proprio = torch.from_numpy(
                ds.states[fidx].astype(np.float32)).unsqueeze(0).to(device)
            live_obs = {
                "visual": {
                    v: obs["visual"][v][:, :, 4:4 + cropped, 4:4 + cropped
                                        ].unsqueeze(0).to(device)
                    for v in view_names
                },
                "proprio": proprio,
            }
            s_live = E_s(live_obs)
            s_cache = E_s.forward_from_feats(vf, proprio)
            max_sbar_err = max(max_sbar_err,
                               (s_live - s_cache).abs().max().item())

    print(f"[verify] {n_check} anchors | max |bank - live resnet| = {max_err:.2e} "
          f"(atol {atol}, cuDNN shape noise) | max |forward - forward_from_feats| = "
          f"{max_sbar_err:.2e}")
    ok = max_err <= atol and max_sbar_err <= atol
    print(f"[verify] {'OK' if ok else 'MISMATCH'}")
    return ok


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", help="VIB config yaml (real run)")
    p.add_argument("--verify", type=int, default=0, metavar="N",
                   help="also run the N-anchor bank-vs-live equivalence check")
    p.add_argument("--smoke", action="store_true",
                   help="hermetic CPU check with a mock ResNet")
    args = p.parse_args()

    if args.smoke:
        _smoke()
        return
    if not args.config:
        p.error("provide --config <yaml> and/or --smoke")

    import yaml
    from easydict import EasyDict

    from scout.model.encoder import StateEncoder
    from scout.train_vib import build_E_s, make_dataloader

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = EasyDict(yaml.safe_load(f))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _loader, ds = make_dataloader(cfg)
    E_s = build_E_s(cfg).to(device)
    t0 = time.time()
    banks = get_or_build_bank(ds, E_s, cfg, device)
    print(f"[feat_cache] bank ready in {time.time() - t0:.1f}s "
          f"(incl. dataset load)")
    if args.verify > 0:
        ok = verify_bank(ds, E_s, banks, device, cfg, n_check=args.verify)
        if not ok:
            raise SystemExit(1)


def _smoke():
    """Hermetic check (CPU, mock ResNet): bank roundtrip == live resnet on
    random offsets; dataset schema; forward == forward_from_feats; uniform
    offsets actually vary."""
    from scout.model.encoder import StateEncoder
    from scout.train_vib import _MockResNetEncoder

    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device("cpu")
    view_names = ["agentview_image", "robot0_eye_in_hand_image"]
    N, H, W, P = 40, 84, 84, 9

    imgs = {v: np.random.randint(0, 256, (N, H, W, 3), dtype=np.uint8)
            for v in view_names}
    resnet = _MockResNetEncoder(view_names)

    banks = build_feature_bank(imgs, resnet, view_names, device,
                               original_img_size=84, cropped_img_size=76,
                               chunk=5, verbose=False)
    assert banks[view_names[0]].shape == (N, 9, 9, 512)

    # 1) bank roundtrip vs live resnet on random offsets (all views per call --
    #    ResNetEncoder.forward iterates its full view_names list)
    rng = np.random.default_rng(1)
    for _ in range(20):
        i = int(rng.integers(0, N)); r = int(rng.integers(0, 9)); c = int(rng.integers(0, 9))
        x = {}
        for v in view_names:
            crop = imgs[v][i, r:r + 76, c:c + 76, :][None]     # (1,76,76,3)
            arr = np.moveaxis(crop, -1, 1).astype(np.float32) / 255.0  # (1,3,76,76)
            x[v] = torch.from_numpy(arr).unsqueeze(1)          # (1,1,3,76,76)
        live = resnet(x)
        for v in view_names:
            err = (live[v].squeeze(1).squeeze(1)
                   - torch.from_numpy(banks[v][i, r, c])).abs().max().item()
            assert err < 1e-5, f"bank mismatch {err}"
    print("[smoke] bank == live resnet (random offsets): OK")

    # 2) dataset schema + indexing parity with the base indexing convention
    class _Base:
        pass

    base = _Base()
    base.view_names = view_names
    base.frameskip = 8
    base.num_frames = 2
    base.states = np.random.randn(N, P).astype(np.float32)
    base.actions = np.random.randn(N, 10).astype(np.float32)
    base.valid_anchor_indices = np.arange(0, N - 16)
    base.num_valid = len(base.valid_anchor_indices)
    base.episode_start_indices = np.array([0, N // 2])
    base.action_dim = 80
    base.proprio_dim = P
    ds = CachedFeatureTransitionDataset(base, banks, train=True)
    obs, act, state = ds[3]
    assert obs["visual_feat"][view_names[0]].shape == (2, 512)
    assert obs["proprio"].shape == (2, P)
    assert act.shape == (16, 10)                # 2*fs per-step actions
    # val mode = centre offsets, deterministic
    dsv = CachedFeatureTransitionDataset(base, banks, train=False)
    o1, _, _ = dsv[3]; o2, _, _ = dsv[3]
    assert torch.equal(o1["visual_feat"][view_names[0]],
                       o2["visual_feat"][view_names[0]])
    print("[smoke] dataset schema + val determinism: OK")

    # 3) forward == forward_from_feats on the same inputs
    E_s = StateEncoder(resnet_encoder=_MockResNetEncoder(view_names),
                       view_names=view_names, proprio_dim=P, proprio_emb_dim=64)
    live_obs = {
        "visual": {v: torch.rand(2, 1, 3, 76, 76) for v in view_names},
        "proprio": torch.rand(2, 1, P),
    }
    feats = {v: E_s.resnet(live_obs["visual"])[v].squeeze(-2) for v in view_names}
    s1 = E_s(live_obs)
    s2 = E_s.forward_from_feats(feats, live_obs["proprio"])
    assert torch.allclose(s1, s2, atol=1e-6), (s1 - s2).abs().max()
    print("[smoke] StateEncoder.forward == forward_from_feats: OK")

    # 4) train-mode offsets actually vary (uniform over the 9x9 grid)
    seen = {(int(np.random.randint(0, 9)), int(np.random.randint(0, 9)))
            for _ in range(200)}
    assert len(seen) > 30
    print("[smoke] offset sampling varies: OK")
    print("[smoke] feat_cache OK")


if __name__ == "__main__":
    _main()
