"""E0: train the base Diffusion Policy (unguided) on robomimic low_dim.

Single-GPU, no DDP. Minimal loop mirroring SOE ``src/train_single_gpu.py``:
AdamW + cosine LR (with warmup); ``DP.forward(obs_dict, actions)`` returns a
scalar loss; ckpt + loss PNG saved every ``save_epochs``.

Why a chunked loader instead of ``scout.data.RobomimicLowdimSource``:
the base DP needs chunked ``(obs_dict, action_chunk)`` batches with per-key
observations -- a different shape from ``RobomimicLowdimSource``'s per-frame
``(S_t, A_t, S_{t+1})`` transitions (which feed the VIB in Phase 3). The two
loaders are complementary, not duplicative. This one is modelled on SOE
``src/dataset/robomimic_v2.py`` but stripped to the low_dim essentials.

Usage:
    python -m scout.train_base_dp --config configs/base_dp_lift_lowdim.yaml
"""

import argparse
import os
import time

import h5py
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from easydict import EasyDict
from diffusers.optimization import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader

from scout.policy.dp import DP


# --------------------------------------------------------------------------- #
# utilities
# --------------------------------------------------------------------------- #
def to_plain(obj):
    """Recursively convert EasyDicts (and tuples) into plain yaml-safe types."""
    if isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_mask_demos(hdf5, mask_key):
    """All ``demo_*`` names under ``data/``, optionally filtered by ``mask/<key>``."""
    all_demos = sorted([k for k in hdf5["data"].keys() if k.startswith("demo")])
    if mask_key is None or f"mask/{mask_key}" not in hdf5:
        return all_demos
    node = hdf5[f"mask/{mask_key}"]
    if isinstance(node, h5py.Group):
        if "mask" in node:
            arr = node["mask"][()]
            if arr.dtype == bool:
                return [d for d, keep in zip(all_demos, arr) if keep]
        return all_demos
    arr = node[()]
    if arr.dtype == bool:
        return [d for d, keep in zip(all_demos, arr) if keep]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


def plot_history(history, save_path):
    if not history:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(history, linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("base DP training loss")
    fig.tight_layout()
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# chunked robomimic low_dim dataset (num_obs = 1)
# --------------------------------------------------------------------------- #
class RobomimicLowdimChunkDataset(Dataset):
    """Chunked ``(obs_dict, action_chunk)`` loader for the base DP, low_dim only.

    Mirrors SOE ``robomimic_v2`` chunking with ``num_obs=1`` and
    ``action_offset=1``: sample ``i`` is
    ``(obs_t = state[cur], action_chunk = actions[cur+1 : cur+1+num_action])``
    with end-padding (repeat the last frame's action). This matches the
    Phase-1 ``RobomimicLowdimSource`` alignment. Only the low_dim obs keys in
    ``obs_shape_meta`` are emitted, as per-key ``(1, dim)`` tensors.
    """

    def __init__(self, path, obs_keys, num_action=20, mask_key="train"):
        self.num_action = int(num_action)
        self.obs_keys = list(obs_keys)
        self.demos_obs = []        # list of {key: (T, dim)}
        self.demos_actions = []    # list of (T, action_dim)
        self.index = []            # (demo_id, cur_idx)

        with h5py.File(path, "r") as f:
            demos = load_mask_demos(f, mask_key)
            assert len(demos) > 0, f"no demos under mask='{mask_key}' in {path}"
            for d_id, demo in enumerate(demos):
                grp = f["data"][demo]
                obs_d = {k: grp["obs"][k][()].astype(np.float32) for k in self.obs_keys}
                acts = grp["actions"][()].astype(np.float32)
                T = acts.shape[0]
                assert T - 1 >= 1, f"demo {demo} too short (T={T})"
                self.demos_obs.append(obs_d)
                self.demos_actions.append(acts)
                for cur in range(T - 1):
                    self.index.append((d_id, cur))
        print(f"[dataset] {len(self.index)} chunked samples from {len(demos)} demos")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        d_id, cur = self.index[i]
        obs_d = self.demos_obs[d_id]
        acts = self.demos_actions[d_id]
        T = acts.shape[0]
        # obs: num_obs=1 -> (1, dim) per key (collate squeezes to (B, dim))
        obs_dict = {k: torch.from_numpy(obs_d[k][cur:cur + 1].copy()).float()
                    for k in self.obs_keys}
        # action chunk: acts[cur+1 : cur+1+num_action], end-pad with last action
        start = cur + 1
        end = min(start + self.num_action, T)
        chunk = acts[start:end]
        if end - start < self.num_action:
            pad = np.repeat(acts[-1:], self.num_action - (end - start), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        return {"obs": obs_dict, "action": torch.from_numpy(chunk).float()}


def collate_fn(batch):
    """Stack per-key obs (squeeze num_obs=1) + action chunks into a DP-ready batch."""
    obs_keys = batch[0]["obs"].keys()
    obs_dict = {k: torch.stack([b["obs"][k] for b in batch], dim=0).squeeze(1)
                for k in obs_keys}                              # (B, dim) per key
    actions = torch.stack([b["action"] for b in batch], dim=0)  # (B, num_action, action_dim)
    return {"obs_dict": obs_dict, "actions": actions}


# --------------------------------------------------------------------------- #
# training loop
# --------------------------------------------------------------------------- #
def train(cfg):
    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # dataset / loader
    obs_keys = list(cfg.policy.params.obs_shape_meta.keys())
    dataset = RobomimicLowdimChunkDataset(
        path=cfg.dataset.path,
        obs_keys=obs_keys,
        num_action=cfg.policy.params.num_action,
        mask_key=cfg.dataset.get("train_filter_key", "train"),
    )
    dataloader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.dataset.get("num_workers", 0),
        collate_fn=collate_fn,
        shuffle=True,
        drop_last=True,
    )

    # policy
    policy = DP(**cfg.policy.params).to(device)
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"trainable params: {n_params / 1e6:.2f}M")

    # optimizer + scheduler
    optimizer = torch.optim.AdamW(policy.parameters(), **cfg.optimizer.params)
    total_steps = len(dataloader) * cfg.num_epochs
    warmup = int(cfg.lr_scheduler.get("num_warmup_steps", 2000))
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=min(warmup, max(1, total_steps // 10)),
        num_training_steps=total_steps,
    )

    # output dirs
    log_root = os.path.join(cfg.log_dir, time.strftime("%Y%m%d-%H%M%S", time.localtime()))
    ckpt_dir = os.path.join(log_root, "ckpt")
    os.makedirs(ckpt_dir, exist_ok=True)
    with open(os.path.join(log_root, "config.yaml"), "w") as f:
        yaml.safe_dump(to_plain(cfg), f, default_flow_style=False)

    # train
    train_history = []
    policy.train()
    for epoch in range(cfg.num_epochs):
        epoch_loss = 0.0
        n_batches = 0
        for batch in dataloader:
            obs_dict = {k: v.to(device) for k, v in batch["obs_dict"].items()}
            actions = batch["actions"].to(device)
            loss = policy(obs_dict, actions)        # scalar
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(1, n_batches)
        train_history.append(avg)
        print(f"epoch {epoch:4d} | loss {avg:.6f}")

        save_now = ((epoch + 1) % cfg.save_epochs == 0) or (epoch + 1 == cfg.num_epochs)
        if save_now:
            torch.save(policy.state_dict(),
                       os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}.ckpt"))
            plot_history(train_history, os.path.join(ckpt_dir, "loss.png"))
            print(f"  saved ckpt + loss.png @ epoch {epoch + 1}")

    return log_root


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="YAML config path")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        cfg = EasyDict(yaml.safe_load(f))
    print(dict(cfg))
    train(cfg)
