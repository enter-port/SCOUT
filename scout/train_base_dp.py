"""E0 base Diffusion Policy training -- thin pointer to the LPB ``train.py``.

**Entry point**: the repo-root ``train.py`` (hydra + ``TrainDiffusionUnetHybridWorkspace``)
plus an LPB-format yaml in ``configs/`` (default ``configs/base_dp_lift_image.yaml``).
The base DP is the LPB ``DiffusionUnetHybridImagePolicy`` -- the same class
:class:`scout.guidance.policy.ScoutPolicy` subclasses at inference. Training is
**not** reimplemented here; LPB's workspace already owns the AdamW + cosine-LR +
EMA + rollout/checkpoint loop (see
``diffusion_policy/workspace/train_diffusion_unet_hybrid_workspace.py``).

This module exists so :mod:`scout.eval.self_improvement`'s ``default_retrain_fn``
has a single SCOUT-side entry point that writes an augmented-hdf5 round config
and shells out to LPB ``train.py`` (the SOE ``run_full_multi_round`` pattern:
each round = subprocess train on augmented data; **no** in-memory buffer).

Usage (manual / round 0):

    python train.py --config-path configs --config-name base_dp_lift_image \\
        task.dataset_path=<core_hdf5> \\
        training.num_epochs=<N> hydra.run.dir=<log_dir>

Or programmatically:

    from scout.train_base_dp import train
    ckpt_path = train(cfg_overrides={"task.dataset_path": "...", ...},
                      config_name="base_dp_lift_image",
                      config_dir="configs", log_dir="logs/round_0",
                      num_epochs=1500)

The lazy imports (``robomimic``, ``hydra``, ``diffusion_policy``) are deferred to
:func:`train` / :func:`build_lpb_overrides` so this module imports cleanly in
environments without the LPB stack (the verify host) -- only the actual
subprocess call needs the training env.

.. note:: The LPB workspace saves checkpoints as ``checkpoints/<epoch>.ckpt`` and
   ``checkpoints/latest.ckpt`` (``save_last_ckpt=True``). :func:`train` returns
   the ``latest.ckpt`` path -- the canonical "DP_i" handle the self-improvement
   loop threads round-to-round.
"""
from __future__ import annotations

import os
import subprocess
from typing import Any, Dict, List, Optional


# --------------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
LPB_TRAIN_PY = os.path.join(REPO_ROOT, "train.py")
DEFAULT_CONFIG_DIR = os.path.join(REPO_ROOT, "configs")
DEFAULT_CONFIG_NAME = "base_dp_lift_image"


def build_lpb_overrides(
    *,
    dataset_path: Optional[str] = None,
    train_filter_key: Optional[str] = None,
    num_epochs: Optional[int] = None,
    resume_ckpt: Optional[str] = None,
    log_dir: Optional[str] = None,
    device: Optional[str] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Compose the hydra CLI override list for one LPB training run.

    Only non-``None`` keys are emitted. Keys map to the LPB
    ``train_diffusion_unet_hybrid_workspace`` schema:

      ``task.dataset_path``         -- robomimic hdf5 (core or augmented).
      ``task.train_filter_key``     -- mask/<key> selecting demos.
      ``training.num_epochs``       -- round epoch budget.
      ``training.resume``           -- True if ``resume_ckpt`` given.
      ``hydra.run.dir``             -- output dir for this run.
      ``training.device``           -- "cuda:0" etc.

    ``extra`` is appended verbatim (caller-specified hydra overrides).
    """
    overrides: List[str] = []
    if dataset_path is not None:
        overrides.append(f"task.dataset_path={os.path.abspath(dataset_path)}")
    if train_filter_key is not None:
        overrides.append(f"task.train_filter_key={train_filter_key}")
    if num_epochs is not None:
        overrides.append(f"training.num_epochs={int(num_epochs)}")
    if resume_ckpt:
        # A non-empty resume path -> ask the LPB workspace to resume from its
        # checkpoint dir (LPB's get_checkpoint_path finds <log_dir>/checkpoints/
        # latest.ckpt). Caller points log_dir at the prior round's dir.
        overrides.append("training.resume=True")
    if log_dir is not None:
        overrides.append(f"hydra.run.dir={os.path.abspath(log_dir)}")
    if device is not None:
        overrides.append(f"training.device={device}")
    if extra:
        for k, v in extra.items():
            overrides.append(f"{k}={v}")
    return overrides


# --------------------------------------------------------------------------- #
# train entry (subprocess to LPB train.py)
# --------------------------------------------------------------------------- #
def train(
    cfg_overrides: Optional[Dict[str, Any]] = None,
    *,
    config_name: str = DEFAULT_CONFIG_NAME,
    config_dir: str = DEFAULT_CONFIG_DIR,
    log_dir: Optional[str] = None,
    num_epochs: Optional[int] = None,
    dataset_path: Optional[str] = None,
    train_filter_key: Optional[str] = None,
    resume_ckpt: Optional[str] = None,
    device: Optional[str] = None,
    extra_overrides: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    cuda_visible_devices: Optional[str] = None,
) -> str:
    """Shell out to LPB ``train.py`` (repo root) and return the new ckpt path.

    Aggregates all kwargs + ``cfg_overrides``/``extra_overrides`` into hydra CLI
    overrides via :func:`build_lpb_overrides`, then ``subprocess.run`` the LPB
    entry. Returns ``<log_dir>/checkpoints/latest.ckpt`` (the LPB
    ``save_last_ckpt=True`` artefact). Use ``dry_run=True`` to get the command +
    path without executing (used by the self-improvement mock -- real training
    needs the SOE/LPB conda env).

    The returned path is what :class:`scout.eval.self_improvement.SelfImprovementLoop`
    threads into the next round's ``dp_factory``.
    """
    merged = dict(cfg_overrides or {})
    # explicit kwargs take precedence over cfg_overrides dict values
    if dataset_path is not None:
        merged["task.dataset_path"] = dataset_path
    if train_filter_key is not None:
        merged["task.train_filter_key"] = train_filter_key
    if num_epochs is not None:
        merged["training.num_epochs"] = num_epochs
    if device is not None:
        merged["training.device"] = device
    if log_dir is not None:
        merged["hydra.run.dir"] = log_dir
    if extra_overrides:
        merged.update(extra_overrides)

    overrides = build_lpb_overrides(
        dataset_path=merged.get("task.dataset_path"),
        train_filter_key=merged.get("task.train_filter_key"),
        num_epochs=merged.get("training.num_epochs"),
        resume_ckpt=resume_ckpt,                       # path kwarg, not the bool flag
        log_dir=merged.get("hydra.run.dir"),
        device=merged.get("training.device"),
        # everything else (including an explicit "training.resume" override)
        # flows through `extra` verbatim -- last-one-wins under hydra.
        extra={k: v for k, v in merged.items()
               if k not in {"task.dataset_path", "task.train_filter_key",
                            "training.num_epochs", "training.device",
                            "hydra.run.dir"}},
    )

    abs_log_dir = os.path.abspath(
        log_dir or merged.get("hydra.run.dir")
        or os.path.join("logs", config_name, "run"))
    cmd = [
        "python", LPB_TRAIN_PY,
        "--config-path", os.path.abspath(config_dir),
        "--config-name", config_name,
    ] + overrides

    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices

    if dry_run:
        print(f"[train_base_dp] DRY-RUN cmd: {' '.join(cmd)}")
        return _latest_ckpt_path(abs_log_dir)

    os.makedirs(abs_log_dir, exist_ok=True)
    print(f"[train_base_dp] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=REPO_ROOT, env=env)
    return _latest_ckpt_path(abs_log_dir)


def _latest_ckpt_path(log_dir: str) -> str:
    """Canonical ckpt path produced by the LPB workspace (``save_last_ckpt``)."""
    return os.path.join(log_dir, "checkpoints", "latest.ckpt")


if __name__ == "__main__":
    # Manual E0 entry: python -m scout.train_base_dp [--config-name ...] [overrides ...]
    # Most users will invoke `python train.py ...` directly; this is a thin
    # convenience that forwards unknown args as hydra overrides.
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config-name", default=DEFAULT_CONFIG_NAME)
    p.add_argument("--config-dir", default=DEFAULT_CONFIG_DIR)
    p.add_argument("--log-dir", default=None)
    p.add_argument("--num-epochs", type=int, default=None)
    p.add_argument("--dataset-path", default=None)
    p.add_argument("--train-filter-key", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("overrides", nargs="*",
                   help="extra hydra overrides (key=value)")
    args = p.parse_args()

    extra = {}
    for kv in args.overrides:
        if "=" in kv:
            k, v = kv.split("=", 1)
            extra[k] = v

    ckpt = train(
        config_name=args.config_name, config_dir=args.config_dir,
        log_dir=args.log_dir, num_epochs=args.num_epochs,
        dataset_path=args.dataset_path,
        train_filter_key=args.train_filter_key,
        dry_run=args.dry_run, extra_overrides=extra,
    )
    print(f"[train_base_dp] ckpt -> {ckpt}")
