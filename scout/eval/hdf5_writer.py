"""Augmented-HDF5 writer: append successful rollouts to a core robomimic hdf5.

Extracted from ``self_improvement.py`` (original ``_write_augmented_hdf5``) so
both :class:`RolloutCollector` (rollout-only hdf5 output) and the
self-improvement retrain path share one writer.

Mirrors robomimic's ``data/demo_N`` schema: per-demo ``obs/<key>`` (T, dim),
``actions`` (T, action_dim), ``done``/``success`` (T,) bool, ``states`` (T, D)
when available. Also writes ``mask/<aug_mask_key>`` = boolean over ALL
``data/`` demos selecting ``core_filter_key`` demos + the appended rollout
demos, so a downstream retrain picks up both via one mask (otherwise it would
train on core-only and silently ignore the new rollouts -- the bug this avoids).

.. warning:: UNTESTED against the real robomimic loader (env deferred). The
   schema is faithful to SOE's write path; if a future real run finds a missing
   attribute (e.g. ``model_file``, ``env`` metadata, per-demo ``num_samples``),
   extend here.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np


def _demo_list(hdf5_file, mask_key: Optional[str]) -> List[str]:
    """Sorted ``data/demo*`` names, optionally filtered by ``mask/<mask_key>``."""
    all_demos = sorted([k for k in hdf5_file["data"].keys() if k.startswith("demo")])
    if mask_key is None or f"mask/{mask_key}" not in hdf5_file:
        return all_demos
    node = hdf5_file[f"mask/{mask_key}"]
    if isinstance(node, h5py_group_class()) and "mask" in node:
        arr = node["mask"][()]
    else:
        arr = node[()]
    if arr.dtype == bool:
        return [d for d, keep in zip(all_demos, arr) if keep]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


def _discover_obs_keys(data_grp, demo_name: str) -> List[str]:
    """Per-key obs keys present in ``data/<demo>/obs`` (robomimic schema)."""
    return sorted(data_grp[demo_name]["obs"].keys())


def h5py_group_class():
    """Lazy h5py import (module top-level imports stay clean)."""
    import h5py
    return h5py.Group


def write_rollouts_to_hdf5(core_path: str, out_path: str,
                           rollouts: List[dict],
                           core_filter_key: str = "train",
                           aug_mask_key: str = "scout_aug",
                           include_core: bool = True):
    """Write ``core_path``'s filtered demos + ``rollouts`` as a new HDF5.

    Starts from a copy of ``core_path`` (preserves env metadata / attrs / mask
    groups), appends one ``data/demo_<id>`` group per rollout, then writes the
    ``mask/<aug_mask_key>`` boolean over all demos (core filtered + new).

    Parameters
    ----------
    core_path : str
        Source robomimic hdf5 (its ``mask/<core_filter_key>`` selects the core
        demos that are kept + marked True in the augmented mask).
    out_path : str
        Destination hdf5 path (overwritten if it exists).
    rollouts : List[dict]
        Successful rollout trajectories. Each MUST carry ``obs``/``next_obs``
        per-frame lists (i.e. collected with ``record_obs=True``), ``actions``,
        ``dones``, and ``horizon``. ``states``/``success`` are optional.
    core_filter_key : str
        Mask key in ``core_path`` selecting the core demos (default ``"train"``;
        real runs use ``"core_20"`` etc.).
    aug_mask_key : str
        Mask key written to ``out_path`` selecting core + new rollouts (default
        ``"scout_aug"``); a retrain reads this key to train on both.
    include_core : bool
        True (default) -> core filtered demos + rollouts (the retraining file).
        False -> drop all core demos from ``data/`` (env metadata / mask groups
        copied from core are kept so the file stays robomimic-loadable) and
        write only the rollouts; the mask then selects just those rollouts
        (the success-only archive, ``{task}_success_exp{N}.hdf5``).
    """
    import h5py
    import shutil

    # start from a copy of core (preserves env metadata / attrs / mask groups)
    shutil.copyfile(core_path, out_path)

    with h5py.File(out_path, "r+") as f:
        core_demos = _demo_list(f, core_filter_key)
        if not core_demos:
            raise RuntimeError(
                f"no core demos under mask='{core_filter_key}' in {core_path}")
        core_obs_keys = set(_discover_obs_keys(f["data"], core_demos[0]))
        # rollout obs carries only the policy's shape_meta keys (use_object_obs=
        # False drops 'object' etc.); write the intersection so appended demos
        # match what the DP loads. Core demos keep their full key set.
        rollout_keys = set()
        for _r in rollouts:
            _ro = _r.get("obs") or []
            if _ro:
                rollout_keys |= set(_ro[0].keys())
        obs_keys = sorted((core_obs_keys & rollout_keys) if rollout_keys
                          else core_obs_keys)
        if include_core:
            core_set = set(core_demos)
        else:
            # success-only file: drop all original core demos (env metadata /
            # mask groups copied from core stay so the file is still
            # robomimic-loadable); only the rollouts are written below.
            for d in [k for k in list(f["data"].keys()) if k.startswith("demo")]:
                del f["data"][d]
            core_set = set()

        # find next free demo id
        existing_ids = [int(d.split("_")[-1]) for d in f["data"].keys()
                        if d.startswith("demo_") and d.split("_")[-1].isdigit()]
        next_id = (max(existing_ids) + 1) if existing_ids else 0

        # abs_action: rollout actions are the policy's 10-dim rot_6d output; the
        # core hdf5 stores 7-dim axis-angle (the loader re-converts to 6d via
        # abs_action=true). Transform back so the augmented hdf5 is consistent.
        from diffusion_policy.model.common.rotation_transformer import (
            RotationTransformer,)
        rot = RotationTransformer("axis_angle", "rotation_6d")  # .inverse = 6d->aa

        def _last_frame(o_k) -> np.ndarray:
            """Last frame of an n_obs_steps windowed obs value -> current frame."""
            return np.asarray(o_k, dtype=np.float32)[-1]

        def _to_storage(k: str, frame) -> np.ndarray:
            """rollout frame -> core hdf5 storage layout.

            rollout image obs is (C,H,W) float (robomimic CHW); core stores
            (H,W,C) uint8. low_dim obs is (D,) float either way.
            """
            if k.endswith("image"):
                img = np.asarray(frame, dtype=np.float32)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))     # CHW -> HWC
                return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            return np.asarray(frame, dtype=np.float32)

        new_demo_names: List[str] = []
        for rollout in rollouts:
            demo_name = f"demo_{next_id}"
            next_id += 1
            ep_len = int(rollout.get("horizon", 0))
            if ep_len == 0:
                continue
            grp = f["data"].create_group(demo_name)
            obs_list = rollout.get("obs") or []
            next_obs_list = rollout.get("next_obs") or []
            if len(obs_list) < ep_len:
                raise ValueError(
                    f"rollout for {demo_name} missing obs (need record_obs=True); "
                    f"got {len(obs_list)} frames, need {ep_len}")
            obs_grp = grp.create_group("obs")
            for k in obs_keys:
                obs_grp.create_dataset(
                    k, data=np.stack([_to_storage(k, _last_frame(o[k]))
                                      for o in obs_list[:ep_len]], axis=0))
            if next_obs_list:
                next_grp = grp.create_group("next_obs")
                for k in obs_keys:
                    next_grp.create_dataset(
                        k, data=np.stack([_to_storage(k, _last_frame(o[k]))
                                          for o in next_obs_list[:ep_len]], axis=0))
            # actions: 10-dim rot_6d -> 7-dim axis-angle (matches core storage).
            acts = np.asarray(rollout["actions"], dtype=np.float32)   # (T,10)
            if acts.shape[-1] == 10:
                pos = acts[..., :3]
                rot_aa = np.asarray(rot.inverse(acts[..., 3:9]))
                grip = acts[..., 9:10]
                acts = np.concatenate([pos, rot_aa, grip], axis=-1)   # (T,7)
            grp.create_dataset("actions", data=acts)
            # abs_actions: same 7-dim absolute aa (the policy emits absolute
            # actions; the loader reads THIS key for training when abs_action=
            # true, and reads demo['actions'] only for episode length).
            grp.create_dataset("abs_actions", data=acts)
            grp.create_dataset("done",
                               data=np.asarray(rollout["dones"], dtype=bool))
            grp.create_dataset("success",
                               data=np.full(ep_len, bool(rollout.get("success", True)),
                                            dtype=bool))
            # states (optional -- write if the rollout recorded them non-empty)
            states = rollout.get("states")
            if states is not None and len(states) >= ep_len:
                grp.create_dataset("states",
                                   data=np.asarray(states[:ep_len], dtype=np.float32))
            grp.attrs["num_samples"] = ep_len
            new_demo_names.append(demo_name)

        # write the augmented mask: True for core_<filter> demos + new rollouts.
        all_demos_after = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        new_set = set(new_demo_names)
        mask = np.array([d in core_set or d in new_set for d in all_demos_after],
                        dtype=bool)
        if f"mask/{aug_mask_key}" in f:
            del f[f"mask/{aug_mask_key}"]
        aug_grp = f.create_group(f"mask/{aug_mask_key}")
        aug_grp.create_dataset("mask", data=mask)
        aug_grp.attrs["num"] = int(mask.sum())
        f.attrs["num_demos_added"] = len(new_demo_names)
