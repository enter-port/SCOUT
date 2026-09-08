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


def get_aa_transformer():
    """Lazy RotationTransformer (axis_angle <- rotation_6d inverse)."""
    from diffusion_policy.model.common.rotation_transformer import (
        RotationTransformer,)
    return RotationTransformer("axis_angle", "rotation_6d")  # .inverse = 6d->aa


def _acts_to_storage(acts: np.ndarray, rot) -> np.ndarray:
    """policy 10|20-dim rot_6d actions -> 7|14-dim axis-angle (core storage).

    Single arm: 10 -> 7; dual arm (transport): 20 -> 14 via per-arm chunks,
    same layout as the training loader. Actions already in aa form pass
    through unchanged.
    """
    acts = np.asarray(acts, dtype=np.float32)   # (T,10|20)
    if acts.shape[-1] in (10, 20):
        n_arms = acts.shape[-1] // 10
        per_arm = acts.reshape(-1, n_arms, 10)
        pos = per_arm[..., :3]
        rot_aa = np.asarray(rot.inverse(per_arm[..., 3:9]))
        grip = per_arm[..., 9:10]
        acts = np.concatenate([pos, rot_aa, grip],
                              axis=-1).reshape(acts.shape[:-1] +
                                               (n_arms * 7,))
    return acts


def _stack_obs(obs_list, k: str, ep_len: int) -> np.ndarray:
    """obs_list[:ep_len] per-frame last-frame conversion -> (T, ...) stacked."""
    return np.stack([_to_storage(k, _last_frame(o[k]))
                     for o in obs_list[:ep_len]], axis=0)


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
        rot = get_aa_transformer()

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
                obs_grp.create_dataset(k, data=_stack_obs(obs_list, k, ep_len))
            if next_obs_list:
                next_grp = grp.create_group("next_obs")
                for k in obs_keys:
                    next_grp.create_dataset(
                        k, data=_stack_obs(next_obs_list, k, ep_len))
            # actions: rot_6d -> axis-angle (matches core storage).
            acts = _acts_to_storage(rollout["actions"], rot)
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


def merge_accumulated_hdf5(core_path: str, round_all_paths: List[str],
                           out_path: str, aug_mask_key: str = "scout_aug") -> dict:
    """dyn-retrain ACCUMULATED dataset: core + every round's trajectories.

    User rule (2026-08-15): round-N DP retrains on round-N successes only
    (``success.hdf5``), but the dyn/VIB retrain accumulates -- core demos plus
    the appended rollout demos of rounds 1..N (every trajectory, success AND
    failure). Each round's ``all.hdf5`` is itself a copy of the SAME core with
    that round's demos appended (ids >= ``n_core``, already in core storage
    format: HWC uint8 images, 7-dim axis-angle ``abs_actions``), so merging =
    copy the core once + cross-file-copy each round's appended demo groups
    (renumbered sequentially) + rebuild the augmented mask over
    core + all copied demos.

    ``core_path`` is the materialized core-only hdf5 (extract_core_subset.py
    output), so ALL of its demos count as core (no filter key needed).

    Returns a small dict ``{"rounds_merged", "demos_copied", "total_demos"}``.
    """
    import h5py
    import shutil

    def _did(k: str) -> int:
        return int(k.split("_")[-1])

    shutil.copyfile(core_path, out_path)
    with h5py.File(out_path, "r+") as f:
        core_names = [k for k in f["data"].keys() if k.startswith("demo")]
        if not core_names:
            raise RuntimeError(f"no demos in core file {core_path}")
        core_set = set(core_names)
        n_core = max(_did(k) for k in f["data"].keys()
                     if k.startswith("demo_") and k.split("_")[-1].isdigit()) + 1
        next_id = n_core
        copied = 0
        for rp in round_all_paths:
            with h5py.File(rp, "r") as rf:
                appended = sorted(
                    [k for k in rf["data"].keys()
                     if k.startswith("demo_") and k.split("_")[-1].isdigit()
                     and _did(k) >= n_core],
                    key=_did)
                for src in appended:
                    rf.copy(f"data/{src}", f["data"], name=f"demo_{next_id}")
                    next_id += 1
                    copied += 1
        all_demos_after = sorted([k for k in f["data"].keys()
                                  if k.startswith("demo")])
        mask = np.array(
            [d in core_set or (d.split("_")[-1].isdigit() and _did(d) >= n_core)
             for d in all_demos_after],
            dtype=bool)
        if f"mask/{aug_mask_key}" in f:
            del f[f"mask/{aug_mask_key}"]
        aug_grp = f.create_group(f"mask/{aug_mask_key}")
        aug_grp.create_dataset("mask", data=mask)
        aug_grp.attrs["num"] = int(mask.sum())
        f.attrs["num_demos_added"] = copied
        return {"rounds_merged": len(round_all_paths),
                "demos_copied": copied,
                "total_demos": len(all_demos_after)}
