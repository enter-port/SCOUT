"""Robomimic low_dim transition backend.

Reads a robomimic ``low_dim_v141.hdf5`` file, concatenates the (dynamically
discovered) low-dim obs keys into a single state vector per frame, and emits
``(S_t, A_t, S_{t+1})`` transitions into an internal :class:`ReplayBuffer`.

Mirrors SOE ``src/dataset/robomimic_v2.py`` conventions:
  * ``obs_keys = sorted(low-dim keys under data/<demo>/obs)`` -> ``S_t = concat``.
  * ``action_offset = 1``: transition ``i`` uses ``S_t=obs[i]``,
    ``A_t=actions[i+1]``, ``S_{t+1}=obs[i+1]``. ``A_t`` and ``S_{t+1}`` share
    frame index ``i+1`` -- i.e. ``A_t`` is the action that produced ``S_{t+1}``
    (SOE alignment).
  * ``mask/<key>`` filter (robomimic group+bool-dataset, bool-dataset, or
    demo-name-dataset -- all handled).

.. note:: **image interface point** (scout_design.md §6). A future image backend
   swaps ``S_t`` for an image / feature tensor and otherwise keeps this exact
   interface (``sample``/``add``/``__len__``/``stats``); the VIB / guidance /
   cost modules are modality-agnostic.
"""

import numpy as np
import h5py
import torch

from scout.data.transition_source import TransitionSource, ReplayBuffer


def _discover_obs_keys(data_grp, demo_name):
    """Sorted low-dim obs keys (per-frame 1-D vectors, i.e. dataset ndim==2)."""
    obs_grp = data_grp[demo_name]["obs"]
    keys = [k for k in obs_grp.keys() if obs_grp[k].ndim == 2]
    return sorted(keys)


def _demo_list(hdf5, mask_key):
    """All ``demo_*`` names under ``data/``, optionally filtered by ``mask/<key>``."""
    data_grp = hdf5["data"]
    all_demos = sorted([k for k in data_grp.keys() if k.startswith("demo")])
    if mask_key is None or f"mask/{mask_key}" not in hdf5:
        return all_demos
    node = hdf5[f"mask/{mask_key}"]
    # Case A: group with a boolean 'mask' dataset (standard robomimic).
    if isinstance(node, h5py.Group):
        if "mask" in node:
            arr = node["mask"][()]
            if arr.dtype == bool:
                return [d for d, keep in zip(all_demos, arr) if keep]
        return all_demos
    # Case B: dataset of demo names (bytes/str) -- SOE-style.
    arr = node[()]
    if arr.dtype == bool:  # boolean over demos
        return [d for d, keep in zip(all_demos, arr) if keep]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


class RobomimicLowdimSource(TransitionSource):
    def __init__(self, path, mask_key="train", action_offset=1, capacity=None):
        self.path = str(path)
        self.mask_key = mask_key
        self.action_offset = int(action_offset)

        with h5py.File(self.path, "r") as hdf5:
            demos = _demo_list(hdf5, mask_key)
            assert len(demos) > 0, f"no demos under mask='{mask_key}'"
            self.obs_keys = _discover_obs_keys(hdf5["data"], demos[0])
            assert len(self.obs_keys) > 0, "no low-dim obs keys found"
            self.state_dim = int(sum(
                hdf5["data"][demos[0]]["obs"][k].shape[1] for k in self.obs_keys
            ))
            self.action_dim = int(hdf5["data"][demos[0]]["actions"].shape[1])

            S_t_chunks, A_t_chunks, S_tp1_chunks = [], [], []
            for demo in demos:
                obs_grp = hdf5["data"][demo]["obs"]
                # (T, state_dim) -- concat sorted low-dim keys along feature axis
                s_all = np.concatenate(
                    [obs_grp[k][()] for k in self.obs_keys], axis=1
                ).astype(np.float32)
                acts = hdf5["data"][demo]["actions"][()].astype(np.float32)
                T = s_all.shape[0]
                off = self.action_offset
                assert T - off > 0, f"demo {demo} too short (T={T}, offset={off})"
                n = T - off
                S_t_chunks.append(s_all[:n])
                A_t_chunks.append(acts[off:off + n])
                S_tp1_chunks.append(s_all[off:off + n])

        S_t = np.concatenate(S_t_chunks, axis=0)
        A_t = np.concatenate(A_t_chunks, axis=0)
        S_tp1 = np.concatenate(S_tp1_chunks, axis=0)
        cap = len(S_t) if capacity is None else int(capacity)
        self._buffer = ReplayBuffer(self.state_dim, self.action_dim, capacity=cap)
        self._buffer.add({"S_t": S_t, "A_t": A_t, "S_tp1": S_tp1})

    # ---- TransitionSource delegation ----
    def sample(self, batch_size):
        return self._buffer.sample(batch_size)

    def add(self, transitions):
        """Online write-back (self-improvement loop). Wraps when capacity is full."""
        self._buffer.add(transitions)

    def stats(self):
        return self._buffer.stats()

    def __len__(self):
        return len(self._buffer)
