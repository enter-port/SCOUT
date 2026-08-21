"""Expert z-bank guidance: pull rollout actions toward the expert (core) data.

User request 2026-08-21 -- a ROLLOUT-time guidance mode that biases actions
toward the expert data (as opposed to SCOUT's exploration mode, which samples
z from the prior to push AWAY from it):

  1. **bank**  :func:`build_expert_z_bank` encodes every expert
     ``(s_t, a-chunk_t)`` transition of the core hdf5 with the frozen VIB
     encoder and stores the per-transition means ``mu_e``. This is LPB's demo
     latent bank (``dyn_model/planner.py:get_demo_latents`` +
     ``compute_nn_reward``) ported to SCOUT's encoder -- with the key
     difference that SCOUT's z is action-conditioned (LPB's bank is obs-only
     and deterministic).
  2. **select** :meth:`ScoutExpertPlanner.select_z` -- at the START of every
     action generation (every denoise loop = every chunk; NOT once per
     trajectory like exploration mode) compute the query distribution
     ``q(z | s_bar_t, x0_hat)`` from the current obs and the half-denoised
     action estimate, then pick the bank entry with the lowest Gaussian NLL
     under the query: a Mahalanobis distance weighted by the query's own
     ``1/sigma^2`` (the log-variance term is constant across bank entries, so
     it drops out of the argmin). Selecting by the SAME metric the guidance
     cost minimizes makes z* the expert transition the current chunk already
     encodes closest to.
  3. **guide** :meth:`ScoutPolicy.guided_conditional_sample` denoises the
     chunk with the existing SCOUT NLL cost (:func:`scout.guidance.cost.
     scout_cost`) at ``z = z*`` fixed for that chunk, pulling ``x0`` toward
     actions that re-encode to the nearest expert latent.

Wiring contract: the planner's ``select_z`` method IS the expert-mode flag.
The policy checks for it and (a) skips locking any prior-sampled z, (b) calls
``select_z(x0_hat, current_obs)`` once at the first guided denoise step of
every chunk. The vec runner checks for it and skips its per-rollout prior z
draw (see :mod:`scout.eval.rollout_vec`). Exploration mode (plain
``ScoutPlanner``) has no ``select_z`` and is completely unaffected.
"""

from __future__ import annotations

from typing import Callable, Optional, Sequence

import numpy as np
import torch

from scout.guidance.planner import ScoutPlanner


def _default_aa_to_6d(actions: np.ndarray) -> np.ndarray:
    """7-dim axis-angle abs actions -> 10-dim rot-6d (LPB parity).

    Exactly what ``RobomimicReplayImageDataset`` does online
    (``_convert_actions``: split pos/rot/gripper, ``RotationTransformer``
    on the rot part only) -- i.e. the action space the VIB encoder was
    trained on. Lazy import: the LPB stack (robomimic etc.) is only
    guaranteed on the training/eval host.
    """
    from diffusion_policy.dataset.robomimic_replay_image_dataset import (
        _convert_actions,
    )
    from diffusion_policy.model.common.rotation_transformer import (
        RotationTransformer,
    )

    rt = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    return _convert_actions(np.asarray(actions, dtype=np.float64), True, rt)


def build_expert_z_bank(
    core_hdf5: str,
    scout_vib,
    view_names: Sequence[str],
    proprio_keys: Sequence[str],
    device,
    stride: int = 1,
    batch: int = 64,
    img_scale: Optional[float] = 1.0 / 255.0,
    crop_size: Optional[int] = 76,
    action_key: str = "abs_actions",
    aa_to_6d: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> torch.Tensor:
    """Encode every expert transition of ``core_hdf5`` -> z-bank ``(N, style)``.

    Per demo, per t (stride-sampled): the encoder input is the SAME
    (s_bar_t, flattened fs-step chunk) pair VIB training used
    (``train_vib._slice_transition``), with obs preprocessed exactly like
    rollout-time inference -- :func:`scout.eval.rollout.make_obs_adapter`
    (/255 + center crop on CHW, proprio concat in ``proprio_keys`` order) --
    so bank mu's live in the same space as the query mu's.

    Args:
        core_hdf5 : path to the core hdf5 (obs HWC uint8 + ``abs_actions``
                    7-dim aa per step; the augmented-hdf5 "core format").
        scout_vib : a :class:`~scout.model.scout_vib.ScoutVIB` in eval mode
                    (carries ``E_s`` + ``vib_enc``; must match the rollout's
                    VIB ckpt).
        stride    : frame stride between consecutive bank transitions
                    (1 = every step, LPB parity).
        batch     : transitions encoded per forward pass.
        aa_to_6d  : 7d-aa -> 10d-6d converter; default = LPB's
                    ``_convert_actions`` (lazy import).

    Returns:
        ``(N, style_dim)`` float32 tensor of encoder means, on ``device``.
    """
    import h5py

    from scout.eval.rollout import make_obs_adapter

    view_names = list(view_names)
    proprio_keys = list(proprio_keys)
    if aa_to_6d is None:
        aa_to_6d = _default_aa_to_6d
    adapter = make_obs_adapter(view_names, proprio_keys,
                               img_scale=img_scale, crop_size=crop_size)
    vib_enc = scout_vib.vib_enc
    chunk_dim = int(vib_enc.action_dim)
    per_step = aa_to_6d(np.zeros((1, 7), dtype=np.float64)).shape[-1]  # 10
    if chunk_dim % per_step != 0:
        raise ValueError(
            f"vib_enc.action_dim={chunk_dim} not divisible by the 6d "
            f"per-step action dim {per_step}"
        )
    n_steps = chunk_dim // per_step

    scout_vib.eval()
    bank_rows: list[torch.Tensor] = []
    with h5py.File(core_hdf5, "r") as f:
        demos = sorted(k for k in f["data"].keys() if k.startswith("demo"))
        for di, demo in enumerate(demos):
            g = f[f"data/{demo}"]
            T = g[action_key].shape[0]
            if T < n_steps:
                continue
            imgs = {v: g[f"obs/{v}"][:] for v in view_names}       # (T,H,W,3)
            props = {k: g[f"obs/{k}"][:] for k in proprio_keys}
            acts = aa_to_6d(g[action_key][:]).astype(np.float32)    # (T,10)
            ts = list(range(0, T - n_steps + 1, stride))
            for s in range(0, len(ts), batch):
                tb = ts[s:s + batch]
                # LPB obs-dict layout per frame: per-key (B, 1, *shape)
                obs_dict = {
                    **{v: torch.from_numpy(
                        np.ascontiguousarray(imgs[v][tb])).permute(0, 3, 1, 2
                        ).float().unsqueeze(1) for v in view_names},
                    **{k: torch.from_numpy(
                        np.ascontiguousarray(props[k][tb])).float()
                        .unsqueeze(1) for k in proprio_keys},
                }
                obs_es = adapter(obs_dict)
                a_flat = np.stack([acts[t:t + n_steps].reshape(-1) for t in tb])
                a_flat = torch.from_numpy(a_flat).float().to(device)
                with torch.no_grad():
                    s_bar = scout_vib.encode(obs_es)
                    mu, _ = vib_enc(s_bar, a_flat)
                bank_rows.append(mu.detach().float().cpu())
            print(f"[expert_bank] {demo}: {len(ts)} transitions "
                  f"(T={T}, chunk={n_steps})")
    if not bank_rows:
        raise RuntimeError(f"no expert transitions encoded from {core_hdf5}")
    bank = torch.cat(bank_rows, dim=0).to(device)
    print(f"[expert_bank] total {tuple(bank.shape)} entries "
          f"from {len(demos)} demos of {core_hdf5}")
    return bank


class ScoutExpertPlanner(ScoutPlanner):
    """SCOUT planner with an expert z-bank: per-chunk nearest-expert guidance.

    ``select_z`` is the expert-mode hook consumed by
    :meth:`scout.guidance.policy.ScoutPolicy.guided_conditional_sample` (its
    presence switches the policy off the per-rollout prior z). Everything
    else -- ``compute_loss``, s_bar caching, bridge -- is inherited unchanged
    from :class:`~scout.guidance.planner.ScoutPlanner`.
    """

    def __init__(self, scout_vib, z_bank: torch.Tensor,
                 bridge=None, obs_adapter: Optional[Callable] = None,
                 bank_chunk: int = 4096):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         z=None)
        if z_bank.dim() != 2 or z_bank.shape[1] != scout_vib.style_dim:
            raise ValueError(
                f"z_bank must be (N, style_dim={scout_vib.style_dim}); "
                f"got {tuple(z_bank.shape)}"
            )
        if z_bank.shape[0] == 0:
            raise ValueError("z_bank is empty")
        self.z_bank = z_bank
        self.bank_chunk = int(bank_chunk)

    @torch.no_grad()
    def select_z(self, x0_hat: torch.Tensor, current_obs) -> torch.Tensor:
        """Pick the bank entry with the lowest NLL under the query
        ``q(z | s_bar_t, x0_hat)`` -- one z* per batch row.

        Args:
            x0_hat      : ``(B, T, per_step)`` DP-space clean-action estimate
                          (the first guided step's ``pred_original_sample``;
                          detached by the caller or here).
            current_obs : same object the policy passed to
                          ``set_current_obs`` -> cached ``s_bar_t`` reused.

        Returns:
            ``(B, style_dim)`` selected bank rows z* (detached).
        """
        if x0_hat.dim() != 3:
            raise ValueError(
                f"x0_hat must be (B, T, per_step); got {tuple(x0_hat.shape)}")
        s_bar_t = self._resolve_s_bar_t(current_obs)
        B, T, per_step = x0_hat.shape
        chunk_dim = int(self.scout_vib.vib_enc.action_dim)
        if chunk_dim % per_step != 0 or chunk_dim // per_step > T:
            raise ValueError(
                f"cannot slice a {chunk_dim}-dim chunk out of x0_hat "
                f"{tuple(x0_hat.shape)}")
        n_steps = chunk_dim // per_step
        a_flat = self.bridge(x0_hat.detach()[:, :n_steps]).reshape(B, chunk_dim)
        mu_q, logvar_q = self.scout_vib.vib_enc(s_bar_t.detach(), a_flat)
        inv_var = torch.exp(-logvar_q)                       # (B, D)

        # chunked Mahalanobis argmin over the bank (LPB compute_nn_reward
        # parity): d_ij = 0.5 * sum_d (mu_e_jd - mu_q_id)^2 / sigma_q_id^2.
        # The logvar term of the NLL is constant across j -> argmin-invariant.
        best_d = None
        best_i = None
        for s in range(0, self.z_bank.shape[0], self.bank_chunk):
            chunk = self.z_bank[s:s + self.bank_chunk]       # (C, D)
            diff = chunk.unsqueeze(0) - mu_q.unsqueeze(1)    # (B, C, D)
            d = 0.5 * (diff * diff * inv_var.unsqueeze(1)).sum(dim=-1)  # (B,C)
            v, i = d.min(dim=-1)
            if best_d is None:
                best_d, best_i = v, i + s
            else:
                m = v < best_d
                best_d[m] = v[m]
                best_i[m] = i[m] + s
        return self.z_bank[best_i].detach()
