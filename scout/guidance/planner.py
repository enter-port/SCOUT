"""SCOUT planner: SCOUT cost plugged into LPB's guided-denoising interface
(scout_design.md §4).

Aligns with the LPB ``dyn_model/planner.py:Planner.compute_loss`` interface --
``compute_loss(x0_hat, current_obs) -> scalar`` -- so it can be driven by
:class:`scout.guidance.policy.ScoutPolicy` (an LPB
``DiffusionUnetHybridImagePolicy`` subclass) inside its overridden
``guided_conditional_sample``. The cost itself is SCOUT's VIB re-encoding gap,
delegated to :func:`scout.guidance.cost.scout_cost` (NOT duplicated, per task
step 3):

    cost(x̂_0, s) = mean_t ‖z − z_θ(s̄_t, a_t)‖²,  z_θ = reparam(vib_enc(s̄_t,a)),  a = bridge(x̂_0)

where ``s̄_t = E_s(current_obs)`` is held fixed across the chunk and ``z`` is
the sampled skill latent held fixed across the whole denoise loop (design §1,
§4: "z 整段定住", "s̄_t 定住").

Normalization bridge conclusion (design §4, §7 risk #4; task step 1):
  - The LPB base DP (``DiffusionUnetHybridImagePolicy``) **does** carry a fitted
    ``self.normalizer['action']`` (``LinearNormalizer``, see
    ``diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:172,208``),
    so the DP's predicted ``x̂_0`` lives in **DP-normalised** action space.
  - The SCOUT VIB encoder (per ``scout/train_vib.py:_slice_transition``)
    consumes **raw** hdf5 actions (no normalizer in ``RobomimicImageDynamicsModelDataset``
    path that feeds VIB training).
  - => the two spaces differ once the LPB base DP is wired for real. The bridge
    must then **unnormalize** the DP output back to raw:
    ``bridge = NormalizerBridge(dp.normalizer['action'], PassthroughNormalizer())``
    (or simply ``lambda x: dp.normalizer['action'].unnormalize(x)``).
  - For the hermetic dummy verify (no LPB stack, no fitted normalizer) both
    spaces are equal and :class:`~scout.normalizer.IdentityBridge` is correct.

This module never imports the LPB diffusion_policy stack (the policy does, with
a try-except); it depends only on :mod:`scout.model` + :mod:`scout.normalizer`,
so it imports cleanly in environments without robomimic.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch

from scout.guidance.cost import scout_cost
from scout.normalizer import ActionNormalizerBridge, IdentityBridge


class ScoutPlanner:
    """SCOUT cost wrapper driven inside ``guided_conditional_sample``.

    Carries three pieces of per-inference-call state, all of which the policy
    fixes once before the denoise loop:

      - the frozen :class:`~scout.model.scout_vib.ScoutVIB` (``E_s`` + VIB
        encoder) -- ``scout_vib``;
      - the action-space ``bridge`` (DP-space -> VIB-space);
      - the fixed skill latent ``z`` of shape ``(B, style_dim)`` and the cached
        encoded obs ``s̄_t`` of shape ``(B, s_bar_dim)``.

    ``compute_loss`` matches the LPB ``Planner.compute_loss(sample, current_obs)``
    signature so the same call-site in ``guided_conditional_sample`` works.
    """

    def __init__(
        self,
        scout_vib,
        bridge: Optional[ActionNormalizerBridge] = None,
        z: Optional[torch.Tensor] = None,
        obs_adapter: Optional[Callable] = None,
    ):
        """
        Args:
            scout_vib    : a :class:`~scout.model.scout_vib.ScoutVIB`. Used for
                           ``E_s`` (encode current obs -> ``s̄_t``) and
                           ``vib_enc`` (compute z_θ = reparam sample, p_θ(s̄_t,a)); put in ``eval()`` mode.
            bridge       : :class:`~scout.normalizer.ActionNormalizerBridge`
                           mapping the DP's ``x̂_0`` into the VIB action space.
                           ``None`` -> :class:`IdentityBridge` (dummy / when the
                           two spaces coincide).
            z            : optional ``(B, style_dim)`` skill latent, fixed across
                           the chunk. ``None`` -> the policy sets it per
                           inference call via :meth:`set_z` (preferred) or it is
                           sampled lazily on the first :meth:`compute_loss` call.
            obs_adapter  : optional callable converting the policy's
                           ``current_obs`` (LPB raw keyed obs_dict) into E_s
                           format ``{"visual": {view: ...}, "proprio": ...}``.
                           ``None`` -> ``current_obs`` is assumed already in E_s
                           format (dummy verify path).
        """
        self.scout_vib = scout_vib
        self.scout_vib.eval()
        self.bridge = bridge if bridge is not None else IdentityBridge()
        self.z = z
        self.obs_adapter = obs_adapter
        # per-call caches (set by ScoutPolicy before the loop):
        self._cached_s_bar_t: Optional[torch.Tensor] = None
        self._cached_obs_id: Optional[int] = None

    # ------------------------------------------------------------------ #
    # per-inference-call state (set by ScoutPolicy before the denoise loop)
    # ------------------------------------------------------------------ #
    def set_z(self, z: Optional[torch.Tensor]):
        """Fix the skill latent for one inference call. ``(B, style_dim)`` or
        ``None`` (the policy samples fresh per call -- design §1)."""
        self.z = z

    def set_current_obs(self, current_obs):
        """Pre-encode ``s̄_t = E_s(current_obs)`` once, cache for the whole
        denoise loop (design §4: "s̄_t 定住"). ``current_obs`` must already be in
        E_s format unless an ``obs_adapter`` was given at construction."""
        obs_es = (
            self.obs_adapter(current_obs)
            if self.obs_adapter is not None
            else current_obs
        )
        with torch.no_grad():
            self._cached_s_bar_t = self.scout_vib.encode(obs_es)
        self._cached_obs_id = id(current_obs)

    def reset(self):
        """Clear cached ``s̄_t`` / ``z`` -- call between inference calls when the
        planner instance is reused across rollouts."""
        self.z = None
        self._cached_s_bar_t = None
        self._cached_obs_id = None

    # ------------------------------------------------------------------ #
    # LPB-parity interface
    # ------------------------------------------------------------------ #
    def compute_loss(
        self,
        x0_hat: torch.Tensor,
        current_obs=None,
    ) -> torch.Tensor:
        """``mean_t ‖z − z_θ(s̄_t, a_t)‖²`` (z_θ = reparam sample, p_θ(s̄_t,a)) -- scalar, differentiable in ``x0_hat``.

        Args:
            x0_hat       : ``(B, T, action_dim)`` -- the DP's one-step
                           clean-action estimate (``scheduler.step(...).
                           pred_original_sample``). Carries the graph back to
                           the trajectory.
            current_obs  : the policy's current obs (E_s format, or whatever
                           ``obs_adapter`` expects). When it is the same object
                           previously passed to :meth:`set_current_obs`, the
                           cached ``s̄_t`` is reused (saves a ResNet forward per
                           denoise step -- design §4 "s̄_t 定住").

        Returns:
            scalar tensor (mean-reduced over batch and chunk). Differentiable
            w.r.t. ``x0_hat`` (identity bridge + VIB MLP path preserves grad).
        """
        if x0_hat.dim() != 3:
            raise ValueError(
                f"x0_hat must be (B, T, action_dim); got {tuple(x0_hat.shape)}"
            )

        # s̄_t -- cached across the loop, encoded once.
        s_bar_t = self._resolve_s_bar_t(current_obs)

        # z -- fixed across the loop; sample lazily if the policy didn't set it.
        if self.z is None:
            self.z = torch.randn(
                s_bar_t.shape[0],
                self.scout_vib.style_dim,
                device=s_bar_t.device,
                dtype=s_bar_t.dtype,
            )

        # SCOUT cost (delegated to scout_cost -- not duplicated).
        return scout_cost(
            x0_hat=x0_hat,
            s_bar_t=s_bar_t.detach(),
            z=self.z.detach(),
            vib_enc=self.scout_vib.vib_enc,
            bridge=self.bridge,
        )

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _resolve_s_bar_t(self, current_obs) -> torch.Tensor:
        """Return cached ``s̄_t`` if the caller is on the same inference call
        (same obs object), else (re-)encode and update the cache."""
        if (
            current_obs is not None
            and id(current_obs) == self._cached_obs_id
            and self._cached_s_bar_t is not None
        ):
            return self._cached_s_bar_t

        if current_obs is None:
            if self._cached_s_bar_t is None:
                raise RuntimeError(
                    "ScoutPlanner.compute_loss needs current_obs on first call"
                    " (or call set_current_obs before the denoise loop)."
                )
            return self._cached_s_bar_t

        # new obs object -> encode + cache.
        obs_es = (
            self.obs_adapter(current_obs)
            if self.obs_adapter is not None
            else current_obs
        )
        with torch.no_grad():
            s_bar_t = self.scout_vib.encode(obs_es)
        self._cached_s_bar_t = s_bar_t
        self._cached_obs_id = id(current_obs)
        return s_bar_t

    def __repr__(self):
        return (
            f"ScoutPlanner(scout_vib={type(self.scout_vib).__name__}, "
            f"bridge={self.bridge!r}, style_dim={self.scout_vib.style_dim})"
        )
