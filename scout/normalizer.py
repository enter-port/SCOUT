"""Action-space normalizer bridge: base-DP action -> VIB encoder action space.

SCOUT cost (``guidance/cost.py``) evaluates the VIB encoder ``μ(s̄_t, a)`` on the
action predicted by the base Diffusion Policy. The two modules may live in
**different action spaces**: the base DP (when ported with its full SOE/LPB
``normalizer['action']`` stack) predicts *DP-normalised* actions, while the VIB
encoder consumes whatever the transitions fed it during ``train_vib.py``
(stage-1 = **raw** actions straight from the robomimic hdf5).

This module is the single place that reconciles the two. It mirrors LPB
``dyn_model/planner.py:211-213`` exactly:

    init_actions_unnormalized = policy_action_normalizer.unnormalize(sample)
    init_actions              = dyn_model_normalizer['act'].normalize(init_actions_unnormalized)

i.e. ``DP-action -> unnormalize -> raw -> normalize -> VIB-action``.

.. note:: **Stage-1 finding (2026-08-07).** The Phase-2 port of SOE's ``DP`` to
   ``scout/policy/`` did **NOT** port ``normalizer.py``: ``DiffusionUNetPolicy``
   has no ``self.normalizer`` -- ``compute_loss`` consumes raw ``actions`` and
   ``conditional_sample`` returns raw ``sample[..., :Da]``. Phase-3 VIB training
   (``train_vib.py``) feeds raw ``A_t`` from :class:`RobomimicLowdimSource`. So
   in stage-1 **both spaces are raw and equal**, and the bridge collapses to
   :class:`IdentityBridge`. The bridge exists anyway as the documented seam:
   porting a real normalizer later is a one-line ``make_bridge`` swap, not a
   silent space-mismatch bug (scout_design.md §7 risk #4).
"""

from __future__ import annotations

from typing import Optional


class ActionNormalizerBridge:
    """Interface: map a base-DP predicted action chunk into the VIB action space.

    ``__call__`` must be differentiable (so ``autograd.grad(cost, x_t)`` flows
    through it back to the DP's ``x0_hat``). Both concrete bridges below satisfy
    this: identity is trivially differentiable; the normalizer bridge composes
    affine maps.
    """

    def __call__(self, x0_hat):  # pragma: no cover - interface
        raise NotImplementedError


class IdentityBridge(ActionNormalizerBridge):
    """No-op bridge (stage-1 default: DP and VIB both use raw actions)."""

    def __call__(self, x0_hat):
        return x0_hat

    def __repr__(self):
        return "IdentityBridge()"


class NormalizerBridge(ActionNormalizerBridge):
    """DP-normalized -> raw -> VIB-normalized (LPB ``planner.py:211-213``).

    Use this once the base DP is ported *with* its action normalizer (and once a
    VIB action normalizer is fit, e.g. from ``source.stats()['A_t']``). Both
    arguments must expose ``.normalize(x)`` and ``.unnormalize(x)`` (the standard
    ``diffusers``/robomimic normalizer API, e.g. ``GaussianNormalizer``).
    """

    def __init__(self, policy_action_normalizer, vib_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer
        self.vib_action_normalizer = vib_action_normalizer

    def __call__(self, x0_hat):
        a_raw = self.policy_action_normalizer.unnormalize(x0_hat)
        return self.vib_action_normalizer.normalize(a_raw)

    def __repr__(self):
        return (
            f"NormalizerBridge(policy={type(self.policy_action_normalizer).__name__}, "
            f"vib={type(self.vib_action_normalizer).__name__})"
        )


class UnnormalizeOnlyBridge(ActionNormalizerBridge):
    """DP-normalized -> raw only (seam ②; scout_design.md §4 "归一化桥").

    The LPB base DP (``DiffusionUnetHybridImagePolicy``) carries a fitted
    ``self.normalizer['action']``; its ``guided_conditional_sample`` operates on
    the **normalized** trajectory, so the one-step clean-action estimate
    ``x̂_0`` lives in DP-normalized space. The SCOUT VIB encoder was trained on
    **raw** hdf5 actions (``train_vib.py::_slice_transition`` feeds raw ``A_t``).
    => the cost must unnormalize ``x̂_0`` back to raw; the VIB side needs no
    further transform (raw == VIB space). This bridge is that single
    ``unnormalize`` call -- differentiable (affine), so ``autograd.grad(cost,
    x_t)`` flows straight through. LPB analogue:
    ``dyn_model/planner.py:211-213`` with a passthrough VIB normalizer.
    """

    def __init__(self, policy_action_normalizer):
        self.policy_action_normalizer = policy_action_normalizer

    def __call__(self, x0_hat):
        return self.policy_action_normalizer.unnormalize(x0_hat)

    def __repr__(self):
        return (
            f"UnnormalizeOnlyBridge(policy="
            f"{type(self.policy_action_normalizer).__name__})"
        )


def make_bridge(
    policy_action_normalizer: Optional[object] = None,
    vib_action_normalizer: Optional[object] = None,
) -> ActionNormalizerBridge:
    """Factory picking the right bridge for the (DP-space -> VIB-space) map.

    - neither given                           -> :class:`IdentityBridge`
      (both spaces equal; hermetic dummy / no-fitted-normalizer path).
    - ``policy_action_normalizer`` only       -> :class:`UnnormalizeOnlyBridge`
      (DP normalized -> raw; VIB consumes raw -- the LPB-wired stage-1 default).
    - both given                              -> :class:`NormalizerBridge`
      (DP normalized -> raw -> VIB normalized; full LPB ``planner.py:211-213``).
    """
    if policy_action_normalizer is None and vib_action_normalizer is None:
        return IdentityBridge()
    if policy_action_normalizer is not None and vib_action_normalizer is None:
        return UnnormalizeOnlyBridge(policy_action_normalizer)
    if policy_action_normalizer is not None and vib_action_normalizer is not None:
        return NormalizerBridge(policy_action_normalizer, vib_action_normalizer)
    # vib given but policy missing -> not a meaningful transform.
    raise ValueError(
        "vib_action_normalizer without policy_action_normalizer is not a valid "
        "bridge (need a source space to unnormalize from)."
    )
