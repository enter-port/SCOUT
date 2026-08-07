"""SCOUT classifier-guidance cost (scout_design.md §4).

Replaces LPB's nearest-neighbour-to-demo latent cost with the VIB re-encoding
gap:

    cost(x̂_0, s̄_t, z) = mean_t ‖z − μ(s̄_t, a_t)‖²,    a = bridge(x̂_0)

where ``x̂_0`` is the base DP's one-step clean-action estimate (the ``pred_original_sample``
of the diffusion scheduler step), ``s̄_t = E_s(S_t)`` is the encoded current
observation (held fixed across the chunk), and ``z`` is the sampled skill latent
(held fixed across the chunk). The guidance gradient pushes ``x̂_0`` toward
actions that re-encode to the sampled ``z``.

Shapes:
    x0_hat : (B, T_chunk, action_dim)  -- DP-predicted clean action chunk
    s_bar_t: (B, s_latent_dim)         -- one encoded obs per batch element
    z      : (B, style_dim)            -- one sampled skill latent per batch element

The VIB encoder is applied *per chunk step* after broadcasting ``s̄_t`` across
the chunk (same obs for every step in the chunk -- matches SOE alignment where a
single obs conditions a whole action chunk). Output is a scalar differentiable
in ``x0_hat``.
"""

from __future__ import annotations

import torch

from scout.normalizer import ActionNormalizerBridge


def scout_cost(
    x0_hat: torch.Tensor,
    s_bar_t: torch.Tensor,
    z: torch.Tensor,
    vib_enc,
    bridge: ActionNormalizerBridge,
) -> torch.Tensor:
    """``mean over (batch, chunk-step) of ‖z − μ(s̄_t, a_t)‖²``.

    Args:
        x0_hat  : ``(B, T, action_dim)`` -- base DP clean-action estimate; must
                  carry gradient to ``trajectory`` (caller sets
                  ``trajectory.requires_grad_()`` before the scheduler step that
                  produces this).
        s_bar_t : ``(B, s_latent_dim)`` -- encoded current obs (fixed across
                  chunk).
        z       : ``(B, style_dim)`` -- sampled skill latent (fixed across chunk).
        vib_enc : a :class:`scout.model.vib.VIBEncoder` (or compatible) callable
                  ``(s_bar, a) -> (mu, logvar)``; only ``mu`` is used.
        bridge  : :class:`scout.normalizer.ActionNormalizerBridge` mapping
                  ``x0_hat`` into the VIB action space.

    Returns:
        scalar tensor (mean-reduced over batch and chunk). Differentiable w.r.t.
        ``x0_hat`` (the identity bridge + VIB MLP path preserves gradient).
    """
    if x0_hat.dim() != 3:
        raise ValueError(
            f"x0_hat must be (B, T, action_dim), got shape {tuple(x0_hat.shape)}"
        )
    B, T, _ = x0_hat.shape

    # 1. bring x0_hat into the VIB action space (identity in stage-1).
    a = bridge(x0_hat)                                  # (B, T, action_dim)

    # 2. broadcast s_bar_t across the chunk, flatten chunk into batch for one
    #    vectorised VIBEncoder call.
    s_bar_exp = s_bar_t.unsqueeze(1).expand(B, T, -1).reshape(B * T, -1)
    a_flat = a.reshape(B * T, -1)
    mu, _ = vib_enc(s_bar_exp, a_flat)                  # (B*T, style_dim)
    mu = mu.reshape(B, T, -1)                           # (B, T, style_dim)

    # 3. z is fixed across the chunk -> broadcast and compare per step.
    z_exp = z.unsqueeze(1).expand(B, T, -1)             # (B, T, style_dim)
    diff = z_exp - mu
    return (diff.pow(2)).sum(dim=-1).mean()
