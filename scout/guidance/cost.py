"""SCOUT classifier-guidance cost (scout_design.md §4; idea.md Cost definition).

Replaces LPB's nearest-neighbour-to-demo latent cost with the VIB re-encoding
gap. The cost target is the VIB encoder's **reparam-sampled output**
``z_θ = p_θ(s̄_t, a)`` (= ``reparam(μ, logvar)``), NOT the mean ``μ`` alone —
matching ``idea.md``'s ``Cost = ‖z − p̄_θ(z|S_t,A_t)‖`` (the earlier design-doc
draft incorrectly reduced this to ``μ``; corrected here):

    cost(x0_hat, s_bar_t, z) = mean_B ||z - z_θ(s_bar_t, a_chunk)||^2,
    a_chunk = bridge(x0_hat[:, :n_steps]).flatten(),
    z_θ     = reparam( vib_enc(s_bar_t, a_chunk) )      # p_θ(s̄_t, a)

where ``x0_hat`` is the base DP's one-step clean-action estimate (the
``pred_original_sample`` of the diffusion scheduler step), ``s_bar_t = E_s(S_t)``
is the encoded current observation (held fixed across the chunk), and ``z`` is
the sampled skill latent (held fixed across the chunk). ``a_chunk`` is the
flattened first ``n_steps`` per-step actions -- the SAME flattened fs-step chunk
the VIB encoder was trained on (``train_vib._slice_transition``), so inference
loads the trained weights (the encoder is a chunk-encoder, NOT per-step; building
it per-step mismatches the saved Linear, 1168 vs 1098). The guidance gradient
pushes the predicted chunk toward actions that re-encode to ``z``.

Why the reparam sample (not μ): the guidance gradient w.r.t. the action is
``∇_a ‖z − z_θ‖² = −2(z − z_θ)·(∂μ/∂a + ε·∂σ/∂a)`` — TWO conduction channels
(μ-sensitivity AND σ-sensitivity × ε). Using only μ would leave a single channel
``∂μ/∂a``, which the KL term drives toward 0 as z→N(0,I) — so a well-trained VIB
(μ≈0, σ≈1) would silently kill guidance. The σ-channel survives KL (σ can stay
input-sensitive even at σ̄→1), keeping guidance alive precisely when z~N(0,I)
sampling is most meaningful. This is the structural reason idea.md defines the
cost on the sampled encoder output, not the mean.

Shapes:
    x0_hat : (B, T, per_step)   -- DP-predicted clean action chunk (T = horizon)
    s_bar_t: (B, s_bar_dim)     -- one encoded obs per batch element
    z      : (B, style_dim)     -- one sampled skill latent per batch element
    -> scalar (differentiable in x0_hat)
"""

from __future__ import annotations

import torch

from scout.normalizer import ActionNormalizerBridge
from scout.model.vib import reparam


def scout_cost(
    x0_hat: torch.Tensor,
    s_bar_t: torch.Tensor,
    z: torch.Tensor,
    vib_enc,
    bridge: ActionNormalizerBridge,
) -> torch.Tensor:
    """``mean over batch of ||z - z_θ(s_bar_t, a_chunk)||^2`` -- scalar,
    differentiable in ``x0_hat``.

    ``a_chunk`` is the **flattened first ``n_steps`` per-step actions** of the
    DP's clean-action estimate ``x0_hat``, where ``n_steps =
    vib_enc.action_dim // per_step``. This matches ``train_vib._slice_transition``
    (the VIB encoder was trained on the flattened fs-step action chunk, NOT
    per-step) -- so inference feeds the same flattened chunk, else the saved
    encoder weights won't load (s_bar + per_step != s_bar + chunk). One sampled
    z_θ per batch element (one skill z per chunk; design §1).

    Args:
        x0_hat  : ``(B, T, per_step)`` -- base DP clean-action estimate (T =
                  horizon); must carry gradient to ``trajectory`` (caller sets
                  ``trajectory.requires_grad_()`` before the scheduler step that
                  produces this).
        s_bar_t : ``(B, s_bar_dim)`` -- encoded current obs (fixed across chunk).
        z       : ``(B, style_dim)`` -- sampled skill latent (fixed across chunk).
        vib_enc : a :class:`scout.model.vib.VIBEncoder` (or compatible) callable
                  ``(s_bar, a) -> (mu, logvar)`` with an ``.action_dim`` attr =
                  the flattened chunk dim it was trained on. The reparam sample
                  ``z_θ = reparam(mu, logvar)`` (= ``p_θ(s̄_t, a)``) is used as
                  the cost target, NOT ``mu`` alone (see module docstring).
        bridge  : :class:`scout.normalizer.ActionNormalizerBridge` mapping
                  ``x0_hat`` (per-step) into the VIB action space.

    Returns:
        scalar tensor (mean-reduced over batch). Differentiable w.r.t. ``x0_hat``
        (the affine bridge + VIB MLP + reparam path preserves gradient).
    """
    if x0_hat.dim() != 3:
        raise ValueError(
            f"x0_hat must be (B, T, per_step); got {tuple(x0_hat.shape)}"
        )
    B, T, per_step = x0_hat.shape
    chunk_dim = int(getattr(vib_enc, "action_dim", per_step))
    if chunk_dim % per_step != 0:
        raise ValueError(
            f"vib_enc.action_dim={chunk_dim} not divisible by the per-step "
            f"action dim {per_step} (encoder trained on a different chunking)."
        )
    n_steps = chunk_dim // per_step
    if n_steps > T:
        raise ValueError(
            f"horizon T={T} shorter than the trained action chunk "
            f"({chunk_dim} = {n_steps}x{per_step})."
        )

    # 1. bring the chunk into the VIB action space (unnormalize per-step), then
    #    flatten to the exact vector the encoder was trained on.
    a = bridge(x0_hat[:, :n_steps])                 # (B, n_steps, per_step)
    a_flat = a.reshape(B, chunk_dim)                # (B, chunk_dim)

    # 2. one sampled skill z_θ per batch element from (s̄_t, a_chunk) via the VIB
    #    encoder + reparam (= p_θ(s̄_t, a), the idea.md cost target); compare to
    #    the fixed guidance target z. Both μ- and σ-sensitivity conduct the
    #    gradient (see module docstring).
    mu, logvar = vib_enc(s_bar_t.detach(), a_flat)  # (B, style_dim) each
    z_enc = reparam(mu, logvar)                      # p_θ(s̄_t, a) = sampled z
    diff = z.detach() - z_enc
    return diff.pow(2).sum(dim=-1).mean()
