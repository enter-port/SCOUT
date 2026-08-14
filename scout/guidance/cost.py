"""SCOUT classifier-guidance cost (scout_design.md §4).

Replaces LPB's nearest-neighbour-to-demo latent cost with the VIB encoder's
**negative log-likelihood of the sampled skill latent**. The encoder outputs a
full diagonal Gaussian ``q_θ(z|s̄_t,a) = N(μ, diag(σ²))``; the cost is the
distribution-vs-sample NLL (user decision 2026-08-14, option A) -- NOT a
reparam-sampled L2 (the earlier form) and NOT the mean-only gap:

    cost(x0_hat, s_bar_t, z) = mean_B [ -log q_θ(z | s̄_t, a_chunk) ]
                             = mean_B ½ Σ_i [ (z_i - μ_i)² / σ_i² + log σ_i² ]
    a_chunk = bridge(x0_hat[:, :n_steps]).flatten()
    (the ½·D·log 2π constant is dropped: no gradient, irrelevant for guidance)

where ``x0_hat`` is the base DP's one-step clean-action estimate (the
``pred_original_sample`` of the diffusion scheduler step), ``s_bar_t = E_s(S_t)``
is the encoded current observation (held fixed across the chunk), and ``z`` is
the sampled skill latent (held fixed across the chunk). ``a_chunk`` is the
flattened first ``n_steps`` per-step actions -- the SAME flattened fs-step chunk
the VIB encoder was trained on (``train_vib._slice_transition``), so inference
loads the trained weights (the encoder is a chunk-encoder, NOT per-step; building
it per-step mismatches the saved Linear, 1168 vs 1098). The guidance gradient
pushes the predicted chunk toward actions that re-encode to ``z``.

Why NLL (vs the previous reparam-sampled L2 ``‖z − (μ+σε)‖²``):
  * it is the native classifier-guidance form -- the injected gradient is
    ``∇_a[−log q_θ(z|s̄_t,a)]``, exactly ``∇_a log p_θ(z|s̄_t,a)`` up to sign;
  * closed form, no ε draw -> lower-variance gradient than the reparam estimator
    (whose expectation ``‖z−μ‖² + Σσ_i²`` differs from the NLL anyway);
  * BOTH conduction channels survive: the μ-channel is weighted per-dim by
    1/σ_i² and the σ-channel carries ``[(z_i−μ_i)²/σ_i³ − 1/σ_i]·∂σ_i/∂a``.
    A mean-only gap would leave just ``∂μ/∂a``, which the KL term drives to 0
    as z→N(0,I) -- silently killing guidance; the σ-channel survives KL, keeping
    guidance alive precisely when z~N(0,I) sampling is most meaningful.

Shapes:
    x0_hat : (B, T, per_step)   -- DP-predicted clean action chunk (T = horizon)
    s_bar_t: (B, s_bar_dim)     -- one encoded obs per batch element
    z      : (B, style_dim)     -- one sampled skill latent per batch element
    -> scalar (differentiable in x0_hat)
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
    """``mean over batch of -log q_θ(z | s_bar_t, a_chunk)`` -- scalar,
    differentiable in ``x0_hat`` (Gaussian NLL, closed form, no ε sampling).

    ``a_chunk`` is the **flattened first ``n_steps`` per-step actions** of the
    DP's clean-action estimate ``x0_hat``, where ``n_steps =
    vib_enc.action_dim // per_step``. This matches ``train_vib._slice_transition``
    (the VIB encoder was trained on the flattened fs-step action chunk, NOT
    per-step) -- so inference feeds the same flattened chunk, else the saved
    encoder weights won't load (s_bar + per_step != s_bar + chunk). One NLL per
    batch element (one skill z per chunk; design §1).

    Args:
        x0_hat  : ``(B, T, per_step)`` -- base DP clean-action estimate (T =
                  horizon); must carry gradient to ``trajectory`` (caller sets
                  ``trajectory.requires_grad_()`` before the scheduler step that
                  produces this).
        s_bar_t : ``(B, s_bar_dim)`` -- encoded current obs (fixed across chunk).
        z       : ``(B, style_dim)`` -- sampled skill latent (fixed across chunk).
        vib_enc : a :class:`scout.model.vib.VIBEncoder` (or compatible) callable
                  ``(s_bar, a) -> (mu, logvar)`` with an ``.action_dim`` attr =
                  the flattened chunk dim it was trained on. The FULL Gaussian
                  (μ and σ) enters the NLL; neither is sampled here.
        bridge  : :class:`scout.normalizer.ActionNormalizerBridge` mapping
                  ``x0_hat`` (per-step) into the VIB action space.

    Returns:
        scalar tensor (mean-reduced over batch). Differentiable w.r.t. ``x0_hat``
        (the affine bridge + VIB MLP path preserves gradient; both the 1/σ²
        -weighted μ-channel and the σ-channel conduct it).
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

    # 2. Gaussian NLL of the fixed guidance target z under the encoder's
    #    q_θ(z|s̄_t, a_chunk) = N(μ, diag σ²):  ½ Σ_i [(z_i−μ_i)²·e^{−logvar_i}
    #    + logvar_i]. The ½D·log2π constant is dropped (no gradient). Both the
    #    1/σ²-weighted μ-gap and the σ-terms conduct the gradient (see module
    #    docstring).
    mu, logvar = vib_enc(s_bar_t.detach(), a_flat)  # (B, style_dim) each
    diff = z.detach() - mu
    inv_var = torch.exp(-logvar)
    nll = 0.5 * (diff.pow(2) * inv_var + logvar).sum(dim=-1)   # (B,)
    return nll.mean()
