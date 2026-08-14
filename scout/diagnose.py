"""VIB life/death diagnostics (scout_design.md §5, §7 risk #1).

The single number that tells whether guidance can work at all:
``sensitivity = ‖∂z_θ/∂a‖ · σ_a / σ_z`` (z_θ = reparam sample = p_θ(s̄_t,a)).
If z_θ does not respond to actions, the guidance gradient
``∇_a[−‖z − z_θ(s,a)‖]`` is ~zero and exploration is dead no matter how good
the rest looks. Threshold ~0.3 (design §5).

Why z_θ (reparam) not μ: the cost is defined on the reparam sample
(idea.md / ``scout.guidance.cost``), whose gradient w.r.t. a has TWO channels
``∂μ/∂a + ε·∂σ/∂a``. Measuring only ``∂μ/∂a`` (the old version) would miss the
σ-channel -- and the KL term drives ``∂μ/∂a → 0`` as z→N(0,I), so a μ-only
sensitivity would falsely report "guidance dead" precisely when sampling is
most meaningful. The σ-channel (``∂σ/∂a``) survives KL better because KL is a
marginal constraint (E[σ²]→1); per-input σ can stay action-sensitive.
"""

import torch

from scout.model.vib import reparam


def sensitivity_ratio(model, obs_t, A_t, sigma_a, sigma_z):
    """``mean_b ‖Σ_k ∂z_θ_{b,k}/∂A_b‖₂ · σ_a / σ_z`` -- a scalar proxy for
    ``‖∂z_θ/∂a‖_F · σ_a / σ_z`` (z_θ = reparam sample = p_θ(s̄_t,a)).

    Inputs:
      model     : a :class:`scout.model.scout_vib.ScoutVIB` (uses ``.E_s`` +
                   ``.vib_enc``); must expose ``.encode`` / ``.E_s`` / ``.vib_enc``.
      obs_t     : ``{"visual": {view: (B,1,3,H,W)}, "proprio": (B,1,P)}`` --
                   the LPB-style dual-input batch (scout_design.md §2). The
                   frozen ResNet inside E_s contributes no grad either way.
      A_t       : ``(B, action_dim)`` tensor; will be turned into a leaf that
                   requires grad (a clone -- the caller's tensor is untouched).
      sigma_a   : scalar, action scale = mean per-dim std of A_t (sampled from
                   the dataset's raw actions).
      sigma_z   : scalar, z_θ scale = mean per-dim std of the reparam-sampled
                   z_θ over a batch (NOT μ -- z_θ = μ + σ·ε has larger spread).

    Returns a Python float.

    Exact formula: with ``z_θ = reparam(vib_enc(s̄_t, A_t))`` and
    ``g = ∂(Σ_k z_θ_k)/∂A_t`` (one backward pass on ``z_θ.sum()``), ``g[b]`` is
    the *row-sum* of the per-sample Jacobian ``(style_dim, action_dim)``,
    capturing both ``∂μ/∂a`` and ``ε·∂σ/∂a``. We report
    ``mean_b ‖g_b‖₂ · σ_a/σ_z``. The row-sum can hide sign cancellations across
    style dims; if a β looks borderline, upgrade to a Hutchinson estimator of
    the squared Frobenius norm (``E_v ‖Jᵀ v‖² ≈ ‖J‖²_F``, one backward pass
    with random v). The threshold (~0.3) is approximate; the point is "is z_θ
    sensitive to a at all". Average over multiple ε draws to reduce the
    reparam noise in the estimate.
    """
    A = A_t.detach().clone().requires_grad_(True)
    s_bar = model.encode(obs_t)                                      # (B, s_bar_dim)
    mu, logvar = model.vib_enc(s_bar, A)
    z_enc = reparam(mu, logvar)                                      # z_θ = p_θ(s̄_t, a)
    g = torch.autograd.grad(z_enc.sum(), A, create_graph=False)[0]   # ∂z_θ/∂a (双通道)
    jac_norm = g.flatten(1).norm(dim=-1).mean()                      # mean over batch
    return float(jac_norm * sigma_a / sigma_z)
