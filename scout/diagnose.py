"""VIB life/death diagnostics (scout_design.md §5, §7 risk #1).

The single number that tells whether guidance can work at all:
``sensitivity = ‖∂μ/∂a‖ · σ_a / σ_μ``. If μ does not respond to actions, the
guidance gradient ``∇_a[−‖z − μ(s,a)‖]`` is ~zero and exploration is dead no
matter how good the rest looks. Threshold ~0.3 (design §5).
"""

import torch


def sensitivity_ratio(model, S_t, A_t, sigma_a, sigma_mu):
    """``mean_b ‖Σ_k ∂μ_{b,k}/∂A_b‖₂ · σ_a / σ_μ`` -- a scalar proxy for
    ``‖∂μ/∂a‖_F · σ_a / σ_μ``.

    Inputs:
      model     : a :class:`scout.model.scout_vib.ScoutVIB` (uses ``.ae`` +
                   ``.vib_enc``).
      S_t       : ``(B, state_dim)`` tensor (CPU or GPU).
      A_t       : ``(B, action_dim)`` tensor; will be turned into a leaf that
                   requires grad (a clone -- the caller's tensor is untouched).
      sigma_a   : scalar, action scale = mean per-dim std of A_t (from
                   ``source.stats()['A_t'].std.mean()``).
      sigma_mu  : scalar, μ scale = mean per-dim std of μ over a batch.

    Returns a Python float.

    .. note:: Exact formula: with ``g = ∂(Σ_k μ_k)/∂A_t`` (a single backward
       pass on ``mu.sum()``), ``g[b]`` is the *row-sum* of the per-sample
       Jacobian ``(style_dim, action_dim)``. We report
       ``mean_b ‖g_b‖₂ · σ_a/σ_μ``. The row-sum can hide sign cancellations
       across style dims; if a β looks borderline, upgrade to a Hutchinson
       estimator of the squared Frobenius norm
       (``E_v ‖Jᵀ v‖² ≈ ‖J‖²_F``, one backward pass with random v). The
       threshold (~0.3) is approximate; the point is "is μ sensitive to a at
       all".
    """
    A = A_t.detach().clone().requires_grad_(True)
    s_bar = model.ae.encode(S_t)
    mu, _ = model.vib_enc(s_bar, A)
    g = torch.autograd.grad(mu.sum(), A, create_graph=False)[0]   # (B, action_dim)
    jac_norm = g.flatten(1).norm(dim=-1).mean()                   # mean over batch
    return float(jac_norm * sigma_a / sigma_mu)
