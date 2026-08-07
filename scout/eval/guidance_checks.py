"""E2 / E3 action-level guidance gates (scout_impl_plan.md Task 4.3,
scout_design.md §5 "前置 action 级闸门").

Minimal, dependency-free metrics on action chunks produced by
:meth:`DiffusionUNetPolicy.guided_conditional_sample`. Full E2/E3 on trained
ckpts + real env is deferred (needs trained VIB + DP + data); the functions
here run on whatever ``action_decoder`` / VIB the caller supplies, so they are
unit-testable on dummy inputs (see ``_phase4_verify_tmp.py``).

**E2 — guidance three判据** (sweep ``guidance_scale``, fixed init noise + fixed z set):
  1. diversity    -- action std *across z* per scale (scale=0 → ~0; monotone up).
  2. consistency  -- ``mean_t ‖z − μ(s̄_t, a_guided)‖`` (should *fall* with scale).
  3. cost-direction -- per-step SCOUT cost over the guided denoise steps
                       (should fall; a rising curve means the guidance sign is
                       flipped -- design §4).

**E3 — on-manifold**: jerk (3rd-difference norm) + Mahalanobis-to-demo of the
guided chunks vs the base DP (same order of magnitude = on-manifold).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch

from scout.guidance.cost import scout_cost
from scout.normalizer import ActionNormalizerBridge, IdentityBridge


# --------------------------------------------------------------------------- #
# E2
# --------------------------------------------------------------------------- #
def eval_e2_guidance(
    action_decoder,
    vib_enc,
    bridge: ActionNormalizerBridge,
    cond_data: torch.Tensor,
    cond_mask: torch.Tensor,
    global_cond: torch.Tensor,
    s_bar_t: torch.Tensor,
    z_set: torch.Tensor,
    scales: Sequence[float] = (0.0, 1.0, 5.0, 10.0, 20.0),
    guidance_start_timestep: int = 50,
    seed: int = 0,
    classifier_guidance: bool = True,
) -> Dict[float, dict]:
    """Sweep ``guidance_scale`` over a fixed set of ``z``; return per-scale metrics.

    Args:
        action_decoder : a :class:`scout.policy.diffusion.DiffusionUNetPolicy`
                         (the base DP's action decoder).
        vib_enc        : a :class:`scout.model.vib.VIBEncoder` (used by the cost
                         and re-used for the consistency metric).
        bridge         : :class:`scout.normalizer.ActionNormalizerBridge`.
        cond_data/mask/global_cond : as for ``conditional_sample``.
        s_bar_t        : ``(B, s_latent_dim)`` encoded current obs (fixed).
        z_set          : ``(Z, B, style_dim)`` or ``(Z, style_dim)`` -- a stack of
                         fixed skill latents to sweep (diversity is across Z).
        scales         : guidance_scale values to sweep.
        guidance_start_timestep : gate (a); guide only when ``t < this``.
        seed           : fixed init-noise seed (identical per (scale, z) so the
                         only varying factors are scale and z).

    Returns:
        ``{scale: {"diversity": float, "consistency_mean": float,
                   "cost_curve_mean": list[float], "actions": tensor}}``.
        ``actions`` is ``(Z, B, T, A)`` per scale.
    """
    action_decoder.eval()
    vib_enc.eval()
    if z_set.dim() == 2:                       # (Z, style_dim) -> (Z, 1, style_dim)
        z_set = z_set.unsqueeze(1)
    Z, B, _ = z_set.shape

    results: Dict[float, dict] = {}
    for scale in scales:
        actions_per_z: List[torch.Tensor] = []
        curves: List[List[float]] = []
        consistencies: List[float] = []
        for z_i in range(Z):
            gen = torch.Generator(device=cond_data.device).manual_seed(seed)
            z_b = z_set[z_i]                                    # (B, style_dim)
            traj, curve = action_decoder.guided_conditional_sample(
                cond_data, cond_mask,
                global_cond=global_cond, generator=gen,
                classifier_guidance=classifier_guidance,
                s_bar_t=s_bar_t, z=z_b, vib_enc=vib_enc, bridge=bridge,
                guidance_scale=float(scale),
                guidance_start_timestep=guidance_start_timestep,
                return_cost_curve=True,
            )
            a = traj[..., : action_decoder.action_dim].detach()  # (B, T, A)
            actions_per_z.append(a)
            if curve:
                curves.append(curve)
            # consistency = scout_cost on the final guided chunk (no grad needed)
            with torch.no_grad():
                consistencies.append(float(scout_cost(a, s_bar_t, z_b, vib_enc, bridge)))

        actions_stack = torch.stack(actions_per_z, dim=0)        # (Z, B, T, A)
        diversity = float(actions_stack.std(dim=0).mean())       # mean std across z
        # mean cost curve across z (curves may be empty if no guidance steps ran)
        if curves:
            min_len = min(len(c) for c in curves)
            mean_curve = [sum(c[i] for c in curves) / len(curves) for i in range(min_len)]
        else:
            mean_curve = []

        results[float(scale)] = {
            "diversity": diversity,
            "consistency_mean": sum(consistencies) / max(1, len(consistencies)),
            "cost_curve_mean": mean_curve,
            "actions": actions_stack,
        }
    return results


def summarize_e2(results: Dict[float, dict]) -> str:
    """Pretty one-line-per-scale summary for logging."""
    lines = ["E2 guidance sweep (scale | diversity | consistency | cost_curve):"]
    for scale in sorted(results.keys()):
        r = results[scale]
        curve = r["cost_curve_mean"]
        curve_head = "[" + ", ".join(f"{c:.2f}" for c in curve[:6]) + "...]" if len(curve) > 6 \
            else "[" + ", ".join(f"{c:.2f}" for c in curve) + "]"
        lines.append(
            f"  scale={scale:5.1f} | diversity={r['diversity']:.4f} | "
            f"consistency={r['consistency_mean']:.4f} | curve={curve_head}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# E3
# --------------------------------------------------------------------------- #
def jerk(actions: torch.Tensor) -> float:
    """Mean per-element norm of the 3rd-order finite difference of an action
    chunk. ``actions: (B, T, A)`` (or ``(T, A)``); returns a Python float.
    ``a[t+3] − 3 a[t+2] + 3 a[t+1] − a[t]``. ``T < 4`` -> 0.0.
    """
    if actions.dim() == 2:
        actions = actions.unsqueeze(0)
    B, T, _ = actions.shape
    if T < 4:
        return 0.0
    d3 = (actions[:, 3:] - 3 * actions[:, 2:-1] + 3 * actions[:, 1:-2] - actions[:, :-3])
    return float(d3.norm(dim=-1).mean())


def mahalanobis_to_demo(actions: torch.Tensor, demo_actions: torch.Tensor,
                        reg: float = 1.0e-3) -> float:
    """Mean Mahalanobis distance of ``actions`` to the ``demo_actions`` dist.

    Per-frame: fits ``(mean, cov)`` from ``demo_actions: (N, A)``; evaluates on
    ``actions`` reshaped to ``(*, A)``. Returns the mean Mahalanobis over all
    frames. ``reg`` is added to the diagonal of ``cov`` before inversion
    (robustness for low-N / degenerate demos).
    """
    if actions.dim() > 2:
        actions_flat = actions.reshape(-1, actions.shape[-1])
    else:
        actions_flat = actions
    demo_flat = demo_actions.reshape(-1, demo_actions.shape[-1]) if demo_actions.dim() > 2 \
        else demo_actions
    demo_flat = demo_flat.double()
    actions_flat = actions_flat.double()
    mean = demo_flat.mean(dim=0)
    centered = demo_flat - mean
    cov = (centered.t() @ centered) / max(1, centered.shape[0] - 1)
    cov = cov + reg * torch.eye(cov.shape[0], dtype=cov.dtype, device=cov.device)
    cov_inv = torch.linalg.inv(cov)
    diff = (actions_flat - mean)
    d2 = (diff @ cov_inv * diff).sum(dim=-1)            # (*,)
    return float(d2.clamp(min=0.0).sqrt().mean())


def eval_e3_on_manifold(actions_guided: torch.Tensor,
                        demo_actions: torch.Tensor,
                        actions_base_dp: Optional[torch.Tensor] = None,
                        ) -> Dict[str, float]:
    """jerk + Mahalanobis-to-demo of guided chunks; optional base-DP comparison.

    Args:
        actions_guided : ``(B, T, A)`` guided action chunks (e.g. max-scale
                         output of :func:`eval_e2_guidance`).
        demo_actions   : ``(N, A)`` or ``(N, T', A)`` reference demo frames.
        actions_base_dp: optional ``(B, T, A)`` unguided base-DP chunks (for the
                         "vs base DP -- same order of magnitude" check).

    Returns ``{"jerk_guided", "mahalanobis_guided",
               "jerk_base_dp"?, "mahalanobis_base_dp"?}``.
    """
    out = {
        "jerk_guided": jerk(actions_guided),
        "mahalanobis_guided": mahalanobis_to_demo(actions_guided, demo_actions),
    }
    if actions_base_dp is not None:
        out["jerk_base_dp"] = jerk(actions_base_dp)
        out["mahalanobis_base_dp"] = mahalanobis_to_demo(actions_base_dp, demo_actions)
    return out
