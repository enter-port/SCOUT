"""Orbit guidance: constrained control on the kappa-shell (user 2026-08-31).

Research frame: ``idea/escape_coverage_research.md`` (约束控制章). The entropy
cost's climb is ARGMAX control -- every retry converges onto one ridge of the
J^T.Lambda.J landscape (narrow cone). Particle guidance widens the ensemble by
mutual repulsion; orbit instead changes the SINGLE-trajectory control law:
once a retry reaches the escape shell {KL >= kappa} it STAYS there and tours
the shell, instead of continuing to push radially outward.

Derivation chain (math session, msg 42):

  1. KL-control / path-integral control (Kappen 2005; Todorov 2006; Dvijotham
     UAI 2011): a diffusion process under KL-regularized control has an HJB
     equation that linearizes under the log transform; the optimal control
     field is u*(x) ∝ grad log V(x).
  2. Classifier guidance's injected grad f is the greedy (argmax) approximation
     of grad log V. Swapping the peak-shaped reward (maximize f) for a
     plateau-shaped one (REACH {f >= kappa}) turns argmax control into
     CONSTRAINT control -- same variational family as the entropy cost
     (paper narrative: "from argmax control to constrained control").
  3. Constrained-manifold sampling: Zappa Holmes-Cerfon & Goodman, "Monte
     Carlo on Manifolds" (2018); Holmes-Cerfon 2024; constrained Langevin with
     Lagrange multipliers (Leimkuhler & Matthews).
  4. Two-phase injection, decided per row at EVERY guided denoise step
     (f = UNCAPPED KL(q(z|s,a) || q(z|s,a^DP)), kappa = atypical cap):
       (i)  f < kappa - delta : the standard climb, verbatim
            (eta * sqrt(1-abar_t) * grad f) -- identical to atypical;
       (ii) f >= kappa - delta: constrained dynamics --
            - tangential noise xi_perp (the grad-f normal component projected
              out): direction coverage comes from touring the level set;
            - Newton feedback -lam*(f-kappa)*grad f / ||grad f||^2, pinning f
              at kappa from either side (never pushes past the shell, so the
              particle-repulsion overdose cliff cannot occur by construction).
  5. The Newton step is reparametrization-invariant: computed with
     g = df/dx_t it equals the x0_hat-space Newton step (the 1/sqrt(abar)
     Jacobian of pred_original_sample cancels between the numerator and the
     ||g||^2 denominator) -- under the frozen-eps (affine x0_hat in x_t)
     approximation; the actual graph also differentiates through the UNet
     Jacobian, so the equivalence is approximate to the same order as every
     other injection here (the climb included). The tangential noise is
     scaled by sqrt(1-abar_t) to match the injection convention (anneals
     like the DDPM process noise; sigma_orb is its own dose knob,
     INDEPENDENT of eta).
  6. Phase 2 REPLACES the climb on its rows (the feedback itself keeps
     climbing below kappa -- -lam*(f-kappa) is positive along grad f when
     f < kappa), so there is no double dose at the hand-over. Note the
     hand-over is DIRECTIONAL, not force, continuity: at the boundary the
     injected magnitude jumps from eta*sqrt(1-abar)*||grad f|| to
     lam*|f-kappa|/||grad f||, and the Newton term deliberately carries no
     sqrt(1-abar_t) annealing (scale-free step) -- so late-denoise feedback
     is relatively stronger than the vanishing climb/noise. Dose calibration
     should read the telemetry's mean|fb| and mean|noise| separately.

No-op sentinel: (orbit_lam=0, orbit_sigma=0, orbit_delta=0) is bit-identical
to --guide atypical (phase-2 rows then have zero capped-climb gradient anyway,
no noise is drawn, and the extra backward consumes no RNG).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward
from scout.guidance.planner import ScoutPlanner


def orbit_displacement(kl: torch.Tensor, g: torch.Tensor, kappa: float,
                       lam: float, delta: float, sigma: float,
                       noise_scale: float = 1.0):
    """Pure phase-2 math (unit-tested directly): given the per-row UNCAPPED
    cost ``kl`` (B,) and its per-row gradient w.r.t. the trajectory ``g``
    (B, T, Da) -- block-diagonal, so row slices are per-row gradients of the
    summed backward -- return ``(disp, phase2, stats)``:

      disp    (B, T, Da) displacement, zero on phase-1 rows;
      phase2  (B,) float mask (1 = constrained dynamics, 0 = keep the climb);
      stats   (fb_norm, noise_norm): per-row L2 norms of the two parts
              (detached; telemetry only).

    ``sigma=0`` draws NO random numbers (bit-identity with atypical holds).
    """
    kl_d = kl.detach()
    p2 = kl_d >= (float(kappa) - float(delta))
    gnorm2 = (g.detach() ** 2).flatten(1).sum(dim=1)             # (B,)
    small = gnorm2 < 1e-16            # flat rows: Newton direction undefined
    safe = gnorm2.clamp(min=1e-16)
    fb_coeff = torch.where(small, torch.zeros_like(kl_d),
                           -float(lam) * (kl_d - float(kappa)) / safe)
    fb = fb_coeff[:, None, None] * g                               # (B,T,Da)
    noise = torch.zeros_like(g)
    if float(sigma) > 0.0:
        # flat rows: ghat exactly 0 -> UNPROJECTED (full) noise; near-flat
        # rows above the threshold keep their (unit) direction (review P2).
        ghat = torch.where(small[:, None, None], torch.zeros_like(g),
                           g / safe.sqrt()[:, None, None])
        xi = torch.randn(g.shape, device=g.device, dtype=g.dtype)
        dot = (xi * ghat).flatten(1).sum(dim=1)
        noise = float(noise_scale) * float(sigma) * (xi - dot[:, None, None] * ghat)
    mask = p2.to(g.dtype)[:, None, None]
    disp = (fb + noise) * mask
    stats = ((fb_coeff.abs() * safe.sqrt()).detach(),
             noise.detach().flatten(1).norm(dim=1))
    return disp, p2.to(g.dtype), stats


class OrbitCostPlanner(ScoutPlanner):
    """Entropy cost + two-phase constrained control (orbit).

    The phase-1 climb (per-chunk KL-to-own-intent, capped) is DELEGATED to an
    internal :class:`AtypicalCostPlanner` -- ``compute_loss`` is verbatim
    atypical, and ``orbit_update`` (duck-typed hook consumed by policy.py's
    injection line) supplies the phase-2 displacement. Rows are INDEPENDENT
    (no inter-particle coupling) -- no group lock, i.i.d. retries exactly as
    atypical.
    """

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0,
                 orbit_lam: float = 0.5, orbit_delta: float = 0.25,
                 orbit_sigma: float = 0.25):
        super().__init__(scout_vib, bridge=bridge, z=None,
                         obs_adapter=obs_adapter)
        self._att = AtypicalCostPlanner(scout_vib, bridge=bridge,
                                        obs_adapter=obs_adapter, cap=cap)
        self.orbit_lam = float(orbit_lam)
        # guard: kappa - delta <= 0 would put EVERY row (KL~0 included) in
        # phase 2 and turn the run into unprojected noise-everything; clamp
        # delta just under kappa instead (review P1).
        if float(cap) - float(orbit_delta) <= 0.0:
            orbit_delta = max(float(cap) - 1e-6, 0.0)
            print(f"[orbit] WARNING: orbit_delta >= cap "
                  f"({orbit_delta} >= {cap}); clamped to {orbit_delta} "
                  f"-- phase 2 would otherwise swallow every row",
                  flush=True)
        self.orbit_delta = float(orbit_delta)
        self.orbit_sigma = float(orbit_sigma)
        # telemetry (dose calibration reads this): device-side accumulators,
        # host sync only on the print tick (the pg-telemetry pattern).
        self._orb_calls = 0          # orbit_update invocations
        self._orb_rows = 0           # rows seen (host int -- shape, no sync)
        self._orb_p2_rows = 0        # rows that entered phase 2 (print tick)
        self._p2_acc: Optional[torch.Tensor] = None    # device sum of p2
        self._fb_acc: Optional[torch.Tensor] = None    # sum per-row |fb| norm
        self._noise_acc: Optional[torch.Tensor] = None  # sum per-row noise norm

    @property
    def p2_rows(self) -> int:
        """Phase-2 row count so far (one host sync -- tests/probes only;
        production reads it off the orbit-telemetry print tick)."""
        return int(self._p2_acc) if self._p2_acc is not None else 0

    # -- context plumbing (rollout_vec._replan hooks) ---------------------- #
    def set_row_context(self, init_ids: Sequence):
        pass    # rows are independent; hook fires regardless (atypical idem)

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Per-chunk intent anchor (mu^0, sigma^0^2) -- delegated verbatim to
        the atypical planner (captured at the FIRST guided denoise step)."""
        return self._att.select_z(x0_hat, current_obs)

    def set_current_obs(self, current_obs):
        self._att.set_current_obs(current_obs)

    # -- the cost (phase 1, verbatim atypical) ------------------------------ #
    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        return self._att.compute_loss(x0_hat, current_obs, reduction)

    # -- the constrained update (phase 2) ----------------------------------- #
    def _encode_and_uncapped(self, x0_hat: torch.Tensor, current_obs=None):
        """ONE extra encoder forward -> per-row UNCapped KL list
        (grad-carrying). Kept separate from AtypicalCostPlanner._
        encode_and_row_losses so that class (shared with particle guidance)
        stays untouched; the encoder is a tiny MLP on top of the cached s_bar,
        negligible next to the UNet backward this shares the graph with."""
        s_bar_t = self._att._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows: List[torch.Tensor] = []
        for i in range(mu.shape[0]):
            if i >= len(self._att._base_mu) or self._att._base_mu[i] is None:
                rows.append(x0_hat[i].sum() * 0.0)   # graph-connected zero
                continue
            m0, lv0 = self._att._base_mu[i], self._att._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            rows.append(kl)
        return mu, rows

    def orbit_update(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                     current_obs=None, noise_scale: float = 1.0):
        """Phase-2 constrained update -- called by policy.py right AFTER its
        capped-loss backward (which retains the shared x0_hat graph for this
        second backward through the UNet). Returns ``(disp, phase2)``; see
        :func:`orbit_displacement`."""
        _, rows = self._encode_and_uncapped(x0_hat, current_obs)
        kl = torch.stack(rows)
        g = torch.autograd.grad(kl.sum(), trajectory)[0]
        disp, p2, (fb_n, noise_n) = orbit_displacement(
            kl, g, kappa=self._att.cap, lam=self.orbit_lam,
            delta=self.orbit_delta, sigma=self.orbit_sigma,
            noise_scale=noise_scale)
        # telemetry (norms masked to phase-2 rows -- the dose numbers must
        # describe what was actually injected, not the pre-mask values).
        # Device-side accumulation only; the host sync happens on the print
        # tick (review P2: a per-call .item() costs ~3.7k stream
        # serializations per guided rollout).
        self._orb_calls += 1
        self._orb_rows += int(p2.shape[0])
        for attr, val in (("_fb_acc", (fb_n * p2).sum()),
                          ("_noise_acc", (noise_n * p2).sum()),
                          ("_p2_acc", p2.sum())):
            cur = getattr(self, attr)
            if cur is None or cur.device != val.device:
                setattr(self, attr, val.detach().clone())
            else:
                setattr(self, attr, cur + val.detach())
        if self._orb_calls % 2500 == 0:
            self._orb_p2_rows = int(self._p2_acc)
            n = max(self._orb_p2_rows, 1)
            print(f"[orbit-telemetry] calls={self._orb_calls} "
                  f"p2_rows={self._orb_p2_rows}/{self._orb_rows} "
                  f"mean|fb|/p2row={float(self._fb_acc) / n:.4g} "
                  f"mean|noise|/p2row={float(self._noise_acc) / n:.4g}",
                  flush=True)
        return disp, p2
