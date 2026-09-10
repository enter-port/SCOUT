"""WalkCloud cost (user 2026-09-10, idea/walkcloud_plan.md, branch drift-dev).

The drifting transplant in its literal-minimal form -- a PURE cost swap
(zero engine changes, zero cloud ledger, zero gates; cf. the reverted
cloudrep, whose j-axis retry-anchor cloud + start gate read the directive
wrong): within each chunk's 100-step denoise walk, every guided step's
one-step x̂₀ estimate is accumulated into a per-chunk CLOUD, and the
step's cost repels the candidate against that cloud with the drifting
kernel:

    H_t = { q_φ(z|s̄, x̂₀_1), ..., q_φ(z|s̄, x̂₀_{t-1}) }   (detached constants)
    S_t = -τ̃ · log Σ_{j<t} exp( -KL( q_φ(z|s̄,x̂₀_t) ‖ H_t[j] ) / τ̃ )   (soft-min)
    cost row = -min(S_t, κ)

  * the cloud is THIS chunk only (select_z clears it per chunk). It is
    appended INSIDE _kl_backward AFTER the backward, so the current step
    never self-references; the t=1 push IS the anchor posterior (select_z
    fires on the same pre-injection x̂₀ within the same step and the
    encoder is deterministic, so no separate anchor capture is needed --
    anchor = history[0] by construction).
  * soft-min IS the drifting repulsion kernel: the NEAREST cloud
    reference dominates, the softmax weights sum to 1 (bounded
    displacement), and the kernel gradient dS/dKL_j =
    softmax(-KL_j/τ̃)_j.
  * boundary degeneracies are native (no rollout-side special cases):
    t=1 empty cloud -> graph-connected zero (the step is not guided);
    t=2 cloud = {x̂₀_1} = the anchor -> S == KL(q_t ‖ q_anchor) EXACTLY
    (the J=1 fast path returns the KL verbatim -- bitwise atypical on
    CPU, any τ̃; on GPU a cross-shape reduction may differ at ulp level,
    same caveat as the _kl_rows vectorization); t>=3 the cloud grows and
    the cost departs from atypical.
  * vs GAElike (same i-axis, falsified under SUM aggregation): GAElike
    summed γ^k-weighted capped KLs (every reference pushes, anchor
    dominates, unnormalized resultant); WalkCloud aggregates by KERNEL
    soft-min (nearest dominates, normalized weights). The i axis has
    never been tested under kernel aggregation -- this A/B answers
    exactly that question.
  * τ̃ (cloud_tau): 'adapt' (default) = clamp(cloud_tau_frac * per-row
    KL-pool RANGE, cloud_tau_min, cloud_tau) -- the cloudrep iter-2
    per-row rule transplanted verbatim (an absolute τ on square's
    compressed KL scale acted as a huge τ -> flat weights -> the kernel
    force averaged away); 'fixed' = cloud_tau verbatim. τ̃ is computed
    from DETACHED KLs -- a schedule constant that consumes no gradient.
  * cloud_hist_max > 0 keeps only the newest N cloud entries (0 = grow
    unbounded; a full 100-step walk holds (B, 100, dz) -- tiny).

Implementation notes (subclass contract): KLCostPlanner owns ALL of
phase 1 (capped climb via _climb_gradient, merged single-backward
guided_step, eta_tilde normalization). This subclass swaps ONLY the
climbed scalar: _kl_backward returns the soft-min score S (UNCAPPED --
the inherited cap mask saturates rows at S > κ exactly as atypical does
on KL) plus the cloud append. compute_loss is a read-only view (no
append). set_row_jobs / try_started_gate / on_try_done are deliberately
NOT defined: the engine's hook dispatch falls through to its None branch,
so rollout_vec needs zero changes. No RNG is consumed anywhere
(bit-comparability with the other KL-cost arms on the same seed).
"""

from __future__ import annotations

import torch

from scout.guidance.entropy_costs import KLCostPlanner, _enc_forward


def _kl_matrix(mu: torch.Tensor, logvar: torch.Tensor,
               hist_mu: torch.Tensor, hist_lv: torch.Tensor) -> torch.Tensor:
    """(B, H) pairwise diagonal-Gaussian KL( q(z|s̄,a_t) ‖ H[j] ) -- the
    (B, H, dz) form of entropy_costs._kl_rows, elementwise identical per
    (row, column): FORWARD KL (current posterior first, matching the
    atypical convention), mean difference Mahalanobis-weighted by the
    REFERENCE posterior's 1/σ²."""
    var = torch.exp(logvar).unsqueeze(1)                    # (B, 1, dz)
    var0 = torch.exp(hist_lv)                               # (B, H, dz)
    return 0.5 * (((mu.unsqueeze(1) - hist_mu) ** 2 / var0)
                  + (var / var0) - 1.0
                  - (logvar.unsqueeze(1) - hist_lv)).sum(dim=-1)   # (B, H)


def _cloud_softmin(kl: torch.Tensor, tau: torch.Tensor):
    """Soft-min over the cloud pool and its normalized kernel weights.

        S = -τ̃ · logsumexp(-KL/τ̃)      (B,)      -- nearest reference dominates
        w = softmax(-KL/τ̃)             (B, H)    -- sums to 1; dS/dKL_j = w_j

    ``tau`` is a per-row (B,) constant (detached by the caller); both
    outputs go through numerically stable logsumexp / softmax."""
    neg = -kl / tau.unsqueeze(1)                            # (B, H)
    S = -tau * torch.logsumexp(neg, dim=1)                  # (B,)
    w = torch.softmax(neg, dim=1)                           # (B, H)
    return S, w


class WalkCloudCostPlanner(KLCostPlanner):
    """WalkCloud cost: soft-min drifting-kernel repulsion from the cloud of
    every x̂₀ iterate the current chunk's denoise walk has produced."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0, eta_dimless: bool = False,
                 cloud_tau: float = 0.5, cloud_tau_mode: str = "adapt",
                 cloud_tau_frac: float = 0.3, cloud_tau_min: float = 0.02,
                 cloud_hist_max: int = 0):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap, eta_dimless=eta_dimless)
        self.cloud_tau = float(cloud_tau)
        self.cloud_tau_mode = str(cloud_tau_mode)
        self.cloud_tau_frac = float(cloud_tau_frac)
        self.cloud_tau_min = float(cloud_tau_min)
        self.cloud_hist_max = int(cloud_hist_max)
        if self.cloud_tau_mode not in ("adapt", "fixed"):
            raise ValueError(f"cloud_tau_mode must be 'adapt' or 'fixed'; "
                             f"got {self.cloud_tau_mode!r}")
        if self.cloud_tau <= 0.0:
            raise ValueError(f"cloud_tau must be > 0; got {self.cloud_tau}")
        if self.cloud_tau_frac <= 0.0:
            raise ValueError(f"cloud_tau_frac must be > 0; "
                             f"got {self.cloud_tau_frac}")
        if not 0.0 < self.cloud_tau_min <= self.cloud_tau:
            raise ValueError(f"cloud_tau_min must be in (0, cloud_tau="
                             f"{self.cloud_tau}]; got {self.cloud_tau_min}")
        if self.cloud_hist_max < 0:
            raise ValueError(f"cloud_hist_max must be >= 0 (0 = unbounded); "
                             f"got {self.cloud_hist_max}")
        # cloud: (B, H, dz) posteriors of x̂₀_1..x̂₀_{t-1}, detached
        # constants w.r.t. the climb gradient (same treatment as the
        # vanilla anchor _base_mu/_base_lv and GAElike's history).
        self._hist_mu: torch.Tensor | None = None
        self._hist_lv: torch.Tensor | None = None
        # telemetry accumulators (printed on the [walkcloud-telemetry] tick,
        # mirroring [gae-telemetry]; mean_S is the probe surface for the
        # pre-registered "S 自行变小 → 引导自然衰减" risk readout).
        self._wc_calls = 0
        self._hist_len_acc = 0.0
        self._s_acc: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # per-chunk reset: clear the cloud -- no anchor capture, no commit
    # ------------------------------------------------------------------ #
    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Per-chunk reset. The anchor is NOT captured here and NOTHING is
        committed: anchor = history[0] by construction (the t=1 _kl_backward
        push encodes the same pre-injection x̂₀ this hook saw)."""
        self._hist_mu = None
        self._hist_lv = None
        return None

    def reset(self):
        """Base reset (clear s̄_t / z caches) + drop the cloud -- a reused
        planner instance must not see a stale, same-B cloud (which would
        slip past the row-alignment guard; same hazard as GAElike P2-5)."""
        super().reset()
        self._hist_mu = None
        self._hist_lv = None

    # ------------------------------------------------------------------ #
    # cloud bookkeeping
    # ------------------------------------------------------------------ #
    def _push_history(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Append this step's (detached) posterior as the newest cloud entry.
        Called from _kl_backward only, AFTER the backward -- the current
        step never self-references. A missing/row-misaligned cloud restarts
        fresh from this posterior (unreachable on the rollout path, where
        one replan call keeps a constant B; restarting makes the guard a
        one-step no-guide event instead of a whole-chunk kill)."""
        mu_d = mu.detach().unsqueeze(1)                     # (B, 1, dz)
        lv_d = logvar.detach().unsqueeze(1)
        if (self._hist_mu is None or self._hist_lv is None
                or self._hist_mu.shape[0] != mu.shape[0]):
            self._hist_mu, self._hist_lv = mu_d, lv_d
        else:
            self._hist_mu = torch.cat([self._hist_mu, mu_d], dim=1)
            self._hist_lv = torch.cat([self._hist_lv, lv_d], dim=1)
        if 0 < self.cloud_hist_max < self._hist_mu.shape[1]:
            self._hist_mu = self._hist_mu[:, -self.cloud_hist_max:]
            self._hist_lv = self._hist_lv[:, -self.cloud_hist_max:]

    def _tau_rows(self, kl: torch.Tensor) -> torch.Tensor:
        """Per-row kernel temperature τ̃ (B,), a detached schedule constant.
        adapt: clamp(frac * (max_j - min_j) KL of THIS row's pool, min,
        cloud_tau) -- per-row so each env's cloud sets its own scale (H=1
        gives range 0 -> τ̃ = cloud_tau_min; irrelevant, the J=1 fast path
        bypasses τ̃ entirely). fixed: cloud_tau verbatim."""
        if self.cloud_tau_mode == "fixed":
            return torch.full_like(kl.detach()[:, 0], self.cloud_tau)
        kld = kl.detach()
        rng = kld.amax(dim=1) - kld.amin(dim=1)             # (B,)
        return torch.clamp(self.cloud_tau_frac * rng,
                           min=self.cloud_tau_min, max=self.cloud_tau)

    # ------------------------------------------------------------------ #
    # the cost
    # ------------------------------------------------------------------ #
    def _cloud_rows(self, mu: torch.Tensor, logvar: torch.Tensor,
                    x0_hat: torch.Tensor) -> torch.Tensor:
        """Per-row soft-min score S (B,), UNCAPPED (the inherited
        _climb_gradient cap mask applies κ downstream). Empty cloud (t=1)
        or a row-misaligned cloud -> graph-connected zeros, exactly the
        ``_kl_rows`` None-baseline contract."""
        if (self._hist_mu is None or self._hist_lv is None
                or self._hist_mu.shape[0] != mu.shape[0]):
            return x0_hat.flatten(1).sum(dim=1).to(mu.dtype) * 0.0
        kl = _kl_matrix(mu, logvar, self._hist_mu, self._hist_lv)
        if kl.shape[1] == 1:
            # J=1 fast path: cloud = {anchor} -> S == the anchor KL
            # verbatim -- bitwise atypical at t=2 under ANY τ̃ (skips the
            # τ̃·(KL/τ̃) round-trips that would break bit-equality; CPU
            # bitwise per smoke check 2, GPU ulp caveat as above).
            return kl[:, 0]
        tau = self._tau_rows(kl)
        S, _ = _cloud_softmin(kl, tau)
        return S

    def _kl_backward(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                     current_obs=None):
        """ONE encoder forward + ONE summed backward -> per-row UNCAPPED
        ``(S, g)`` where g is the gradient of ``S.sum()`` w.r.t. the
        trajectory (ascent on S = repulsion from the cloud). The cloud
        advance happens AFTER the backward; the empty-cloud t=1 step still
        pushes (its posterior becomes history[0] = the anchor for t=2)
        while returning a graph-connected zero (value, gradient)."""
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        S = self._cloud_rows(mu, logvar, x0_hat)
        g = torch.autograd.grad(S.sum(), trajectory)[0]
        # cloud advance AFTER the backward: the current step never
        # self-references (the t=1 push IS the anchor posterior).
        self._push_history(mu, logvar)
        self._wc_calls += 1
        if self._hist_mu is not None:
            self._hist_len_acc += float(self._hist_mu.shape[1])
        _s_mean = S.detach().mean()
        self._s_acc = (_s_mean if self._s_acc is None
                       or self._s_acc.device != _s_mean.device
                       else self._s_acc + _s_mean)
        if self._wc_calls % 2500 == 0:
            print(f"[walkcloud-telemetry] calls={self._wc_calls} "
                  f"mean_hist_len={self._hist_len_acc / self._wc_calls:.1f} "
                  f"mean_S={float(self._s_acc) / self._wc_calls:.4g} "
                  f"tau_mode={self.cloud_tau_mode} "
                  f"hist_max={self.cloud_hist_max}",
                  flush=True)
        return S, g

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        """Read-only view of the WalkCloud row cost (NO cloud append), for
        monitoring / diagnostics; the rollout hot path uses guided_step.
        Row = -min(S, κ), exactly atypical's -min(KL, κ) form."""
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        S = self._cloud_rows(mu, logvar, x0_hat)
        nll = -torch.clamp(S, max=float(self.cap))
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
