"""Entropy-exploration costs (entropy-dev, user 2026-08-24 方案二/三, v2).

Both planners REPLACE the SCOUT NLL cost inside the SAME guided-denoise
injection path (policy.py ``guided_conditional_sample``): the base DP stays
frozen and acts as the trust region (guided sampler ~ p_DP * exp(-cost)),
only ``compute_loss`` changes.

方案二 NoveltyCostPlanner (v2 -- after the group-1 reflection):
  Objective (trajectory level): maximize the empirical entropy of the skill
  codes EXECUTED on a scene across its retries.  Greedy per-chunk surrogate
  (pseudo-count / KDE, Bellemare'16, Tang'17): minimize the KDE density of
  the candidate action's encoder code within the codes already EXECUTED on
  this scene by EARLIER RETRIES (inter-try novelty only -- a retry must
  differ from the previous retries of the same scene; within a retry there
  is no self-repulsion, so the try keeps behavioral coherence).

      cost(a) = log[ (1/N) Σ_j exp(-Σ_i (μ_i(s̄,a)-z_j,i)² / (2 h_i²)) + ε ]

  * z_j = μ(s̄ at chunk start, EXECUTED raw action chunk) of a previous
    retry -- recorded by rollout_vec at replan time (the exhausted chunk)
    and committed to the scene buffer when the retry FINALIZES (tries of a
    scene are launched serially via the runner's job gate, so retry j sees
    everything from retries < j).
  * h_i = max(h_scale·σ̄_i, spread_c·std_i(buffer codes)) -- the kernel
    width adapts to the actual spread of tried behaviors; with the default
    h_scale=5 the cost is effectively quadratic repulsion from the buffer
    centroid (constant-magnitude force, no far-field saturation).
  * Rows with an empty buffer contribute a graph-connected ZERO (no pull --
    retry 0 of a scene is the natural unguided retry).

方案三 KLCostPlanner (renamed from AtypicalCostPlanner 2026-09-04, user
order -- same mechanism, new name): cost =
-min(KL(q(z|s̄,a) ‖ q(z|s̄,a^DP)), κ) with the per-chunk unguided-intent
baseline (μ⁰, σ⁰²) captured by the select_z hook. The class owns the WHOLE
phase-1 machinery (capped climb, merged single-backward guided_step,
eta_tilde normalization) that the orbit subclass (orbit_costs.py) inherits
verbatim -- the subclass adds ONLY phase 2.

方案A ShellTargetCostPlanner (user 2026-08-27): 方案二 abandoned (its diversity
came only from the evolving per-scene cache -- with a fixed cache the guided
direction is definite, same narrow-cone flaw); instead SOE's per-retry random
direction moves INTO the cost: pull q(z|s̄,a) toward a per-retry random target
posterior on the kappa-shell of the intent posterior.  See class docstring.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from scout.guidance.planner import ScoutPlanner


def _enc_forward(planner: ScoutPlanner, x0_hat: torch.Tensor) -> torch.Tensor:
    """bridge(x̂₀) -> flattened chunk -> raw action vector fed to the encoder."""
    per_step = x0_hat.shape[-1]
    chunk_dim = int(getattr(planner.scout_vib.vib_enc, "action_dim", per_step))
    n_steps = chunk_dim // per_step
    a = planner.bridge(x0_hat[:, :n_steps])
    return a.reshape(x0_hat.shape[0], chunk_dim)


def _kl_rows(mu: torch.Tensor, logvar: torch.Tensor,
             base_mu: Sequence[Optional[torch.Tensor]],
             base_lv: Sequence[Optional[torch.Tensor]],
             x0_hat: torch.Tensor) -> torch.Tensor:
    """Vectorized per-row UNCAPPED KL(q(z|s̄,a) ‖ q(z|s̄,a⁰)) -> (B,).

    Exact batched rewrite of the historical per-row loop (perf 2026-09-01,
    方案二: the loop cost ~6 kernel launches x B per call on the guided hot
    path -- py-spy on the running orbit chain showed the guidance section at
    ~55% of wall clock, kernel-launch bound). Elementwise ops are
    per-element identical to the loop; the ``style_dim``-sized last-dim
    reduction replaces B per-row ``.sum()`` calls (CPU: bitwise-identical,
    asserted by verify check 16d; CUDA: a batched row-reduce may reorder the
    16-element accumulation vs the standalone per-row sum, so cross-version
    GPU bit-replay of pre-refactor runs is not guaranteed -- ulp-level,
    science-neutral). Rows without a captured baseline (missing/None in
    EITHER list -- only possible on direct calls before ``select_z``, which
    sets both atomically) keep the loop's graph-connected-zero contract via
    ``torch.where`` on a zero branch built from ``x0_hat`` (value AND
    gradient contribution exactly zero, as ``x0_hat[i].sum() * 0.0`` was;
    the pre-vectorization loop crashed loudly on an asymmetric None, the
    new code zeroes the row instead -- unreachable via the public API).
    """
    B = mu.shape[0]

    def _has(it, i):
        return i < len(it) and it[i] is not None

    m0 = torch.stack([
        (base_mu[i] if _has(base_mu, i) else torch.zeros_like(mu[i]))
        for i in range(B)])
    lv0 = torch.stack([
        (base_lv[i] if _has(base_lv, i) else torch.zeros_like(mu[i]))
        for i in range(B)])
    var, var0 = torch.exp(logvar), torch.exp(lv0)
    kl = 0.5 * (((mu - m0) ** 2 / var0)
                + (var / var0) - 1.0 - (logvar - lv0)).sum(dim=-1)
    if any(not (_has(base_mu, i) and _has(base_lv, i)) for i in range(B)):
        valid = torch.tensor(
            [_has(base_mu, i) and _has(base_lv, i)
             for i in range(B)], device=mu.device)
        kl = torch.where(valid, kl,
                         x0_hat.flatten(1).sum(dim=1).to(kl.dtype) * 0.0)
    return kl


class NoveltyCostPlanner(ScoutPlanner):
    """方案二 v2: minimize the KDE density of the candidate code among the
    EXECUTED codes of earlier retries of the same scene."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 h_scale: float = 5.0, spread_c: float = 1.0,
                 sample_z: bool = False, kde_eps: float = 1e-4):
        super().__init__(scout_vib, bridge=bridge, z=None, obs_adapter=obs_adapter)
        self.h_scale = float(h_scale)
        self.spread_c = float(spread_c)
        self.sample_z = bool(sample_z)
        self.kde_eps = float(kde_eps)
        # init_idx -> list of EXECUTED codes from finalized retries
        self._buffers: dict = {}
        self._row_keys: List = []
        self._row_eps: List[Optional[torch.Tensor]] = []
        self._sigma_bar: Optional[torch.Tensor] = None

    # -- context plumbing (rollout_vec._replan) ---------------------------- #
    def set_row_context(self, init_ids: Sequence):
        """Map each batch row to its scene (init_idx). No positional state."""
        self._row_keys = list(init_ids)

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Keep the per-chunk hook path active (vec skips per-rollout z draws
        for planners with select_z); the novelty cost itself needs no anchor.
        Also refreshes the running σ̄ from the unguided intent."""
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t, a)
            self._update_sigma_bar(torch.exp(0.5 * logvar))
            if self.sample_z:
                eps = torch.randn_like(mu[0])
                self._row_eps = [e for e in eps.unsqueeze(0)]
        return None

    # -- executed-code recording (rollout_vec) ----------------------------- #
    def encode_executed(self, obs, chunk_np) -> torch.Tensor:
        """Code of an EXECUTED chunk: raw actions + the obs the chunk was
        conditioned on (single env, unbatched numpy obs dict)."""
        dev = next(self.scout_vib.parameters()).device
        # match the policy-path convention: E_s consumes the LAST obs frame
        # (predict_action_dyn_guided slices x[:, -1:, ...] before the adapter)
        obs_t = {k: torch.as_tensor(np.asarray(v, dtype=np.float32))[-1:].unsqueeze(0).to(dev)
                 for k, v in obs.items()}
        obs_es = (self.obs_adapter(obs_t) if self.obs_adapter is not None
                  else obs_t)
        a_flat = torch.as_tensor(
            np.asarray(chunk_np, dtype=np.float32).reshape(1, -1)).to(dev)
        with torch.no_grad():
            s_bar = self.scout_vib.encode(obs_es)
            mu, _ = self.scout_vib.vib_enc(s_bar.to(a_flat.device), a_flat)
        return mu[0].detach().cpu()

    def on_try_done(self, init_idx, codes: Sequence[torch.Tensor]):
        """A retry finalized -- commit its executed codes to the scene buffer."""
        if codes:
            self._buffers.setdefault(init_idx, []).extend(codes)

    def _update_sigma_bar(self, sigma: torch.Tensor):
        batch_mean = sigma.detach().mean(dim=0)
        if self._sigma_bar is None:
            self._sigma_bar = batch_mean.clone()
        else:
            self._sigma_bar = self._sigma_bar.to(batch_mean.device)
            self._sigma_bar.mul_(0.99).add_(batch_mean, alpha=0.01)

    # -- the cost ----------------------------------------------------------- #
    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        if self.sample_z and self._row_eps:
            eps = torch.stack([self._row_eps[i] if i < len(self._row_eps)
                               else torch.zeros_like(mu[0])
                               for i in range(mu.shape[0])]).to(mu.device, mu.dtype)
            code = mu + torch.exp(0.5 * logvar) * eps
        else:
            code = mu
        sig = (self._sigma_bar.to(mu.device) if self._sigma_bar is not None
               else torch.ones_like(mu[0]))
        rows = []
        for i in range(code.shape[0]):
            key = self._row_keys[i] if i < len(self._row_keys) else None
            buf = self._buffers.get(key, [])
            if not buf:
                # graph-connected zero: retry 0 / fresh scene -> no pull
                rows.append(x0_hat[i].sum() * 0.0)
                continue
            Z = torch.stack(buf).to(mu.device, mu.dtype)              # (N,16)
            spread = Z.std(dim=0)                                     # (16,)
            h2 = torch.maximum((self.h_scale * sig) ** 2,
                               (self.spread_c * spread) ** 2)         # (16,)
            d2 = ((code[i][None, :] - Z) ** 2 / (2.0 * h2[None, :])).sum(-1)
            kde = torch.exp(-d2).mean() + self.kde_eps
            # MINIMIZE log-kde == push the candidate's code AWAY from
            # everything already executed on this scene.
            rows.append(torch.log(kde))
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)


class KLCostPlanner(ScoutPlanner):
    """方案三: maximize KL(q(z|s̄,a) ‖ q(z|s̄,a^DP)) (capped at κ) -- move the
    code distribution of the chosen action away from the policy's own
    unguided intent for this chunk, in the encoder's own units.

    Renamed from AtypicalCostPlanner (2026-09-04, user order): this is the
    base of the KL-cost family. It owns ALL of phase 1 -- the per-chunk
    intent anchor (select_z), the merged single-backward :meth:`guided_step`
    consumed by policy.py's injection line, and the eta_tilde climb
    normalization (:attr:`eta_dimless`). OrbitCostPlanner subclasses it and
    adds ONLY the phase-2 constrained update, so "atypical vs orbit"
    decomposes exactly into "phase 1 only" vs "phase 1 + phase 2" with
    identical parameters and structure on the shared part.
    """

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0, eta_dimless: bool = False):
        super().__init__(scout_vib, bridge=bridge, z=None, obs_adapter=obs_adapter)
        self.cap = float(cap)
        # eta-dimless mode (2026-09-02 orbit-hparam-dev; moved into the base
        # 2026-09-04 so atypical and orbit share it): normalize the climb
        # gradient by the LIVE-CLIMB MEAN per-row gradient norm before the
        # policy's guidance_scale multiplies it. The injected climb becomes
        #   eta_tilde * sqrt(1-abar_t) * (g / g_med)
        # i.e. a fixed ACTION-SPACE displacement per step, so one eta_tilde
        # transfers across tasks whose VIB gradient scales differ (tool_hang
        # needed eta=12 vs 3.0 on square/can -- a per-task hand calibration
        # the live-climb mean absorbs automatically). Scale conversion on a
        # given task/ckpt: eta_tilde = eta_legacy * <g_med> measured on data
        # (per-step exact: eta_tilde = eta * g_med(step) reproduces the raw
        # injection exactly; a FIXED eta_tilde matches it to the extent g_med
        # is stable across steps -- the [kl-telemetry] mean_g_med print is
        # the probe surface). Telemetry: the policy's mean_inject lands at
        # the legacy value when eta_tilde = eta_legacy * g_med. OFF
        # (default) = bit-identical legacy injection. Phase 2 (orbit
        # subclass) is untouched: the Newton term carries its own
        # ||grad||^2 normalization and the tangent noise is sigma-scaled,
        # both already dimensionless w.r.t. the gradient scale.
        self.eta_dimless = bool(eta_dimless)
        # rows within [cap - _norm_band, cap) are excluded from the g_med
        # divisor -- the orbit subclass sets its phase-2 handover delta
        # (those rows swap the climb for the constrained update); the base
        # keeps 0.0 so EVERY injecting row counts.
        self._norm_band = 0.0
        self._base_mu: List[Optional[torch.Tensor]] = []
        self._base_lv: List[Optional[torch.Tensor]] = []
        # eta-dimless: last divisor + running mean (debug/probe surface).
        # _kl_calls drives the [kl-telemetry] print tick (base class only --
        # the orbit subclass prints its own merged telemetry line).
        self._last_g_med: Optional[torch.Tensor] = None
        self._gmed_acc: Optional[torch.Tensor] = None
        self._kl_calls = 0

    def set_row_context(self, init_ids: Sequence):
        pass    # rows identified positionally per chunk; nothing to persist

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Store the per-row baseline (μ⁰, σ⁰²) from the unguided x̂₀."""
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t, a)
        self._base_mu = [m.detach() for m in mu]
        self._base_lv = [v.detach() for v in logvar]
        return None

    def _encode_and_row_losses(self, x0_hat: torch.Tensor,
                               current_obs=None):
        """Shared core: ONE vib_enc forward -> (mu [grad-carrying], per-row
        capped-KL losses list). Vectorized (2026-09-01, 方案二): the per-row
        python loop cost ~6 kernel launches x B per call on the guided hot
        path; the batched form via :func:`_kl_rows` is math-identical
        (elementwise per element; the ``style_dim`` last-dim reduction
        replaces B per-row ``.sum()`` calls), and ``unbind`` keeps the
        historical 0-dim-tensor-per-row list contract. Clamp/cap semantics
        unchanged: the row VALUE caps at ``-cap``, the row GRADIENT is
        masked to zero at/above cap (torch clamp(max) backward)."""
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        kl = _kl_rows(mu, logvar, self._base_mu, self._base_lv, x0_hat)
        return mu, list((-torch.clamp(kl, max=self.cap)).unbind(0))

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        _, rows = self._encode_and_row_losses(x0_hat, current_obs)
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)

    # -- phase-1 core (shared verbatim with the orbit subclass) ------------ #
    def _kl_backward(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                     current_obs=None):
        """ONE encoder forward + ONE summed backward -> per-row UNCAPPED
        ``(kl, g)`` -- the shared graph that every guidance consumer (capped
        climb, orbit phase 2) differentiates through. ``g`` is the gradient
        of ``kl.sum()`` w.r.t. the trajectory; rows are block-diagonal, so
        row slices of ``g`` are the per-row gradients of the summed
        backward."""
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        kl = _kl_rows(mu, logvar, self._base_mu, self._base_lv, x0_hat)
        g = torch.autograd.grad(kl.sum(), trajectory)[0]
        return kl, g

    def _climb_gradient(self, kl: torch.Tensor, g: torch.Tensor):
        """The capped climb gradient (phase 1), optionally eta_tilde-
        normalized. Equivalence with ``-autograd.grad(compute_loss(...,
        reduction="sum"))``: the cap's gradient mask is ``1[KL <= kappa]``
        (torch clamp(max) backward PASSES gradient at input == max --
        empirically confirmed, review P1-1), and applying it AFTER the
        uncapped backward is exact because rows are block-diagonal -- a
        row's vjp never mixes with other rows, so a row at/below kappa
        keeps its full uncapped gradient (bitwise: ``where`` returns ``g``
        verbatim on the selected branch), and a row above kappa gets exact
        zeros (the old zero-upstream backward; ``where`` additionally
        avoids the ``0 * inf = NaN`` pathology of a post-multiply)."""
        cap_mask = (kl.detach() <= float(self.cap)).view(
            -1, *([1] * (g.dim() - 1)))
        cond_grad = torch.where(cap_mask, g, torch.zeros_like(g))
        if self.eta_dimless:
            # eta-dimless: divide by the MEAN per-row gradient norm over the
            # LIVE CLIMB rows (detached scalar; consumes no RNG, so streams
            # stay aligned with the legacy path). Guards (review 2026-09-02 +
            # 20k-call can telemetry: an all-batch median let the shell rows
            # -- orbit phase 2, LARGE norms -- inflate the divisor and starve
            # the climb 3x on can where p2=58%):
            #   * live-only: the divisor must describe the rows that
            #     actually inject the climb. The base counts every uncapped
            #     row (kl < cap); the orbit subclass additionally excludes
            #     its phase-2 handover band via _norm_band = orbit_delta
            #     (rows in [cap-delta, cap) swap the climb for the
            #     constrained update -- policy's _keep zeroes them).
            #   * roundoff exclusion: rows with norm <= 1e-4 (at-anchor
            #     first-step rows: KL==0 exactly, gradients ~1e-8 roundoff)
            #     are dropped from the MEAN as well as floored in the
            #     division -- otherwise they drag the divisor down and
            #     re-amplify noise.
            #   * NaN containment: map NaN row-norms to 0 so they drop out
            #     of the weighted mean; if EVERY live row is NaN the count
            #     clamps to 1 and g_med -> 0 -> floor -> zero climb (safe
            #     no-injection direction), never batch-wide NaN.
            #   * MEAN, not median: after normalization the per-row norm
            #     ||g_i||/mean is bounded by B (mean >= ||g_i||/B), so a
            #     single finite outlier row can only cause batch-wide
            #     UNDER-injection (safe: the DP prior dominates, and the
            #     mean_g_med telemetry makes it visible). A median divisor
            #     would allow unbounded over-injection of rows below it.
            # Real working-point row norms are O(0.1) (square/can chain
            # telemetry), three orders above the 1e-4 floor. ray_rotate
            # (orbit subclass) runs AFTER this in policy.py and preserves
            # per-row norms, so the normalized semantics survive the ray
            # mode. Calibration note (review P2): injected magnitudes are
            # batch-coupled through this statistic even though
            # directions/RNG stay row-independent.
            row_norms = g.detach().flatten(1).norm(dim=1)          # (B,)
            row_norms = torch.nan_to_num(row_norms, nan=0.0,
                                         posinf=0.0, neginf=0.0)
            live = ((kl.detach() < float(self.cap) - self._norm_band)
                    & (row_norms > 1e-4)).to(g.dtype)              # (B,)
            g_med = (row_norms * live).sum() / live.sum().clamp(min=1.0)
            cond_grad = cond_grad / g_med.clamp(min=1e-4)
            # Per-row cap at 3x nominal dose (2026-09-02, telemetry round):
            # the divisor describes LIVE CLIMB rows (small norms); the
            # orbit subclass's handover-band rows [cap-delta, cap) keep
            # nonzero cond_grad with LARGER norms and would normalize to
            # 10-50x nominal -- they do not inject (policy's _keep zeroes
            # them) but they dominate the injection telemetry and any
            # future consumer (ray_rotate). Clamp every normalized row to
            # <= 3x so the statistic, the telemetry and the (possible)
            # injection all stay in the eta_tilde band; normal rows (ratio
            # ~1) are untouched. Bit-identity of the OFF path is
            # unaffected.
            row_ratio = cond_grad.detach().flatten(1).norm(dim=1)
            row_scale = (3.0 / row_ratio.clamp(min=1e-12)).clamp(max=1.0)
            cond_grad = cond_grad * row_scale.view(-1, *([1] * (g.dim() - 1)))
            self._last_g_med = g_med.clamp(min=1e-4).detach()
            self._gmed_acc = (self._last_g_med
                              if self._gmed_acc is None
                              or self._gmed_acc.device != self._last_g_med.device
                              else self._gmed_acc + self._last_g_med)
        return cond_grad

    def guided_step(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                    current_obs=None, noise_scale: float = 1.0):
        """The planner-level merged guided step consumed by policy.py's
        injection line (duck-typed hook). Phase-1 only: ONE encoder forward
        + ONE backward supply the capped climb gradient and the detached
        per-row cost. Returns ``(cond_grad, None, None, row_losses)`` --
        the ``None``s mark "no phase-2 displacement", and policy.py uses
        the plain injection line for this planner.
        ``noise_scale`` is accepted for interface parity with the orbit
        subclass's override (the pure climb carries no sigma)."""
        kl, g = self._kl_backward(trajectory, x0_hat, current_obs)
        cond_grad = self._climb_gradient(kl, g)
        row_losses = -torch.clamp(kl.detach(), max=float(self.cap))
        if self.eta_dimless:
            self._kl_calls += 1
            if self._kl_calls % 2500 == 0 and self._gmed_acc is not None:
                print(f"[kl-telemetry] calls={self._kl_calls} "
                      f"mean_g_med={float(self._gmed_acc) / self._kl_calls:.4g}",
                      flush=True)
        return cond_grad, None, None, row_losses


class ShellTargetCostPlanner(ScoutPlanner):
    """方案A (user 2026-08-27): per-retry random target posterior on the
    kappa-shell of the DP intent -- SOE's wide-distribution spray transplanted
    into cost form.  Each retry (scene i, try k) draws a FIXED random unit
    direction u(i,k) (deterministic from shell_seed, frozen across the whole
    retry = trajectory-level coherence, SOE's z-per-trajectory analog); per
    chunk the intent baseline (mu^0, sigma^0^2) is captured by the select_z
    hook exactly as in 方案三, and the cost pulls the candidate posterior
    toward

        q* = N(mu^0 + sqrt(2*kappa) * sigma^0 * u,  sigma^0^2)

    which sits exactly kappa nats from the intent: KL(q*||q^0) = kappa*||u||^2
    = kappa for unit u.  N retries = N differently-tilted distributions, one
    sample each (instead of 方案三's ONE tilted distribution sampled N times --
    the narrow-cone critique).  cost = KL(q_a||q*) is a quadratic well centered
    at q*: the guided density p_DP * exp(-cost) only REWEIGHTS (exp(-cost) <=
    1, never amplifies above the DP prior, unlike 方案三's exp(+min(KL,kappa))
    boost), and the variance term pins sigma_a to sigma^0 (sigma-escape
    closed)."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 shell_kappa: float = 2.5, shell_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, z=None, obs_adapter=obs_adapter)
        self.shell_kappa = float(shell_kappa)
        self.shell_seed = int(shell_seed)
        self._base_mu: List[Optional[torch.Tensor]] = []
        self._base_lv: List[Optional[torch.Tensor]] = []
        self._row_jobs: List = []      # (init_idx, try_idx) per batch row
        self._u_cache: dict = {}       # (init_idx, try_idx) -> unit u tensor

    # -- context plumbing (rollout_vec._replan) ---------------------------- #
    def set_row_context(self, init_ids: Sequence):
        pass    # superseded by set_row_jobs (kept: the hook fires regardless)

    def set_row_jobs(self, jobs: Sequence):
        """Full job tuples (state, init_idx, try_idx) -- u is keyed by
        (init_idx, try_idx), so retries may run in parallel (no job gate)."""
        self._row_jobs = [(j[1], j[2]) for j in jobs]

    def _u_for(self, key, device, dtype) -> torch.Tensor:
        u = self._u_cache.get(key)
        if u is None:
            rng = np.random.default_rng([self.shell_seed,
                                         int(key[0]), int(key[1])])
            v = rng.standard_normal(int(self.scout_vib.style_dim))
            n = float(np.linalg.norm(v))
            u = torch.as_tensor(v / (n if n > 0.0 else 1.0),
                                dtype=torch.float32)
            self._u_cache[key] = u
        return u.to(device=device, dtype=dtype)

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Per-chunk intent baseline (mu^0, sigma^0^2) from the unguided x̂₀
        (same hook and semantics as KLCostPlanner)."""
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t, a)
        self._base_mu = [m.detach() for m in mu]
        self._base_lv = [v.detach() for v in logvar]
        return None

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows = []
        for i in range(mu.shape[0]):
            if i >= len(self._base_mu) or self._base_mu[i] is None:
                rows.append(x0_hat[i].sum() * 0.0)
                continue
            key = (self._row_jobs[i] if i < len(self._row_jobs)
                   else (None, None))
            if key[0] is None:
                rows.append(x0_hat[i].sum() * 0.0)
                continue
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var0 = torch.exp(lv0)
            sig0 = torch.exp(0.5 * lv0)
            u = self._u_for(key, mu.device, mu.dtype)
            target_mu = m0 + (2.0 * self.shell_kappa) ** 0.5 * sig0 * u
            var = torch.exp(logvar[i])
            kl = 0.5 * (((mu[i] - target_mu) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            rows.append(kl)     # 0 at the target; quadratic well around it
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)


class ComboCostPlanner(ScoutPlanner):
    """方案二+方案三 (reflection #3): summed cost -- KDE repulsion from the
    scene's EXECUTED codes PLUS the capped KL bonus away from the policy's
    own unguided intent.  The two mechanisms cracked disjoint never-rescued
    scenes on the same 39-failed set (novelty: 95; atypical: 75/83), so both
    pushes are kept at their calibrated strengths (h_scale as passed, cap as
    passed) and simply added; the DP prior remains the only trust region."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 h_scale: float = 0.5, spread_c: float = 1.0,
                 sample_z: bool = False, cap: float = 2.5,
                 nov_weight: float = 1.0, att_weight: float = 1.0):
        super().__init__(scout_vib, bridge=bridge, z=None, obs_adapter=obs_adapter)
        self._nov = NoveltyCostPlanner(
            scout_vib, bridge=bridge, obs_adapter=obs_adapter,
            h_scale=h_scale, spread_c=spread_c, sample_z=sample_z)
        self._att = KLCostPlanner(
            scout_vib, bridge=bridge, obs_adapter=obs_adapter, cap=cap)
        self.nov_weight = float(nov_weight)
        self.att_weight = float(att_weight)

    # lifecycle calls fan out to the mechanism that consumes them
    def set_row_context(self, init_ids: Sequence):
        self._nov.set_row_context(init_ids)

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        self._nov.select_z(x0_hat, current_obs)
        self._att.select_z(x0_hat, current_obs)
        return None

    def set_current_obs(self, current_obs):
        self._nov.set_current_obs(current_obs)
        self._att.set_current_obs(current_obs)

    def encode_executed(self, obs, chunk_np) -> torch.Tensor:
        return self._nov.encode_executed(obs, chunk_np)

    def on_try_done(self, init_idx, codes: Sequence[torch.Tensor]):
        self._nov.on_try_done(init_idx, codes)

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        return (self.nov_weight
                * self._nov.compute_loss(x0_hat, current_obs, reduction)
                + self.att_weight
                * self._att.compute_loss(x0_hat, current_obs, reduction))


# Backward-compat alias (renamed 2026-09-04, user order): historical
# probes/notebooks/older branches import the old name.
AtypicalCostPlanner = KLCostPlanner
