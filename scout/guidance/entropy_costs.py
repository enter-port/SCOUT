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

方案三 AtypicalCostPlanner (unchanged from v1): cost =
-min(KL(q(z|s̄,a) ‖ q(z|s̄,a^DP)), κ) with the per-chunk unguided-intent
baseline (μ⁰, σ⁰²) captured by the select_z hook.

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


class AtypicalCostPlanner(ScoutPlanner):
    """方案三: maximize KL(q(z|s̄,a) ‖ q(z|s̄,a^DP)) (capped at κ) -- move the
    code distribution of the chosen action away from the policy's own
    unguarded intent for this chunk, in the encoder's own units."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0):
        super().__init__(scout_vib, bridge=bridge, z=None, obs_adapter=obs_adapter)
        self.cap = float(cap)
        self._base_mu: List[Optional[torch.Tensor]] = []
        self._base_lv: List[Optional[torch.Tensor]] = []

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
        (same hook and semantics as AtypicalCostPlanner)."""
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
        self._att = AtypicalCostPlanner(
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
