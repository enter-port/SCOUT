"""Entropy-exploration costs (entropy-dev, user 2026-08-24 方案二/三).

Both planners REPLACE the SCOUT NLL cost inside the SAME guided-denoise
injection path (policy.py ``guided_conditional_sample``): the base DP stays
frozen and acts as the trust region (guided sampler ~ p_DP * exp(-cost)),
only ``compute_loss`` changes -- per the user's constraint that the dyn
model, the guidance mechanism, and "entropy as the core idea" stay fixed.

方案二 NoveltyCostPlanner -- trajectory-entropy exploration, greedy form.
  Objective (trajectory level): maximize the empirical entropy of the skill
  codes visited by the rollouts of a scene.  Standard count/KDE surrogate
  (Bellemare'16 pseudo-counts; Tang'17 #Exploration): per-chunk novelty
  bonus = -log p̂(code), where p̂ is a kernel density over the codes already
  visited ON THIS SCENE (across its retries AND the chunks within each
  retry -- retry j must differ from retries < j, and chunk k must differ
  from chunks < k).  As cost (to MINIMIZE):

      cost(a) = log[ (1/N) Σ_j exp(-Σ_i (μ_i(s̄,a)-z_j,i)² / (2 h_i²)) + ε ]

  with μ = encoder mean of the CANDIDATE action, z_j = visited codes,
  h_i = h_scale * σ̄_i (per-dim running average of the encoder σ -- the
  kernel lives in the encoder's own uncertainty units).  Rows whose scene
  buffer is still empty get cost 0 (no pull -> natural behavior first).

方案三 AtypicalCostPlanner -- the "difference of two entropies" done right.
  Pure entropy differences with a constant reference have identical
  gradients (the reference term does not depend on a); the meaningful
  two-distribution form is a KL with a per-chunk, state-conditional
  baseline: the encoder of the policy's OWN unguided action estimate at
  this chunk (the anchored baseline).  Maximizing the KL = maximize
  I(z; a | s) with the variational marginal q(z|s̄, a^DP):

      cost(a) = -min( D_KL( q(z|s̄,a) ‖ q(z|s̄,a^DP) ), κ )

  closed form over per-dim Gaussians (both μ- and σ-channels, normalized
  by the baseline σ -- state-adaptive metric).

Shared plumbing:
  * ``select_z(x0_hat, current_obs)`` is called by the policy at the FIRST
    guided denoise step of every chunk (the pre-existing expert-mode hook).
    For novelty it (a) commits the previous chunk's pending anchor code to
    that scene's buffer, (b) computes this chunk's anchor code
    μ(s̄, bridge(x̂₀^DP)) (+ a fixed reparam draw) and parks it as pending.
    For atypical it computes and stores the per-row baseline (μ⁰, σ⁰²).
    Both return None (no z-locking; the override of compute_loss ignores z).
  * ``set_row_context(init_ids)`` is called by rollout_vec._replan so each
    batch row knows which scene (init_idx) it belongs to; buffers are
    per-scene and accumulate across that scene's retries.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from scout.guidance.planner import ScoutPlanner


def _enc_forward(planner: ScoutPlanner, x0_hat: torch.Tensor) -> torch.Tensor:
    """bridge(x̂₀) -> flattened chunk -> raw action vector fed to the encoder."""
    per_step = x0_hat.shape[-1]
    chunk_dim = int(getattr(planner.scout_vib.vib_enc, "action_dim", per_step))
    n_steps = chunk_dim // per_step
    a = planner.bridge(x0_hat[:, :n_steps])
    return a.reshape(x0_hat.shape[0], chunk_dim)


class NoveltyCostPlanner(ScoutPlanner):
    """方案二: minimize KDE density of the candidate code in the scene's
    visited-code set == maximize per-scene empirical code entropy (greedy)."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 h_scale: float = 1.0, sample_z: bool = True,
                 kde_eps: float = 1e-4, ema: float = 0.99):
        super().__init__(scout_vib, bridge=bridge, z=None, obs_adapter=obs_adapter)
        self.h_scale = float(h_scale)
        self.sample_z = bool(sample_z)
        self.kde_eps = float(kde_eps)
        self.ema = float(ema)
        # per-scene visited-code buffers: init_idx -> list[(B=1-free) code rows]
        self._buffers: dict = {}
        # per-row pending anchor codes (committed on the NEXT select call)
        self._pending: List[Optional[torch.Tensor]] = []
        self._row_keys: List = []
        # per-row fixed reparam eps for this chunk (smooths the cost)
        self._row_eps: List[Optional[torch.Tensor]] = []
        # running per-dim σ̄ (encoder uncertainty units), init lazily
        self._sigma_bar: Optional[torch.Tensor] = None

    # -- context plumbing (called by rollout_vec._replan) ------------------ #
    def set_row_context(self, init_ids: Sequence):
        """Tell the planner which scene each batch row belongs to."""
        device = next(p for p in [self._sigma_bar] if p is not None).device \
            if self._sigma_bar is not None else None
        # commit any pending rows whose scene changed (new rollout started)
        for i, key in enumerate(init_ids):
            if i < len(self._pending) and self._pending[i] is not None:
                if i >= len(self._row_keys) or self._row_keys[i] != key:
                    self._commit(i)
        self._row_keys = list(init_ids)
        while len(self._pending) < len(init_ids):
            self._pending.append(None)
            self._row_eps.append(None)

    def _commit(self, i: int):
        code = self._pending[i]
        if code is not None:
            key = self._row_keys[i]
            self._buffers.setdefault(key, []).append(code.detach())
            self._pending[i] = None

    # -- per-chunk hook (policy calls this at the first guided step) ------- #
    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Anchor bookkeeping: commit last chunk's code, park this chunk's."""
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t, a)
            sigma = torch.exp(0.5 * logvar)
            self._update_sigma_bar(sigma)
            codes = mu + sigma * torch.randn_like(mu) if self.sample_z else mu
        B = codes.shape[0]
        for i in range(B):
            if i < len(self._pending) and self._pending[i] is not None:
                self._commit(i)
            while len(self._pending) <= i:
                self._pending.append(None)
                self._row_eps.append(None)
            self._pending[i] = codes[i].detach()
            self._row_eps[i] = torch.randn_like(codes[i]).detach()
        return None            # no z-locking; compute_loss ignores z

    def _update_sigma_bar(self, sigma: torch.Tensor):
        batch_mean = sigma.detach().mean(dim=0)          # (style_dim,)
        if self._sigma_bar is None:
            self._sigma_bar = batch_mean.clone()
        else:
            self._sigma_bar.mul_(self.ema).add_(batch_mean, alpha=1 - self.ema)

    # -- the cost ----------------------------------------------------------- #
    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        if self.sample_z:
            eps = torch.stack([e if e is not None else torch.zeros_like(mu[0])
                               for e in self._row_eps[:mu.shape[0]]]
                              ).to(mu.device, mu.dtype) if self._row_eps else None
            code = mu + torch.exp(0.5 * logvar) * (eps if eps is not None else 0)
        else:
            code = mu
        h2 = (self.h_scale * (self._sigma_bar.to(mu.device) if self._sigma_bar
                              is not None else torch.ones_like(mu[0]))) ** 2
        rows = []
        for i in range(code.shape[0]):
            key = self._row_keys[i] if i < len(self._row_keys) else None
            buf = self._buffers.get(key, [])
            if not buf:
                rows.append(x0_hat.new_zeros(()))
                continue
            Z = torch.stack(buf).to(mu.device, mu.dtype)              # (N,16)
            d2 = ((code[i][None, :] - Z) ** 2 / (2.0 * h2[None, :])).sum(-1)
            kde = torch.exp(-d2).mean() + self.kde_eps
            # MINIMIZE log-kde == push the candidate's code AWAY from the
            # visited set (the novelty bonus -log p̂ is a reward to MAXIMIZE;
            # as a minimized cost its sign flips to +log p̂).
            rows.append(torch.log(kde))
        nll = torch.stack(rows)                                       # (B,)
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

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows = []
        for i in range(mu.shape[0]):
            if i >= len(self._base_mu) or self._base_mu[i] is None:
                rows.append(x0_hat.new_zeros(()))
                continue
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            rows.append(-torch.clamp(kl, max=self.cap))
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
