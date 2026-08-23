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
        conditioned on (single env, unbatched obs dict)."""
        obs_es = (self.obs_adapter({k: np.asarray(v)[None] for k, v in obs.items()})
                  if self.obs_adapter is not None
                  else {k: np.asarray(v)[None] for k, v in obs.items()})
        a_flat = torch.as_tensor(
            np.asarray(chunk_np, dtype=np.float32).reshape(1, -1))
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
