"""rand_entropyseek -- proposal 3, sigma-axis cost add-on (2026-08-28).

Released by diagnosis 2 (2026-08-27): the exact J_sigma = dlogvar/da of the
frozen VIB encoder is full-rank (16/16), FLAT (lambda2/lambda1 = 0.50-0.76,
lambda10/lambda1 ~ 0.15-0.20 -- the flattest operator measured in this
campaign) and its gain is ~0.5x J_mu's top singular value (NOT <<, refuting
LESSONS proposal-3's predicted death "sigma is decided by s_bar"); a
workpoint-scale action displacement moves ln sigma by ~1.1 nats, so
kappa_sigma <= 1.25 is reachable, 2.5 marginal.  Every dead line (PlanA /
dose / mjitter / rshell / smask) pinned sigma = sigma^0 in its reference --
the sigma head is the only tensor axis of the encoder output that NO arm
has ever touched.

Cost (user 2026-08-28, two knobs, minimal form):

    C_k(a) = -min( relu( sum_i m_{k,i} * (ln sigma_i(s_bar,a) - ln sigma0_i) ),
                   kappa_sigma )
             - w_A * min( KL(q_a || q^0), kappa )

  * m_k = Bernoulli(rho) mask, rng = default_rng([rand_seed, init, try]),
    frozen for the WHOLE retry (trajectory-level coherence, dose.py
    convention); ONE redraw if the first draw is all-zero (anti-extinction);
  * relu([.]_+ on the masked SUM): only REWARDS sigma growing -- encoder
    less certain = DIAYN dual (discriminator uncertain = behavior summary
    uncommitted = unexplored); sigma shrinking is not punished (floors at 0);
  * anchor term = byte-identical 方案三 (w_A = 1 default): the KL-cone escape
    stays the floor while the sigma term spends a SEPARATE kappa_sigma
    budget.  kappa accounting: at a = a^0 the sigma floor is exactly 0
    (ln(sigma/sigma0) = 0), so no mjitter-b>=1-style silent shutdown.

rho = 0 (or rows without job context) degenerates BYTE-FOR-BYTE to 方案三
at w_A = 1 -- the free regression anchor (LESSONS 7.7).

ek knobs: rand_rho (default 0.5), rand_kappa_sigma (1.25), rand_anchor_w
(1.0), rand_cap (kappa of the anchor KL; default = atypical_cap = 2.5),
rand_seed (42).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward

NAME = "entropyseek"


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    return EntropySeekCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        rho=float(ek.get("rand_rho", 0.5)),
        kappa_sigma=float(ek.get("rand_kappa_sigma", 1.25)),
        anchor_w=float(ek.get("rand_anchor_w", 1.0)),
        cap=float(ek.get("rand_cap", ek.get("atypical_cap", 2.5))),
        rand_seed=int(ek.get("rand_seed", 42)),
    )


class EntropySeekCostPlanner(AtypicalCostPlanner):
    """方案三 anchor + capped relu reward for growing ln sigma on a
    per-retry random Bernoulli(rho) subset of the style dims."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 rho: float = 0.5, kappa_sigma: float = 1.25,
                 anchor_w: float = 1.0, cap: float = 2.5,
                 rand_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap)
        self.rho = float(rho)
        self.kappa_sigma = float(kappa_sigma)
        self.anchor_w = float(anchor_w)
        self.rand_seed = int(rand_seed)
        self._row_jobs: List = []      # (init_idx, try_idx) per batch row
        self._mask_cache: dict = {}    # (init_idx, try_idx) -> 0/1 tensor

    # hooks ------------------------------------------------------------- #
    def set_row_jobs(self, jobs: Sequence):
        self._row_jobs = [(j[1], j[2]) for j in jobs]

    def _mask_for(self, key, device, dtype) -> torch.Tensor:
        """Bernoulli(rho) mask for this retry; deterministic in
        (rand_seed, init_idx, try_idx); one redraw if all-zero."""
        m = self._mask_cache.get(key)
        if m is None:
            dz = int(self.scout_vib.style_dim)
            rng = np.random.default_rng([self.rand_seed,
                                         int(key[0]), int(key[1])])
            draw = rng.random(dz) < self.rho
            if not draw.any():                      # anti-extinction redraw
                draw = rng.random(dz) < self.rho
            m = torch.as_tensor(draw.astype(np.float32))
            self._mask_cache[key] = m
        return m.to(device=device, dtype=dtype)

    # cost -------------------------------------------------------------- #
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
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            anchor = -torch.clamp(kl, max=self.cap)
            if self.anchor_w != 1.0:                # keep w_A=1 byte-exact
                anchor = self.anchor_w * anchor
            if self.rho > 0.0 and key[0] is not None:
                m = self._mask_for(key, mu.device, mu.dtype)
                r = (m * (logvar[i] - lv0)).sum()
                sig = -torch.clamp(torch.relu(r), max=self.kappa_sigma)
            else:
                sig = x0_hat[i].sum() * 0.0         # graph-connected zero
            rows.append(sig + anchor)
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
