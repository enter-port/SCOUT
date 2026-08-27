"""rand_mjitter -- 方案D random metric rotation (geometry.md D, 2026-08-27).

方案三 keeps ONE fixed escape metric -- the intent posterior's own precision
Lambda^0 = diag(1/sigma0^2) -- so every retry pushes along the same narrow
J^T Lambda^0 J cone (the 方案A diagnosis).  方案D does NOT sample a z
direction (that dies to projection); instead it ROTATES THE METRIC per
retry: each retry (scene i, try k) draws xi ~ N(0, I_Dz)
(rng = default_rng([rand_seed, i, k]), frozen for the whole retry) and
builds a random reference PRECISION

    1/tau^2_{k,i} = s_{k,i} / sigma0^2,      s_k = Dz * softmax(b * xi_k)

(sum(s) = Dz: total precision conserved -- pure rotation/redistribution of
the intent posterior's precision across dims, no global tightening).  The
cost is 方案三's capped KL with the reference N(mu^0, tau^2_k):

    cost_k(x0_hat) = -min( KL( q(z|s_bar,a) || N(mu^0, tau^2_k) ), kappa )

    KL = 0.5 * sum_j [ (mu_j-mu^0_j)^2 / tau^2_j + sigma^2_j / tau^2_j - 1
                       - ln(sigma^2_j / tau^2_j) ]

Mechanics (geometry.md): the mean-term force is always J^T Lambda_k (mu-mu^0)
-- it lives in the row space of J no matter the metric, so there is ZERO
projection recovery; what rotates is the weighting, i.e. the top eigendir of
J^T Lambda_k J differs per retry = sampling WITHOUT replacement from the
family of natural directions.  The variance term additionally pulls sigma^2
toward the random tau^2 (a random variance target per dim).

b = 0 degenerates BYTE-FOR-BYTE to 方案三 (softmax(0) = uniform -> s = 1
-> tau^2 = sigma0^2); the b==0 shortcut below never even touches the rng.
Rows without job context fall back to s = 1 (plain 方案三), like dose.py.

ek knobs: rand_b (default 1.0), rand_cap (kappa, default 2.5), rand_seed
(default 42).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward

NAME = "mjitter"


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    return MJitterCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        rand_b=float(ek.get("rand_b", 1.0)),
        cap=float(ek.get("rand_cap", 2.5)),
        rand_seed=int(ek.get("rand_seed", 42)),
    )


class MJitterCostPlanner(AtypicalCostPlanner):
    """方案D: 方案三's capped KL under a per-retry rotated reference
    precision 1/tau^2 = s / sigma0^2, s = Dz*softmax(b*xi(i,k))."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 rand_b: float = 1.0, cap: float = 2.5,
                 rand_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap)
        self.rand_b = float(rand_b)
        self.rand_seed = int(rand_seed)
        self._row_jobs: List = []      # (init_idx, try_idx) per batch row
        self._s_cache: dict = {}       # (init_idx, try_idx) -> (s, ln_s)

    # hooks ------------------------------------------------------------- #
    def set_row_jobs(self, jobs: Sequence):
        self._row_jobs = [(j[1], j[2]) for j in jobs]

    def _s_for(self, key, device, dtype):
        """Per-retry precision rotation s (and ln s, for the KL log term).

        Returns (s, ln_s) with s = Dz*softmax(b*xi), sum(s) = Dz; identity
        (ones / zeros) when b == 0 or the row carries no job context."""
        dim = int(self.scout_vib.style_dim)
        if key is None or key[0] is None or self.rand_b == 0.0:
            # identity rotation -- tau^2 == sigma0^2 (byte-exact 方案三)
            return (torch.ones(dim, device=device, dtype=dtype),
                    torch.zeros(dim, device=device, dtype=dtype))
        hit = self._s_cache.get(key)
        if hit is None:
            rng = np.random.default_rng([self.rand_seed,
                                         int(key[0]), int(key[1])])
            y = self.rand_b * rng.standard_normal(dim)
            # stable log-softmax; ln s = ln(dim) + y - logsumexp(y)
            m = float(np.max(y))
            ln_s = np.log(dim) + y - (m + np.log(np.exp(y - m).sum()))
            s = np.exp(ln_s)
            hit = (torch.as_tensor(s, dtype=torch.float32),
                   torch.as_tensor(ln_s, dtype=torch.float32))
            self._s_cache[key] = hit
        return tuple(t.to(device=device, dtype=dtype) for t in hit)

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
            s, ln_s = self._s_for(key, mu.device, mu.dtype)
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            # tau^2 = var0 / s  ->  x / tau^2 == s * x / var0,
            # ln(sigma^2 / tau^2) == (logvar - lv0) + ln_s
            kl = 0.5 * ((s * ((mu[i] - m0) ** 2) / var0)
                        + (s * var / var0) - 1.0
                        - (logvar[i] - lv0) - ln_s).sum()
            rows.append(-torch.clamp(kl, max=self.cap))
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
