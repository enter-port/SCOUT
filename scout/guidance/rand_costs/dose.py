"""rand_dose -- reference idea + registry template (2026-08-27).

Keep 方案三's J^T*Lambda*J-informed escape direction (today's diagnosis: the
cone IS the reachable direction set, that's why it is efficient) but
randomize the DOSE per retry: each retry (scene i, try k) draws a fixed
multiplier w(i,k) from a log-grid, so the N retries of a scene probe N
different DEPTHS along the high-gain direction instead of N samples at one
depth.  Diversity in depth rather than direction.

    cost_k(x0_hat) = w(i,k) * [ -min(KL(q_a || q^0), kappa) ]

w = exp(uniform(log w_lo, log w_hi)), seeded by (rand_seed, i, k); rows
without job context fall back to w = 1 (plain 方案三).

ek knobs: rand_w_lo (default 0.3), rand_w_hi (default 3.0), rand_seed.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward

NAME = "dose"


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    return DoseCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        w_lo=float(ek.get("rand_w_lo", 0.3)),
        w_hi=float(ek.get("rand_w_hi", 3.0)),
        rand_seed=int(ek.get("rand_seed", 42)),
        cap=float(ek.get("atypical_cap", 2.5)),
    )


class DoseCostPlanner(AtypicalCostPlanner):
    """方案三 cost x per-retry deterministic random dose multiplier."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 w_lo: float = 0.3, w_hi: float = 3.0,
                 rand_seed: int = 42, cap: float = 2.5):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap)
        self.w_lo = float(w_lo)
        self.w_hi = float(w_hi)
        self.rand_seed = int(rand_seed)
        self._row_jobs: List = []

    # hooks ------------------------------------------------------------- #
    def set_row_jobs(self, jobs: Sequence):
        self._row_jobs = [(j[1], j[2]) for j in jobs]

    def _w_for(self, key) -> float:
        rng = np.random.default_rng([self.rand_seed, int(key[0]), int(key[1])])
        return float(np.exp(rng.uniform(np.log(self.w_lo), np.log(self.w_hi))))

    # cost -------------------------------------------------------------- #
    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        # AtypicalCostPlanner's per-row cost, each row scaled by its own
        # per-retry dose multiplier w(i,k)
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows = []
        for i in range(mu.shape[0]):
            if i >= len(self._base_mu) or self._base_mu[i] is None:
                rows.append(x0_hat[i].sum() * 0.0)
                continue
            w = 1.0
            if i < len(self._row_jobs):
                w = self._w_for(self._row_jobs[i])
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            rows.append(w * -torch.clamp(kl, max=self.cap))
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
