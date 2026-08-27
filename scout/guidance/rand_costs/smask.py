"""rand_smask -- 方案F random temporal gating (geometry.md F, 2026-08-27).

方案三 guides EVERY chunk of every retry with the same capped-KL escape cost.
方案F keeps that cost VERBATIM (byte-identical per-row implementation, so the
J^T Lambda^0 J cone stays -- today's diagnosis says the cone IS the reachable
direction set) and randomizes only WHEN it is applied: a chunk-level 0/1
schedule w_k(c) multiplies the cost,

    cost_k(x0_hat; c) = w_k(c) * [ -min( KL(q(z|s_bar,a) || q^0), kappa ) ]

with c = the chunk index of retry (scene i, try k) within its episode
(counted by set_row_jobs: 0 at the first replan of the row).  w is a plain
float multiplier applied OUTSIDE the autograd graph, so w=0 rows yield a
graph-connected exact 0 (zero force, the chunk is pure DP) and w=1 rows are
BIT-EQUAL to 方案三 (cost AND gradient, both reductions).  Diversity enters
through the schedule, not through the direction -- and "front" additionally
tests the never-wall hypothesis (hand control back to the DP near success).

Schedule families (rand_family):
  * "window": w = 1_{c0 <= c < c0 + L}.  The 10 retries of a scene get
    NON-OVERLAPPING windows from the grid {0, L, 2L, ...} (spacing == window
    length -> disjoint by construction): the grid is shuffled by
    (rand_seed, init_idx) and retry k takes the k-th shuffled point, so each
    retry explores the cost in a DIFFERENT part of the episode.  Retries whose
    window starts beyond the episode's chunk count (L*10 > ceil(horizon/8)
    for L=8: c0 in {40..72} vs ~38 chunks) are designed-unguided retries.
    rand_win_c0 is OVERRIDDEN by this designed spread.
  * "front": w = 1_{c < rand_win_c0} -- guide the early episode, release the
    DP for the endgame (near success the unguided DP prior should take over).
  * "bern":  w ~ Bernoulli(rho) iid per chunk, seeded
    (rand_seed, init_idx, try_idx, chunk).

Rows without job context fall back to w = 1 (plain 方案三), like dose.py.

ek knobs: rand_family (default "window"), rand_win_len (L, default 8),
rand_win_c0 (front/window start; default 16, window family overrides),
rand_bern_rho (default 0.7), rand_cap (kappa, default = atypical_cap = 2.5),
rand_seed (default 42).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward

NAME = "smask"

# number of retries the window grid is designed for (protocol try-times = 10)
N_GRID = 10


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    return SMaskCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        family=str(ek.get("rand_family", "window")),
        win_len=int(float(ek.get("rand_win_len", 8))),
        win_c0=int(float(ek.get("rand_win_c0", 16))),
        bern_rho=float(ek.get("rand_bern_rho", 0.7)),
        cap=float(ek.get("rand_cap", ek.get("atypical_cap", 2.5))),
        rand_seed=int(float(ek.get("rand_seed", 42))),
    )


class SMaskCostPlanner(AtypicalCostPlanner):
    """方案三 cost x per-chunk 0/1 temporal schedule (cost body untouched)."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 family: str = "window", win_len: int = 8, win_c0: int = 16,
                 bern_rho: float = 0.7, cap: float = 2.5,
                 rand_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap)
        if family not in ("window", "front", "bern"):
            raise ValueError(f"rand_family must be window|front|bern, "
                             f"got {family!r}")
        self.family = str(family)
        self.win_len = int(win_len)
        self.win_c0 = int(win_c0)
        self.bern_rho = float(bern_rho)
        self.rand_seed = int(rand_seed)
        # (init_idx, try_idx) -> number of set_row_jobs appearances so far
        self._chunk_seen: Dict[Tuple[int, int], int] = {}
        self._row_keys: List[Optional[Tuple[int, int]]] = []
        self._row_c: List[int] = []

    # hooks ------------------------------------------------------------- #
    def set_row_jobs(self, jobs: Sequence):
        """Per replan batch: jobs = [(state, init_idx, try_idx), ...].

        Chunk counting (user spec): c = (number of times this row's
        (init, try) key has appeared in set_row_jobs) - 1 -- chunk 0 is the
        row's FIRST replan, which fires here for the first time right after
        the slot's launch.
        """
        self._row_keys = []
        self._row_c = []
        for j in jobs:
            if j is None or j[1] is None or j[2] is None:
                key = None
                c = 0
            else:
                key = (int(j[1]), int(j[2]))
                self._chunk_seen[key] = self._chunk_seen.get(key, 0) + 1
                c = self._chunk_seen[key] - 1
            self._row_keys.append(key)
            self._row_c.append(c)

    # schedules --------------------------------------------------------- #
    def _window_c0(self, key: Tuple[int, int]) -> int:
        """Designed spread: shuffle the grid {0, L, ..., (N-1)L} by
        (rand_seed, init); retry k takes the k-th shuffled point.  Distinct
        grid points spaced by L -> retry windows never overlap."""
        grid = [g * self.win_len for g in range(N_GRID)]
        rng = np.random.default_rng([self.rand_seed, int(key[0]), 0])
        order = rng.permutation(len(grid))
        return int(grid[order[int(key[1]) % len(grid)]])

    def _omega_for(self, key, c: int) -> float:
        """The 0/1 schedule multiplier for chunk c of retry `key`."""
        if key is None:
            return 1.0                       # no job context -> plain 方案三
        if self.family == "window":
            c0 = self._window_c0(key)
            return 1.0 if c0 <= c < c0 + self.win_len else 0.0
        if self.family == "front":
            return 1.0 if c < self.win_c0 else 0.0
        # bern
        rng = np.random.default_rng(
            [self.rand_seed, int(key[0]), int(key[1]), int(c)])
        return 1.0 if rng.random() < self.bern_rho else 0.0

    # cost (AtypicalCostPlanner.compute_loss verbatim + w multiplier) ---- #
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
            key = self._row_keys[i] if i < len(self._row_keys) else None
            c = self._row_c[i] if i < len(self._row_c) else 0
            w = self._omega_for(key, c)
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            # w in {0.,1.} (python float, outside the graph): w=1 keeps the
            # row BIT-EQUAL to 方案三; w=0 keeps it graph-connected at exact 0.
            rows.append(w * -torch.clamp(kl, max=self.cap))
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
