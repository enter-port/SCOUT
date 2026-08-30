"""Particle guidance: parallel inter-particle repulsion (user 2026-08-30).

Idea doc: ``idea/particle_design.md`` (approved 2026-08-30, all 7 decision
points). Research frame: ``idea/escape_coverage_research.md`` -- the entropy
cost's guided density p* already spreads mass over ALL escape directions, but
the deterministic mode-seeking injection collapses a scene's K i.i.d. retries
onto ONE ridge of the J^T.Lambda.J cost landscape (narrow cone, PR 1.30).
This planner adds the missing ENSEMBLE term: while the K retries of one scene
are being generated IN PARALLEL (same replan batch -- the runner's group-locked
scheduling), each particle is pushed away from the OTHERS' current behaviour
codes. Design lineage: joint-particle repulsive potential during diffusion
sampling (Corso et al., "Particle Guidance", ICLR 2024), serialized-greedy
variant = NoveltyCostPlanner (repulsion from already-finalized retries -- a
different mechanism, kept separate).

Derivation chain (idea/particle_design.md §1):

  joint target  q({a_i}) ~ prod_i p_DP(a_i|s).exp(beta.cost(a_i)).psi,
  psi           = exp(-lambda * sum_{i<j} k(f(a_i), f(a_j)))   [Gibbs repulsion]
  k(d)          = exp(-d^2 / (2h^2))  on  d = ||mu_i - mu_j||_2  (encoder mean,
                  16-d -- same space the PR / d_act width metrics live in)
  per-particle force = the existing entropy-cost force (verbatim, trust region
                  unchanged) + lambda * grad of its own repulsion. Other
                  particles are DETACHED (Particle Guidance's frozen-others
                  step; autograd block-diagonality via the sum reduction).
  h             = pg_h_scale * median of the group's pairwise distances
                  (median heuristic, SVGD-standard; recomputed per replan so
                  "near" auto-adapts to the particles' current spread).
  pg_start      = denoise-step index (0-based, 100-step loop) at which the
                  repulsion switches ON -- the G1/G2/G3 = 0/50/90 timing
                  ablation (user 2026-08-30). Before it, loss is bit-identical
                  to AtypicalCostPlanner.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner
from scout.guidance.planner import ScoutPlanner


class ParticleCostPlanner(ScoutPlanner):
    """Entropy cost + within-scene parallel particle repulsion.

    The atypical part (per-chunk KL-to-own-intent, capped) is DELEGATED to an
    internal :class:`AtypicalCostPlanner` (same select_z anchor capture, same
    trust region). ``GROUP_LOCK = True`` tells ``rollout_vec`` to schedule a
    scene's retries as ONE group (launch together, replan together) so the
    repulsion sees all K particles of the scene in the same batch.
    """

    GROUP_LOCK = True    # rollout_vec.run(group_lock=...) reads this

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0,
                 pg_lambda: float = 1.0, pg_h_scale: float = 1.0,
                 pg_start: int = 0):
        super().__init__(scout_vib, bridge=bridge, z=None,
                         obs_adapter=obs_adapter)
        self._att = AtypicalCostPlanner(scout_vib, bridge=bridge,
                                        obs_adapter=obs_adapter, cap=cap)
        self.pg_lambda = float(pg_lambda)
        self.pg_h_scale = float(pg_h_scale)
        self.pg_start = int(pg_start)
        # denoise-step counter, set by policy.py's per-step callback. Defaults
        # to 0 (repulsion ON): direct compute_loss calls (tests / probes)
        # without the callback still see the repulsion at its G1 semantics;
        # the policy path always calls set_denoise_step first, so the gate is
        # exact there.
        self._denoise_n = 0
        # batch row -> init_idx (scene id), set per replan by rollout_vec
        self._row_keys: List = []
        # telemetry: repulsion magnitude (lambda-calibration smoke reads this)
        self._rep_calls = 0          # total compute_loss calls
        self._rep_on_calls = 0       # calls where the repulsion was active
        self._rep_row_sum = 0.0      # running sum of mean-per-row L_rep

    # -- context plumbing (rollout_vec._replan hooks) ---------------------- #
    def set_row_context(self, init_ids: Sequence):
        """Map each batch row to its scene (init_idx) -- repulsion pairs are
        ONLY same-scene row pairs; cross-scene rows never repel."""
        self._row_keys = list(init_ids)

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Per-chunk intent anchor (mu^0, sigma^0^2) -- delegated verbatim to
        the atypical planner (captured at the FIRST guided denoise step)."""
        return self._att.select_z(x0_hat, current_obs)

    def set_current_obs(self, current_obs):
        self._att.set_current_obs(current_obs)

    def set_denoise_step(self, n: int):
        """Per-step callback from policy.py's denoise loop (duck-typed like
        select_z). ``n`` is the 0-based loop index over scheduler.timesteps,
        i.e. the FIRST denoise step is n=0 (NOT the DDPM t value)."""
        self._denoise_n = int(n)

    # -- the cost ----------------------------------------------------------- #
    def _repulsion_rows(self, mu: torch.Tensor) -> torch.Tensor:
        """Per-row repulsion loss: sum_j k(||mu_i - mu_j||) over same-scene
        others j, with mu_j DETACHED (frozen others). Rows without same-scene
        partners contribute a graph-connected zero (single-particle group ->
        nothing to repel). Bandwidth per group: pg_h_scale * median of the
        group's pairwise mu distances (median heuristic; scalar, no grad)."""
        B = mu.shape[0]
        keys = [self._row_keys[i] if i < len(self._row_keys) else None
                for i in range(B)]
        mu_d = mu.detach()
        with torch.no_grad():
            dmat = torch.cdist(mu_d, mu_d)                  # (B, B), no grad
        rows: List[Optional[torch.Tensor]] = []
        for i in range(B):
            key = keys[i]
            grp = ([j for j in range(B) if j != i and keys[j] == key]
                   if key is not None else [])
            if not grp:
                rows.append(mu[i].sum() * 0.0)
                continue
            g_all = [i] + grp
            idx = torch.as_tensor(g_all, device=dmat.device)
            sub = dmat[idx][:, idx]                         # (G+1, G+1)
            G1 = sub.shape[0]
            iu = torch.triu_indices(G1, G1, offset=1)
            pd = sub[iu[0], iu[1]]                          # all group pairs
            h = self.pg_h_scale * float(pd.median())
            h = max(h, 1e-8)
            # force carriers: distance from mu_i (grad) to FROZEN mu_j
            di = torch.stack([torch.norm(mu[i] - mu_d[j]) for j in grp])
            k = torch.exp(-(di ** 2) / (2.0 * h * h))
            rows.append(k.sum())
        return torch.stack(rows)

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        mu, att_rows = self._att._encode_and_row_losses(x0_hat, current_obs)
        nll = torch.stack(att_rows)
        self._rep_calls += 1
        if self._denoise_n >= self.pg_start:
            rep = self._repulsion_rows(mu)
            nll = nll + self.pg_lambda * rep
            self._rep_on_calls += 1
            self._rep_row_sum += float(rep.detach().mean())
            if self._rep_calls % 2500 == 0:
                print(f"[pg-telemetry] active {self._rep_on_calls}/"
                      f"{self._rep_calls} calls, mean L_rep/row="
                      f"{self._rep_row_sum / max(self._rep_on_calls, 1):.4g}",
                      flush=True)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
