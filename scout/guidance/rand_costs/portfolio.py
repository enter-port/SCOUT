"""rand_portfolio -- retry-index split portfolio (open problem 2026-08-28).

CONTEXT.  After the campaign re-aim (width = main metric, rescue = guard
rail) NO single cost configuration satisfies BOTH objectives: entropy cost
(atypical, eta=3.0) holds the rescue line {3,9,12,18} but its retry cone is
narrow (the capped bonus tilts every retry along the SAME high-gain
direction); shell (chunk-refreshed anchor = 方案A/pshell_c, eta=0.35) is the
widest arm measured (PR 1.72 / d_act 0.77) but rescues only a placebo-tier
subset.  Portfolio is the "don't choose, split" bet: the ten retries of one
scene are DIVISIONED BY RETRY INDEX --

    try_idx <  K (rand_split):  entropy cost  cost = -min(KL(q_a||q0), kappa)
    try_idx >= K:               shell         cost =  KL(q_a||q*) with
                                q* = N(mu0 + sqrt(2*shell_kappa)*sigma0*u(i,k),
                                       sigma0^2)

The first K retries are the rescue bank (the calibrated entropy-cost
machinery, bit-equal formulas), the last 10-K are the width bank (random
kappa-shell targets, one per retry).  The scene is rescued if ANY try
succeeds, so the width bank is free to sacrifice per-try success rate.

MECHANICS (one file, one knob -- rand_split):
  * select_z (inherited from ShellTargetCostPlanner, byte-equal to
    AtypicalCostPlanner's) captures the per-chunk intent baseline
    (mu^0, sigma^0^2) ONCE PER CHUNK -- BOTH formulas share the SAME
    baseline; on the shell side this is the chunk-refresh anchor = 方案A
    original behavior (pshell_c, the widest caliber arm);
  * set_row_jobs (inherited) maps each replan row to its (init_idx,
    try_idx) -- retries stay parallel, no job gate;
  * u(i,k) is drawn once per (init_idx, try_idx) from
    default_rng([rand_seed, init_idx, try_idx]) and frozen for the whole
    retry (inherited _u_for with shell_seed := rand_seed);
  * rows WITHOUT job context fall back to the ENTROPY-COST formula (the
    campaign main line is the default; the shell branch exists only with
    real retry-index context).

THE ETA CONSTRAINT (real, physical -- read before interpreting results):
guidance_scale (eta) is a SINGLE GLOBAL scalar on the injection path
(policy.guided_conditional_sample), shared by every row of every replan.
entropy cost's working point is eta=3.0; shell's is eta=0.35 (MINI-shell:
eta=3.0 injects ~26x the working band).  A portfolio run therefore CANNOT
give each formula its own calibrated dose.  Arms:
    pf_k5_s3  -- K=5, eta=3.0 (rescue bank at its working point; shell rows
                 run ~8.6x their calibrated dose -> expected over-injection)
    pf_k7_s3  -- K=7, eta=3.0 (rescue-leaning split)
    pf_k5_s1  -- K=5, eta=1.0 (midpoint: ent rows at 1/3 dose, shell rows
                 at ~2.9x dose -- both sides damaged, test of containment)

TELEMETRY (this file only; policy.py's mean_inject is a global blend and is
a SHARED file): compute_loss(reduction="sum") additionally takes ONE extra
cheap backward through bridge+vib_enc only (the small path; the UNet graph
is untouched) and accumulates, PER FORMULA, the mean per-row x0-level
gradient norm ||d cost_row / d x0_hat||, the mean row cost, and the entropy
rows' clamp-saturation share.  Printed as [portfolio-telemetry] every
_tl_every rows (cumulative, like policy.py's [guidance-telemetry]).  The
true injected force is eta*sqrt(1-alpha_bar_t)*d cost/d trajectory -- the
scheduler/epsilon Jacobian between x0_hat and trajectory is NOT included,
so the two branch numbers are a PROXY whose BRANCH RATIO at a given eta is
approximately the inject ratio (both branches pass through the same
eta*sqrt(1-alpha_bar_t) factor); the faithful headline number remains the
global mean_inject from [guidance-telemetry].

ek knobs: rand_split (K, default 5), shell_kappa (2.5), rand_cap (kappa for
the entropy-cost clamp, 2.5), rand_seed (42; also the u seed).
"""
from __future__ import annotations

from typing import List, Optional

import torch

from scout.guidance.entropy_costs import ShellTargetCostPlanner, _enc_forward

NAME = "portfolio"


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    return PortfolioCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        rand_split=int(ek.get("rand_split", 5)),
        shell_kappa=float(ek.get("shell_kappa", 2.5)),
        rand_cap=float(ek.get("rand_cap", 2.5)),
        rand_seed=int(ek.get("rand_seed", 42)),
    )


class PortfolioCostPlanner(ShellTargetCostPlanner):
    """Retry-index split portfolio: entropy cost for tries < rand_split,
    shell (chunk-refresh anchor, frozen u) for the rest.  Both formulas
    share the per-chunk baseline captured by the (inherited) select_z."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 rand_split: int = 5, shell_kappa: float = 2.5,
                 rand_cap: float = 2.5, rand_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         shell_kappa=shell_kappa, shell_seed=int(rand_seed))
        self.rand_split = int(rand_split)
        self.rand_cap = float(rand_cap)
        self.rand_seed = int(rand_seed)
        # per-formula telemetry buffers (lazy, on first guided call):
        # [0] sum ent-row grad norms   [1] sum shell-row grad norms
        # [2] sum ent-row costs        [3] sum shell-row costs
        # [4] ent-row clamp-saturation count
        self._tl_acc: Optional[torch.Tensor] = None
        self._tl_n = [0, 0]           # [ent rows, shell rows]
        self._tl_every = 5000         # print tick (attribute: smoke shrinks it)

    # hooks: set_row_jobs / set_row_context / select_z / _u_for inherited.

    # -- the cost ----------------------------------------------------------- #
    def _row_costs(self, mu, logvar, x0_hat):
        """Per-row costs + branch labels (exposed for the smoke's per-row
        dispatch check; compute_loss is the only caller on the real path)."""
        rows: List[torch.Tensor] = []
        branches: List[str] = []
        sats: List[Optional[torch.Tensor]] = []
        for i in range(mu.shape[0]):
            if i >= len(self._base_mu) or self._base_mu[i] is None:
                rows.append(x0_hat[i].sum() * 0.0)
                branches.append("zero")
                sats.append(None)
                continue
            job = (self._row_jobs[i] if i < len(self._row_jobs)
                   else (None, None))
            m0, lv0 = self._base_mu[i], self._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            if (job[0] is None or job[1] is None
                    or int(job[1]) < self.rand_split):
                # entropy-cost row (try_idx < K) or no-job fallback.
                # VERBATIM AtypicalCostPlanner expression (bitwise-equal at
                # rand_split=10 / on fallback rows).
                kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                            + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
                rows.append(-torch.clamp(kl, max=self.rand_cap))
                branches.append("ent")
                sats.append(kl > self.rand_cap)          # bool, no graph
            else:
                # shell row: KL(q_a || q*), chunk-refresh anchor (m0 = this
                # chunk's capture), u frozen per (init, try) from rand_seed.
                # VERBATIM ShellTargetCostPlanner expression (bitwise-equal
                # at rand_split=0).
                sig0 = torch.exp(0.5 * lv0)
                u = self._u_for((int(job[0]), int(job[1])),
                                mu.device, mu.dtype)
                target_mu = m0 + (2.0 * self.shell_kappa) ** 0.5 * sig0 * u
                kl = 0.5 * (((mu[i] - target_mu) ** 2 / var0)
                            + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
                rows.append(kl)
                branches.append("shell")
                sats.append(None)
        return rows, branches, sats

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows, branches, sats = self._row_costs(mu, logvar, x0_hat)
        nll = torch.stack(rows)
        if reduction == "sum":
            if torch.is_grad_enabled() and x0_hat.requires_grad:
                self._telemetry(nll, rows, branches, sats, x0_hat)
            return nll.sum()
        if reduction == "mean":
            return nll.mean()
        raise ValueError(reduction)

    # -- per-formula inject proxy (see module docstring) -------------------- #
    def _telemetry(self, nll, rows, branches, sats, x0_hat) -> None:
        if self._tl_acc is None or self._tl_acc.device != rows[0].device:
            self._tl_acc = torch.zeros(5, device=rows[0].device)
        # one extra backward through bridge+vib_enc ONLY (retain_graph:
        # policy.py's outer autograd.grad(loss, trajectory) still to come)
        loss = nll.sum()
        g = torch.autograd.grad(loss, x0_hat, retain_graph=True)[0]
        gn = g.reshape(g.shape[0], -1).norm(dim=1)
        for i, br in enumerate(branches):
            if br == "ent":
                self._tl_acc[0] += gn[i].detach()
                self._tl_acc[2] += rows[i].detach()
                if sats[i] is not None:
                    self._tl_acc[4] += sats[i].to(self._tl_acc.dtype)
                self._tl_n[0] += 1
            elif br == "shell":
                self._tl_acc[1] += gn[i].detach()
                self._tl_acc[3] += rows[i].detach()
                self._tl_n[1] += 1
        tot = self._tl_n[0] + self._tl_n[1]
        if tot >= self._tl_every and tot % self._tl_every == 0:
            n_e = max(self._tl_n[0], 1)
            n_s = max(self._tl_n[1], 1)
            ge = float(self._tl_acc[0]) / n_e
            gs = float(self._tl_acc[1]) / n_s
            ratio = (gs / ge) if ge > 0 else float("inf")
            print(f"[portfolio-telemetry] rows ent={self._tl_n[0]} "
                  f"shell={self._tl_n[1]} | ent g={ge:.4g} "
                  f"cost={float(self._tl_acc[2]) / n_e:.4g} "
                  f"sat={100 * float(self._tl_acc[4]) / n_e:.0f}% | "
                  f"shell g={gs:.4g} "
                  f"cost={float(self._tl_acc[3]) / n_s:.4g} | "
                  f"g-ratio shell/ent={ratio:.2f}", flush=True)
