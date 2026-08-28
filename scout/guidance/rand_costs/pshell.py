"""rand_pshell -- persistent anchored shell (pshell, campaign re-aim 2026-08-28).

Campaign goal REDEFINED by the user: WIDEN the retry distribution during
exploration (not raise rescue).  方案A's random kappa-shell target died
from ANCHOR CHASING: select_z re-captured the intent baseline (mu^0,
sigma^0^2) EVERY CHUNK, so the well center mu* = mu^0_chunk +
sqrt(2*kappa)*sigma^0_chunk*u moved WITH the (already deflected) intent --
the closed DP attractor re-absorbed the shell displacement chunk by chunk
and the trajectory-level retry distribution collapsed back onto the
attractor (retry PR 1.56; 72/72 COLLAPSE finals on the same corner).

SOE has TWO parts; 方案A ported only the first:
  * a random direction drawn per retry             <- ported (方案A)
  * the latent held FIXED for the whole trajectory <- NOT ported (this file)
A persistent external field gets INTEGRATED by the trajectory instead of
being re-absorbed step by step.

pshell keeps ShellTargetCostPlanner's (方案A) KL-well math VERBATIM:
u(i,k) deterministic from (shell_seed, init, try) frozen across the whole
retry, target posterior q* = N(mu*, (sigma^0)^2) with mu* = mu^0 +
sqrt(2*kappa)*sigma^0*u (exactly kappa nats from the intent), and

    cost = KL( q(z|s_bar,a) || q* )        (quadratic well, uncapped)

-- and changes ONLY when the anchor (mu^0, sigma^0^2) is captured,
gated by ``rand_anchor_refresh``:

  retry  -- captured ONCE at the retry's FIRST select_z firing (the row's
            first chunk), cached per (init_idx, try_idx), reused verbatim
            for every later chunk of that retry: mu* is then CONSTANT over
            the entire retry = persistent field.        [main arm]
  chunk  -- re-captured every chunk (byte-equal 方案A original behavior;
            the anchor-chasing control arm).
  w5     -- re-captured every 5 chunks (intermediate; chunk counter per
            (init_idx, try_idx), refresh at chunks 1, 6, 11, ...).

No job-gate serialization is needed (anchor keyed by (init, try), retries
stay parallel -- unlike failanchor).  Rows without job context fall back
to a graph-connected zero exactly like 方案A.  If compute_loss somehow
fires before any capture for a key (cannot happen on the real path --
select_z precedes the first guided step), the row falls back to THIS
chunk's own baseline = 方案A semantics, never crashes.

ek knobs: rand_anchor_refresh ("retry"|"chunk"|"w5", default "retry"),
shell_kappa (default 2.5; also a run_rollout CLI flag), shell_seed (42).
"""
from __future__ import annotations

import torch

from scout.guidance.entropy_costs import ShellTargetCostPlanner, _enc_forward

NAME = "pshell"

_REFRESH_MODES = ("retry", "chunk", "w5")


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    refresh = str(ek.get("rand_anchor_refresh", "retry"))
    if refresh not in _REFRESH_MODES:
        raise ValueError(f"rand_anchor_refresh must be one of {_REFRESH_MODES},"
                         f" got {refresh!r}")
    return PShellCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        anchor_refresh=refresh,
        shell_kappa=float(ek.get("shell_kappa", 2.5)),
        shell_seed=int(ek.get("shell_seed", 42)),
    )


class PShellCostPlanner(ShellTargetCostPlanner):
    """方案A KL well with the anchor-capture granularity decoupled from the
    chunk clock (see module docstring)."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 anchor_refresh: str = "retry", shell_kappa: float = 2.5,
                 shell_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         shell_kappa=shell_kappa, shell_seed=shell_seed)
        if anchor_refresh not in _REFRESH_MODES:
            raise ValueError(f"anchor_refresh must be one of {_REFRESH_MODES},"
                             f" got {anchor_refresh!r}")
        self.anchor_refresh = str(anchor_refresh)
        # (init_idx, try_idx) -> captured (mu^0, logvar^0), cpu tensors
        self._anchor_mu: dict = {}
        self._anchor_lv: dict = {}
        # (init_idx, try_idx) -> number of select_z firings seen (chunk clock)
        self._chunk_ctr: dict = {}

    def _row_key(self, i: int):
        return (self._row_jobs[i] if i < len(self._row_jobs)
                else (None, None))

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Per-chunk intent baseline capture.  refresh=chunk keeps 方案A's
        exact per-chunk behavior; retry/w5 additionally freeze the FIRST
        (or every-5th) capture as the persistent per-retry anchor."""
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t, a)
        self._base_mu = [m.detach() for m in mu]
        self._base_lv = [v.detach() for v in logvar]
        if self.anchor_refresh == "chunk":
            return None
        for i in range(mu.shape[0]):
            key = self._row_key(i)
            if key[0] is None:
                continue                     # no job context -> nothing to pin
            ctr = self._chunk_ctr.get(key, 0)
            self._chunk_ctr[key] = ctr + 1
            if self.anchor_refresh == "retry":
                if key not in self._anchor_mu:       # capture ONCE per retry
                    self._anchor_mu[key] = mu[i].detach().cpu()
                    self._anchor_lv[key] = logvar[i].detach().cpu()
            elif ctr % 5 == 0:                       # w5: chunks 1,6,11,...
                self._anchor_mu[key] = mu[i].detach().cpu()
                self._anchor_lv[key] = logvar[i].detach().cpu()
        return None

    def _anchor_for(self, key, row: int, device, dtype):
        """(m0, lv0) for this row: the persistent anchor when one exists,
        else THIS chunk's own capture (方案A fallback, graph-safe)."""
        am = self._anchor_mu.get(key)
        if am is None:
            return self._base_mu[row], self._base_lv[row]
        return (am.to(device=device, dtype=dtype),
                self._anchor_lv[key].to(device=device, dtype=dtype))

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
            key = self._row_key(i)
            if key[0] is None:
                rows.append(x0_hat[i].sum() * 0.0)
                continue
            if self.anchor_refresh == "chunk":
                m0, lv0 = self._base_mu[i], self._base_lv[i]
            else:
                m0, lv0 = self._anchor_for(key, i, mu.device, mu.dtype)
            var0 = torch.exp(lv0)
            sig0 = torch.exp(0.5 * lv0)
            u = self._u_for(key, mu.device, mu.dtype)
            target_mu = m0 + (2.0 * self.shell_kappa) ** 0.5 * sig0 * u
            var = torch.exp(logvar[i])
            kl = 0.5 * (((mu[i] - target_mu) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            rows.append(kl)     # quadratic well around the (persistent) target
        nll = torch.stack(rows)
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
