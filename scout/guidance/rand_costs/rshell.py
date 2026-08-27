"""rand_rshell -- reachable-subspace whitening shell (geometry.md 方案 B).

方案A's diagnosis: uniform random 16-d directions are projected back onto the
narrow cone of the encoder Jacobian G = J^T Lambda^0 J (low effective rank) by
the guided dynamics -- the sampled diversity never reaches the code space.
rshell therefore samples the target direction INSIDE the measured reachable
subspace (image of Lambda^0{1/2} J), so projection cannot erase it:

  1. PROBE (select_z, no_grad, once per chunk, after the intent baseline
     (mu^0, sigma^0^2) is captured): perturb the intent action chunk
     a^0 = bridge(x̂₀) by delta_a_p ~ N(0, tau^2 I_chunk), p = 1..P; feed
     through the frozen VIB encoder and whiten:

         w_hat_p = (mu(s̄, a^0 + delta_a_p) - mu^0) / sigma^0
                 ~ Lambda^0{1/2} J · (tau · eps_p)

     Batched SVD of W_hat/tau (P x 16) gives eigenpairs (v_i in R^16, lambda_i)
     of the reachable operator (lambda_i ~= sqrt(P)·sigma_i(Lambda^0{1/2}J),
     scale-free after normalizing by lambda_1).  Keep lambda_i >= eps·lambda_1
     -> V_r (16 x r), lambda_hat_i = lambda_i/lambda_1.  r / lambda_1 /
     lambda_2 are PRINTED to stdout (free pre-diagnosis: if r <= 2 the retry
     direction diversity is physically capped -> report it).

     tau is self-calibrated per row: rand_probe_tau x mean over action dims of
     the per-dim std of the intent chunk across its steps (probe stays in the
     local linear regime of the encoder).

  2. DIRECTION: each retry (scene i, try k) draws eps ~ N(0, I_r) from
     rng = default_rng([rand_seed, i, k]) -- deterministic, redrawn identically
     at every chunk of the retry (trajectory-level coherence: the SAME random
     coefficients re-expressed in the CURRENT chunk's subspace) --

         u_hat = normalize( V_r · (lambda_hat^{-alpha} ⊙ eps) )

     alpha = 0 -> uniform on the subsphere (equal-KL shell); alpha > 0
     overweights low-gain dims; alpha = -1/2 equalises action displacement.

  3. TARGET + COST (inherited well math from 方案A's ShellTargetCostPlanner):
     mu* = mu^0 + sqrt(2·kappa)·sigma^0 ⊙ u_hat  (exactly kappa nats from the
     intent: KL(q*||q^0) = kappa·||u_hat||^2); cost = KL(q(z|s̄,a) || N(mu*,
     sigma^0^2)) closed form -- a quadratic well at q*; the DP prior stays the
     only trust region (p ∝ p_DP·exp(-cost) reweights, never amplifies).

Probe noise uses a DEDICATED torch.Generator (seeded once from rand_seed) so
the global RNG stream is untouched (scale=0 == unguided byte-parity invariant).

ek knobs: rand_alpha (0), rand_probe_p (32), rand_probe_eps (1e-2),
rand_probe_tau (0.1), rand_cap (2.5, falls back to atypical_cap then 2.5),
rand_seed (42, falls back to shell_seed).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from scout.guidance.entropy_costs import ShellTargetCostPlanner, _enc_forward

NAME = "rshell"


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    return RShellCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        alpha=float(ek.get("rand_alpha", 0.0)),
        probe_p=int(ek.get("rand_probe_p", 32)),
        probe_eps=float(ek.get("rand_probe_eps", 1e-2)),
        probe_tau=float(ek.get("rand_probe_tau", 0.1)),
        kappa=float(ek.get("rand_cap", ek.get("atypical_cap", 2.5))),
        rand_seed=int(ek.get("rand_seed", ek.get("shell_seed", 42))),
    )


class RShellCostPlanner(ShellTargetCostPlanner):
    """方案B rshell: 方案A's kappa-shell well, target direction drawn from the
    PROBED reachable subspace of Lambda^0{1/2}J instead of R^16."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 alpha: float = 0.0, probe_p: int = 32,
                 probe_eps: float = 1e-2, probe_tau: float = 0.1,
                 kappa: float = 2.5, rand_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         shell_kappa=kappa, shell_seed=rand_seed)
        self.alpha = float(alpha)
        self.probe_p = int(probe_p)
        self.probe_eps = float(probe_eps)
        self.probe_tau = float(probe_tau)
        self.rand_seed = int(rand_seed)
        # per-chunk probe state (rebuilt in select_z)
        self._row_V: List[Optional[torch.Tensor]] = []     # (16, r) per row
        self._row_lam: List[Optional[torch.Tensor]] = []   # (r,) = lam/lam1
        # probe RNG: dedicated generator, global stream untouched
        self._probe_gen: Optional[torch.Generator] = None
        self._probe_calls = 0

    # -- hooks ------------------------------------------------------------ #
    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Capture the per-row intent baseline (mu^0, sigma^0^2) from the
        unguided x̂₀ AND probe the reachable subspace for the chunk (both
        no_grad; one extra batched encoder forward of B*P rows)."""
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a0 = _enc_forward(self, x0_hat)
            mu0, lv0 = self.scout_vib.vib_enc(s_bar_t, a0)
            self._probe(x0_hat, s_bar_t, a0, mu0, lv0)
        self._base_mu = [m.detach() for m in mu0]
        self._base_lv = [v.detach() for v in lv0]
        return None

    def _probe(self, x0_hat, s_bar_t, a0, mu0, lv0):
        B, dch = a0.shape
        P = self.probe_p
        per_step = x0_hat.shape[-1]
        n_steps = max(1, dch // per_step)
        # tau: per-dim std of the intent chunk across its steps, averaged
        # over dims (scalar per row -> isotropic probe in chunk space)
        tau = self.probe_tau * a0.reshape(B, n_steps, -1).std(dim=1).mean(dim=1)
        tau_safe = tau.clamp(min=1e-8)
        if self._probe_gen is None or self._probe_gen.device != a0.device:
            self._probe_gen = torch.Generator(device=a0.device)
            self._probe_gen.manual_seed(self.rand_seed)
        eps = torch.randn((B, P, dch), generator=self._probe_gen,
                          device=a0.device, dtype=a0.dtype)
        A = (a0.unsqueeze(1) + eps * tau_safe.view(B, 1, 1)).reshape(B * P, dch)
        s_rep = s_bar_t.repeat_interleave(P, dim=0)
        mu_p = self.scout_vib.vib_enc(s_rep, A)[0].reshape(B, P, -1)
        sig0 = torch.exp(0.5 * lv0)
        # w_hat_p / tau = Lambda^{1/2} J eps_p  (linear-regime estimate)
        W_hat = (mu_p - mu0.unsqueeze(1)) / (
            sig0.unsqueeze(1) * tau_safe.view(B, 1, 1))
        _, S, Vh = torch.linalg.svd(W_hat, full_matrices=False)   # S: (B, m)
        lam1 = S[:, 0]
        self._row_V, self._row_lam = [], []
        for i in range(B):
            l1 = float(lam1[i])
            if l1 < 1e-12:
                self._row_V.append(None)      # encoder insensitive -> no pull
                self._row_lam.append(None)
                continue
            keep = S[i] >= self.probe_eps * l1
            r_i = int(keep.sum().item())
            self._row_V.append(Vh[i, :r_i].T.detach() if r_i >= 1 else None)
            self._row_lam.append((S[i, :r_i] / l1).detach() if r_i >= 1 else None)
        # free pre-diagnosis: subspace dim + spectrum head (first 3 chunks,
        # then every 200th -- stdout is block-buffered, flush explicitly)
        self._probe_calls += 1
        if self._probe_calls == 1:
            print(f"[rshell] probe knobs: P={P} eps={self.probe_eps} "
                  f"tau_c={self.probe_tau} alpha={self.alpha} "
                  f"kappa={self.shell_kappa} seed={self.rand_seed}", flush=True)
        if self._probe_calls <= 3 or self._probe_calls % 200 == 0:
            rs = [int(v.shape[1]) if v is not None else 0
                  for v in self._row_V]
            l1s = [round(float(lam1[i]), 3) for i in range(B)]
            l2s = [round(float(S[i, 1]), 3) if S.shape[1] > 1 else 0.0
                   for i in range(B)]
            print(f"[rshell-probe] call#{self._probe_calls} rows={B} "
                  f"r={rs} lam1={l1s} lam2={l2s}", flush=True)

    def _u_for(self, row: int, key, device, dtype) -> torch.Tensor:
        """Unit direction for batch-row ``row`` (uses the CURRENT chunk's
        probed V_r) keyed deterministically by (rand_seed, init, try)."""
        V = self._row_V[row].to(device=device, dtype=dtype)
        lam = self._row_lam[row].to(device=device, dtype=dtype)
        rng = np.random.default_rng([self.rand_seed,
                                     int(key[0]), int(key[1])])
        e = torch.as_tensor(rng.standard_normal(int(lam.shape[0])),
                            dtype=dtype, device=device)
        w = V @ (lam ** (-self.alpha) * e)
        n = float(w.norm())
        return w / (n if n > 0.0 else 1.0)

    # -- cost ------------------------------------------------------------- #
    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows = []
        for i in range(mu.shape[0]):
            if (i >= len(self._base_mu) or self._base_mu[i] is None
                    or i >= len(self._row_V) or self._row_V[i] is None):
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
            u = self._u_for(i, key, mu.device, mu.dtype)
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
