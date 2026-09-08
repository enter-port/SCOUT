"""GAE-like history-weighted KL cost (user 2026-09-08, branch GAElike-dev).

RL's Generalized Advantage Estimation (Schulman et al. 2015) aggregates
MULTIPLE single-step signals into one estimate with exponential decay:

    A_t = sum_{l>=0} (gamma*lambda)^l * delta_{t+l}

-- the decay keeps the infinite sum bounded and interpolates between
near-term (low variance) and long-range (low bias) evidence. The user's
transplant to SCOUT: the entropy cost currently compares the candidate
intent ONLY against the per-chunk anchor a^0 (the DP's unguided intent at
the first guided denoise step); GAElike instead compares it against EVERY
action intent the denoise walk has passed through since that anchor,
GAE-weighted:

    cost_t = - sum_{k=0}^{t-1} g^k * min( KL( q(z|s̄,a_t) ‖ q(z|s̄,a_k) ), κ )
             / ( sum_{k=0}^{t-1} g^k  if gae_normalize else 1 )

  * q_k = the encoder posterior of the x̂₀ iterate at guided step k of THIS
    chunk (k=0 is the anchor captured by select_z -- the DP's unguided
    intent; k>=1 are the post-injection iterates). History is per-chunk:
    every new anchor (select_z) resets it, mirroring the vanilla baseline.
  * g^k (``gae_gamma``) is the GAE discount: the anchor term keeps weight
    1 and older-to-newer history decays geometrically, so the history sum
    stays bounded no matter how many guided steps the chunk runs. g -> 0
    keeps ONLY the k=0 term = the vanilla entropy cost exactly on CPU
    (value AND gradient; the normalize divisor is then 1; GPU accumulation
    may differ at ulp level -- same caveat as the 2026-09-01 _kl_rows
    vectorization), so the new mechanism is a strict superset of
    ``--guide atypical`` and inherits its calibrated dose.
  * each pairwise KL is capped at κ exactly as the vanilla anchor term
    (double trust region per comparison); ``gae_normalize`` (default ON)
    divides by the weight sum so the ROW cost envelope stays [0, κ] and
    the existing η/guidance_scale semantics carry over unchanged. OFF
    (review P1-1, semantics pending user decision) = raw unnormalized
    weights with the anchor still dominant, BUT the inherited climb's
    AGGREGATE kappa budget then kills the whole row (injection gradient)
    as soon as the weighted SUM reaches κ -- rows die EARLIER than
    atypical, it is NOT a ~(1/(1-g))x dose uplift. Calibrate with ON.
  * mechanism reading: keep "escape the anchor" as the dominant push, add
    a decaying "do not fall back onto any intent already visited"
    pressure (anti-return / momentum across the denoise walk).

Implementation notes (subclass contract): KLCostPlanner owns ALL of
phase 1 (select_z anchor capture, capped climb, eta_tilde normalization,
merged single-backward guided_step). This subclass swaps ONLY the scalar
being climbed: `_kl_backward` returns the GAE-weighted per-row quantity
instead of the anchor-only KL; the inherited `_climb_gradient` row mask
(kl <= cap) then saturates on the weighted value, whose envelope equals
the vanilla one under normalization. The history buffer is (B, H, dz)
detached, row-aligned with the CURRENT batch: rollout_vec._replan batches
all slots replanning at a tick into ONE guided_conditional_sample call,
select_z fires for that whole batch at the loop's first guided step, and
all rows share the loop -- so H is call-local and identical across rows
(the per-env chunk-phase desync lives BETWEEN replan calls, each of which
resets the buffer). The current posterior is appended to the history
(detached) INSIDE `_kl_backward` after the backward, EXCEPT on the anchor
step itself (its posterior IS the anchor; appending would double-count
the anchor with weight g^1). `compute_loss` is a read-only view (no
append) so monitoring/diagnostic callers cannot desync the buffer. No
RNG is consumed anywhere: bit-comparability with the other KL-cost arms
on the same seed is preserved (g=0 replays atypical's values).
"""

from __future__ import annotations

import torch

from scout.guidance.entropy_costs import KLCostPlanner, _enc_forward


class GAELikeCostPlanner(KLCostPlanner):
    """GAElike cost: gamma^k-weighted sum of capped KLs from the candidate
    intent to EVERY intent visited since the per-chunk anchor."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0, eta_dimless: bool = False,
                 gae_gamma: float = 0.9, gae_normalize: bool = True):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap, eta_dimless=eta_dimless)
        self.gae_gamma = float(gae_gamma)
        self.gae_normalize = bool(gae_normalize)
        if not (0.0 <= self.gae_gamma <= 1.0):
            raise ValueError(
                f"gae_gamma must be in [0, 1] (0 = vanilla atypical); got "
                f"{self.gae_gamma}")
        # history buffer: (B, H, dz) posteriors of the anchor + the guided
        # steps since it, detached (constants w.r.t. the climb gradient --
        # same treatment as the vanilla anchor (_base_mu/_base_lv)).
        self._hist_mu: torch.Tensor | None = None
        self._hist_lv: torch.Tensor | None = None
        # True between select_z and the first _kl_backward of a chunk: the
        # anchor step's own posterior is the history's last column already.
        self._gae_fresh = False
        # telemetry accumulators (lightweight; printed on a tick like the
        # base [kl-telemetry] line).
        self._gae_calls = 0
        self._hist_len_acc = 0.0
        self._kl_acc: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # per-chunk reset: anchor capture (base) + history := [anchor]
    # ------------------------------------------------------------------ #
    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        super().select_z(x0_hat, current_obs)
        # rows are row-aligned with the CURRENT replan batch (see class
        # docstring); stack the just-captured per-row anchor list -> (B,1,dz).
        self._hist_mu = torch.stack(self._base_mu).detach().unsqueeze(1)
        self._hist_lv = torch.stack(self._base_lv).detach().unsqueeze(1)
        self._gae_fresh = True
        return None

    def reset(self):
        """Base reset (clear s̄_t / z caches) + drop the GAE history -- a
        reused planner instance must not see a stale, same-B history (which
        would slip past the row-alignment guard; review P2-5)."""
        super().reset()
        self._hist_mu = None
        self._hist_lv = None
        self._gae_fresh = False

    # ------------------------------------------------------------------ #
    # the GAE-weighted per-row cost (B,)
    # ------------------------------------------------------------------ #
    def _gae_rows(self, mu: torch.Tensor, logvar: torch.Tensor,
                  x0_hat: torch.Tensor) -> torch.Tensor:
        """Per-row sum_k g^k * min(KL(q_t ‖ q_k), κ), optionally weight-sum
        normalized. Graph-connected zeros when no history is usable (direct
        calls before select_z -- unreachable via the public rollout path;
        mirrors ``_kl_rows``' None-baseline contract)."""
        if (self._hist_mu is None or self._hist_lv is None
                or self._hist_mu.shape[0] != mu.shape[0]
                or self._hist_mu.shape[1] < 1):
            return x0_hat.flatten(1).sum(dim=1).to(mu.dtype) * 0.0
        H = self._hist_mu.shape[1]
        # (B, H, dz) pairwise diagonal-Gaussian KL, elementwise identical to
        # _kl_rows per (row, history column): 0.5*sum_d[(mu-m0)^2/var0 +
        # var/var0 - 1 - (logvar-lv0)].
        var = torch.exp(logvar).unsqueeze(1)                    # (B, 1, dz)
        var0 = torch.exp(self._hist_lv)                         # (B, H, dz)
        kl = 0.5 * (((mu.unsqueeze(1) - self._hist_mu) ** 2 / var0)
                    + (var / var0) - 1.0
                    - (logvar.unsqueeze(1) - self._hist_lv)).sum(dim=-1)  # (B,H)
        kl = torch.clamp(kl, max=float(self.cap))               # per-term κ
        w = torch.as_tensor(
            [self.gae_gamma ** k for k in range(H)],
            device=mu.device, dtype=mu.dtype)                    # (H,)
        total = (kl * w).sum(dim=1)                              # (B,)
        if self.gae_normalize:
            total = total / w.sum()
        return total

    def _push_history(self, mu: torch.Tensor, logvar: torch.Tensor):
        """Append this step's (detached) posterior as the newest history
        column. Called from _kl_backward only (the once-per-guided-step
        consumer), so monitoring reads cannot skew the history. No-op when
        there is no usable row-aligned history (pre-select_z / B mismatch --
        the guard path's rows are zeros; appending them would corrupt the
        buffer shape or crash on None)."""
        if (self._hist_mu is None or self._hist_lv is None
                or self._hist_mu.shape[0] != mu.shape[0]):
            return
        self._hist_mu = torch.cat(
            [self._hist_mu, mu.detach().unsqueeze(1)], dim=1)
        self._hist_lv = torch.cat(
            [self._hist_lv, logvar.detach().unsqueeze(1)], dim=1)

    # ------------------------------------------------------------------ #
    # overrides: swap the climbed scalar, keep everything else verbatim
    # ------------------------------------------------------------------ #
    def _kl_backward(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                     current_obs=None):
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        kl = self._gae_rows(mu, logvar, x0_hat)
        g = torch.autograd.grad(kl.sum(), trajectory)[0]
        # history advance: the anchor step's posterior IS the anchor (same
        # pre-injection x̂₀, deterministic encoder) -- skip the duplicate.
        if self._gae_fresh:
            self._gae_fresh = False
        else:
            self._push_history(mu, logvar)
        # telemetry tick (mirrors the base [kl-telemetry] cadence).
        self._gae_calls += 1
        if self._hist_mu is not None:
            self._hist_len_acc += float(self._hist_mu.shape[1])
        _kl_mean = kl.detach().mean()
        self._kl_acc = (_kl_mean if self._kl_acc is None
                        or self._kl_acc.device != _kl_mean.device
                        else self._kl_acc + _kl_mean)
        if self._gae_calls % 2500 == 0:
            print(f"[gae-telemetry] calls={self._gae_calls} "
                  f"mean_hist_len={self._hist_len_acc / self._gae_calls:.1f} "
                  f"mean_kl={float(self._kl_acc) / self._gae_calls:.4g} "
                  f"gamma={self.gae_gamma} norm={int(self.gae_normalize)}",
                  flush=True)
        return kl, g

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        """Read-only view of the GAElike row cost (NO history append), for
        monitoring / diagnostics; the rollout hot path uses guided_step."""
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        kl = self._gae_rows(mu, logvar, x0_hat)
        nll = -torch.clamp(kl, max=float(self.cap))
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
