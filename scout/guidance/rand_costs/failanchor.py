"""rand_failanchor -- 反锚定 (fail-anchor reference, rand LESSONS 提案 1, 2026-08-28).

方案三's KL form is kept VERBATIM; only the reference MEAN is replaced by an
anchor derived from the scene's OWN already-failed retries (the conditional /
asymmetric randomization axis none of the five closed lines touched):

    C_k = -min( KL( q(z|s̄,a) ‖ N(μ̄_anchor(i), σ⁰²_k) ), κ )

σ⁰²_k stays the per-chunk unguided-intent variance captured by select_z
(exactly 方案三); μ⁰ is swapped for the scene anchor μ̄.  Force =
JᵀΛ⁰(μ−μ̄) -- still in the encoder Jacobian row space (no projection
recovery loss), still κ-capped, geometry still chosen by the encoder (only
the anchor point moved, per mjitter-r2's "same magnitude, rotated reference
= all gain lost" lesson).

Anchor modes (rand_anchor_mode):
  retry0   -- anchor = MEAN over retry 0's per-chunk intent means μ⁰
              (accumulated in select_z for rows with try_idx==0, frozen at
              the scene's first on_try_done).  Retry 0 itself runs plain
              方案三 (its anchor cannot exist yet).  No success-flag issue:
              retry 0 is by construction the first try of a FAILED scene.
  trailing -- anchor = per-dim TRIMMED mean (rand_trim_q=0.2) over the
              EXECUTED-chunk posterior means of all COMPLETED retries of the
              scene (codes recorded by the encode_executed hook, committed
              at on_try_done; successful retries included -- under the SOE
              rescue protocol a rescued scene's data is banked anyway).
              The anchor DRIFTS as retries accumulate: 10 retries become a
              coarse search of the failed-mode space.

Serialization: on_try_done is defined unconditionally -> rollout_vec's job
gate serializes a scene's retries (try j waits for try j-1), which BOTH
modes require (retry k's anchor must include retry k-1) -- anticipated ~2x
screen slowdown (LESSONS §8 提案1).

Randomness: NONE is drawn -- the anchor CONSUMES the realized retry
randomness (DDPM noise, guidance dynamics).  rand_seed is accepted for CLI
uniformity and currently unused; determinism given the run seeds is exact.

κ accounting (LESSONS §7.4): the KL floor at the unguided intent a⁰ is
floor = 0.5·‖μ⁰−μ̄‖²_{σ⁰²} (σ terms cancel exactly).  Floor > κ does NOT
silently shut the force down (unlike mjitter's metric floor): min(KL,κ)
simply saturates = a⁰ is already >κ nats from the anchor = no push needed.
The real risk is the opposite: an anchor so far away that the ENTIRE DP
support sits >κ nats away -> uniform e^κ boost ≈ placebo.  [failanchor]
telemetry prints per-chunk floor stats (mean/max, fraction floor≥κ) so
over-escape is visible in stdout.

Regression safety: every row WITHOUT an anchor (retry 0 always; try≥1 only
if the anchor never froze) is BYTE-EQUAL 方案三 -- {9,18}-type scenes where
retry 0 ≈ the DP's own spread should degrade gracefully to baseline
behavior.

ek knobs: rand_anchor_mode ("retry0"|"trailing", default "retry0"),
rand_cap (κ, default 2.5), rand_trim_q (default 0.2), rand_seed (42, unused).
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np
import torch

from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward

NAME = "failanchor"


def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict = None):
    ek = ek or {}
    mode = str(ek.get("rand_anchor_mode", "retry0"))
    if mode not in ("retry0", "trailing"):
        raise ValueError(
            f"rand_anchor_mode must be 'retry0' or 'trailing', got {mode!r}")
    return FailAnchorCostPlanner(
        scout_vib, bridge=bridge, obs_adapter=obs_adapter,
        anchor_mode=mode,
        cap=float(ek.get("rand_cap", ek.get("atypical_cap", 2.5))),
        trim_q=float(ek.get("rand_trim_q", 0.2)),
        rand_seed=int(ek.get("rand_seed", 42)),
    )


def _trimmed_mean(mats: Sequence[torch.Tensor], q: float) -> torch.Tensor:
    """Per-dimension trimmed mean over a stack of (D,) tensors; trims
    ``int(N*q)`` samples off EACH tail per dim (q<=0 or N<=1 -> plain mean)."""
    Z = torch.stack(list(mats))                       # (N, D)
    n = Z.shape[0]
    if n == 1 or q <= 0.0:
        return Z.mean(dim=0)
    k = int(n * q)
    if 2 * k >= n:
        k = max((n - 1) // 2, 0)
    if k > 0:
        Z = torch.sort(Z, dim=0).values[k:n - k]
    return Z.mean(dim=0)


class FailAnchorCostPlanner(AtypicalCostPlanner):
    """方案三 KL cost against a per-scene fail-anchor mean (see module doc)."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 anchor_mode: str = "retry0", cap: float = 2.5,
                 trim_q: float = 0.2, rand_seed: int = 42):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap)
        self.anchor_mode = str(anchor_mode)
        self.trim_q = float(trim_q)
        self.rand_seed = int(rand_seed)      # no new randomness drawn (doc)
        self._row_jobs: List = []            # [(init_idx, try_idx)] per row
        self._acc: dict = {}                 # init -> list[(D,) cpu tensors]
        self._anchor: dict = {}              # init -> (D,) cpu tensor
        self._r0_done: dict = {}             # init -> retry-0 finalized
        # floor telemetry (LESSONS §7.4/7.5): KL(anchor‖intent) at a^0
        self._fl_n = 0
        self._fl_sum = 0.0
        self._fl_sat = 0
        self._fl_max = 0.0

    # -- context plumbing (rollout_vec._replan) ---------------------------- #
    def set_row_jobs(self, jobs: Sequence):
        self._row_jobs = [(j[1], j[2]) for j in jobs]

    # -- executed-code recording (rollout_vec; also gates on_try_done on) -- #
    def encode_executed(self, obs, chunk_np) -> torch.Tensor:
        """Posterior mean of an EXECUTED chunk (trailing-mode anchor
        material); mirrors NoveltyCostPlanner.encode_executed exactly."""
        dev = next(self.scout_vib.parameters()).device
        obs_t = {k: torch.as_tensor(np.asarray(v, dtype=np.float32))[-1:].unsqueeze(0).to(dev)
                 for k, v in obs.items()}
        obs_es = (self.obs_adapter(obs_t) if self.obs_adapter is not None
                  else obs_t)
        a_flat = torch.as_tensor(
            np.asarray(chunk_np, dtype=np.float32).reshape(1, -1)).to(dev)
        with torch.no_grad():
            s_bar = self.scout_vib.encode(obs_es)
            mu, _ = self.scout_vib.vib_enc(s_bar.to(a_flat.device), a_flat)
        return mu[0].detach().cpu()

    def on_try_done(self, init_idx, codes: Sequence[torch.Tensor]):
        """A retry of scene ``init_idx`` finalized (job-gate serialized, so
        the FIRST call per scene is retry 0)."""
        first = init_idx not in self._r0_done
        self._r0_done[init_idx] = True
        if self.anchor_mode == "retry0":
            if not first:
                return                      # anchor frozen at retry 0, fixed
            mats = self._acc.pop(init_idx, None)
            if not mats:
                # pathological: retry 0 died inside its FIRST chunk (no
                # select_z ever fired for it) -- fall back to its executed
                # codes; if those are empty too, leave the scene unanchored
                # (every retry stays byte-equal 方案三).
                mats = list(codes)
            if mats:
                self._anchor[init_idx] = torch.stack(
                    [m.detach().cpu().float() for m in mats]).mean(dim=0)
        else:                               # trailing: anchor drifts
            acc = self._acc.setdefault(init_idx, [])
            acc.extend(c.detach().cpu().float() for c in codes)
            if acc:
                self._anchor[init_idx] = _trimmed_mean(acc, self.trim_q)

    # -- per-chunk intent baseline capture (= 方案三) + retry0 accumulation - #
    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        with torch.no_grad():
            s_bar_t = self._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t, a)
        self._base_mu = [m.detach() for m in mu]
        self._base_lv = [v.detach() for v in logvar]
        if self.anchor_mode == "retry0":
            for i in range(mu.shape[0]):
                key = self._row_jobs[i] if i < len(self._row_jobs) else None
                if (key is not None and key[0] is not None and key[1] == 0
                        and not self._r0_done.get(key[0], False)):
                    self._acc.setdefault(key[0], []).append(
                        mu[i].detach().cpu().float())
        # floor telemetry on anchored rows (cheap: reuse captured tensors)
        for i in range(mu.shape[0]):
            key = self._row_jobs[i] if i < len(self._row_jobs) else None
            anc = (self._anchor.get(key[0])
                   if key is not None and key[0] is not None else None)
            if anc is None:
                continue
            mu0c = self._base_mu[i].detach().cpu().float()
            var0c = torch.exp(self._base_lv[i].detach().cpu().float())
            floor = 0.5 * float((((mu0c - anc.float()) ** 2) / var0c).sum())
            self._fl_n += 1
            self._fl_sum += floor
            self._fl_sat += int(floor >= self.cap)
            self._fl_max = max(self._fl_max, floor)
        if self._fl_n and self._fl_n % 500 == 0:
            print(f"[failanchor] floor@intent n={self._fl_n} "
                  f"mean={self._fl_sum / self._fl_n:.3f} max={self._fl_max:.3f} "
                  f"frac>=kappa={self._fl_sat / self._fl_n:.3f} "
                  f"(mode={self.anchor_mode}, "
                  f"anchored_scenes={len(self._anchor)})")
        return None

    # -- the cost: 方案三 with m0 -> anchor when available ------------------ #
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
            m0 = self._base_mu[i]              # 方案三 fallback (byte-equal)
            key = self._row_jobs[i] if i < len(self._row_jobs) else None
            if key is not None and key[0] is not None:
                anc = self._anchor.get(key[0])
                if anc is not None and (self.anchor_mode == "trailing"
                                        or key[1] != 0):
                    m0 = anc.to(mu.device, mu.dtype)
            lv0 = self._base_lv[i]
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
