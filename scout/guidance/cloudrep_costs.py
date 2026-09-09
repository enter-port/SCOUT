"""Cloud-referenced repulsion cost (drift-dev, user 2026-09-09 方案一).

Transplants the Drifting Model (arXiv:2602.04770) repulsion field into the
SCOUT KL-cost family: the drifting field's negative term V^- is a
KERNEL-NORMALIZED push away from the model's own sample cloud, and the
entropy cost's atypicality semantics ("do what the policy would NOT
habitually do") becomes, under this reading, repulsion from the policy's
own samples -- with the reference set widened from the single per-chunk
anchor a^0 to the cloud of anchors the policy has ALREADY produced on this
scene. Formally a new instance of the target-family equation
cost = -min(d(q_phi(.|s̄,a) || T), κ) with

    T = { q(z|s̄, y_j) }_j   (the policy's own posterior cloud on the scene)

whose members are the chunk-0 anchors (μ⁰, σ⁰²) captured by select_z at the
FIRST chunk of every retry of the scene. Ordering is enforced by the
START-gate (``try_started_gate``, consumed by rollout_vec's job-gate
dispatch): try k launches only once tries < k have STARTED -- i.e. committed
their chunk-0 anchor -- not once they finished. Retry k's chunk-0 pool
therefore provably contains the anchors of retries < k, while slots stay
fully parallel (a finished-serialization gate, the NoveltyCostPlanner
on_try_done contract, would idle slots whenever scenes < n_envs and inflate
wall clock ~2.3x on the probe's 11-16 scenes/worker vs 25 slots). Per row
the climbed scalar is the temperature soft-min over the pool of pairwise
KLs:

    S_i  = -τ · log Σ_{j∈R_i} exp(-KL_ij / τ)          (soft-min)
    R_i  = [current chunk anchor] ∪ [scene cloud anchors]
    cost = -min(S_i, κ)

  * τ (``cloudrep_tau``) is the drifting kernel temperature. The GRADIENT
    dS/dKL_ij = exp(-KL_ij/τ)/Σ_k exp(-KL_ik/τ) is exactly the drifting
    normalized kernel (softmax over reference axis, Alg. 2): the push is a
    WEIGHTED MEAN of per-reference escape gradients with weights summing
    to 1 -- bounded by construction, no dose explosion. Small τ hardens to
    "escape the NEAREST thing already tried"; large τ averages over the
    cloud.
  * Retry 0 (empty cloud) has pool = {anchor} alone: S reduces to the KL
    exactly (logsumexp of one element is the identity). For power-of-two τ
    (incl. the default 0.5) the ·τ / ÷τ round-trips are strictly invertible
    in IEEE754, so VALUE and GRADIENT are bit-identical to ``--guide
    atypical`` and the calibrated dose η3.0/κ2.5 carries over unchanged;
    for other τ the INJECTION gradient stays bit-identical (single-ref
    softmax weights are τ-free) while the cost VALUES may differ by 1 ulp
    (logsumexp's internal max-shift) -- telemetry-level only. The A/B
    contrast IS the cloud effect, appearing from retry 1 onward.
  * Anti-repetition across retries (the j-axis anchor bank that the GAElike
    post-mortem pre-registered as the untried axis -- the i-axis within a
    denoise walk was falsified there): retry k is pushed away from what
    retries 0..k-1 stood for at the SAME initial state (rescue resets to
    the same init, so chunk-0 posteriors live at identical s̄ and are
    directly comparable).
  * Trust region: the inherited cap mask uses S, and S <= min_j KL_ij, so a
    row stops climbing only when it is κ nats (in soft-min units) from
    EVERYTHING in the pool. In nearest-neighbor KL units the effective climb
    envelope WIDENS with the pool: up to κ + τ·log J_eff (τ=0.5, J=6 ⟹
    ≈κ+0.9 nats). exp(min(S,κ)) <= e^κ still bounds the DP-prior weight
    ratio and the kernel weights sum to 1, so no dose explosion -- but dose
    transfer off the mean_S telemetry must account for the soft-min units.
  * The cloud persists on the planner instance for the whole rollout
    process (keyed by scene init_idx; shard workers own disjoint scene
    sets). ``reset()`` (called once before the loop) drops only the
    per-chunk reference tensors, never the cloud.

Implementation notes (subclass contract, the KLCostPlanner pattern): KLCostPlanner
owns ALL of phase 1 (select_z anchor capture, capped climb, eta_tilde
normalization, merged single-backward guided_step). This subclass swaps ONLY
the scalar being climbed: ``_kl_backward`` returns the per-row soft-min S
instead of the anchor-only KL. The reference pool is rebuilt per chunk in
select_z (BEFORE committing the new chunk-0 anchors, so a row's own fresh
anchor appears exactly once as pool column 0). No RNG is consumed anywhere:
bit-comparability with the other KL-cost arms on the same seed is preserved.
"""

from __future__ import annotations

import torch

from scout.guidance.entropy_costs import KLCostPlanner, _enc_forward


class CloudRepCostPlanner(KLCostPlanner):
    """cloudrep: soft-min KL to the policy's own per-scene anchor cloud
    (current chunk anchor + chunk-0 anchors of the scene's earlier
    retries)."""

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0, eta_dimless: bool = False,
                 cloud_tau: float = 0.5, cloud_max: int = 8):
        super().__init__(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                         cap=cap, eta_dimless=eta_dimless)
        self.cloud_tau = float(cloud_tau)
        self.cloud_max = int(cloud_max)
        if not (self.cloud_tau > 0.0):
            raise ValueError(
                f"cloud_tau must be > 0 (soft-min temperature); got "
                f"{cloud_tau}")
        if self.cloud_max < 1:
            raise ValueError(
                f"cloud_max must be >= 1 (pool capacity per scene); got "
                f"{cloud_max}")
        # scene init_idx -> list[(mu, logvar)] chunk-0 anchors of its
        # retries (detached constants, same treatment as _base_mu/_base_lv).
        self._cloud: dict = {}
        # (init_idx, try_idx) whose chunk-0 anchor has been committed --
        # the once-per-retry commit guard (a retry's later chunks must not
        # re-commit their fresh anchors).
        self._committed: set = set()
        # (init_idx, try_idx) per batch row, refreshed by rollout_vec's
        # set_row_jobs before EVERY replan batch (empty on direct/smoke
        # calls -- select_z then guards through the None-key branch).
        self._row_jobs: list = []
        # scene init_idx -> highest try_idx whose chunk-0 anchor has been
        # committed (= the retry has STARTED). Drives try_started_gate: the
        # j-axis ordering contract (retry k's chunk-0 pool must contain the
        # anchors of retries < k) WITHOUT finished-serialization -- try k
        # becomes admissible seconds after try k-1 launched, so slots stay
        # full whenever scenes >= n_envs (the probe regime).
        self._started: dict = {}
        # per-chunk reference pool for the CURRENT replan batch:
        # (B, J, dz) posterior tensors, column 0 = the fresh anchor; J-1
        # padded cloud columns + (B, J) validity mask. Rebuilt in select_z.
        self._ref_mu: torch.Tensor | None = None
        self._ref_lv: torch.Tensor | None = None
        self._ref_valid: torch.Tensor | None = None
        # telemetry accumulators (printed on the base [kl-telemetry] cadence).
        self._crep_calls = 0
        self._pool_acc = 0.0
        self._s_acc: torch.Tensor | None = None

    # ------------------------------------------------------------------ #
    # row context (rollout_vec._replan, fires before every batched call)
    # ------------------------------------------------------------------ #
    def set_row_jobs(self, jobs):
        """Full job tuples (state, init_idx, try_idx), row-aligned with the
        incoming replan batch -- rollout_vec._replan calls this immediately
        before dp.predict_action_dyn_guided, and select_z's x0_hat rows
        carry the same order (all slots replanning at a tick share one
        denoise loop)."""
        self._row_jobs = [(j[1], j[2]) for j in jobs]

    # ------------------------------------------------------------------ #
    # per-chunk: anchor capture (base) + pool rebuild + once-per-retry commit
    # ------------------------------------------------------------------ #
    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        super().select_z(x0_hat, current_obs)
        B = len(self._base_mu)
        if B == 0:
            return None
        # 1) rebuild the reference pool from the cloud AS IT IS (previous
        #    retries' chunk-0 anchors only): column 0 = fresh anchor, so a
        #    row's own chunk-0 anchor enters the pool exactly once.
        pools = [
            self._cloud.get(self._row_jobs[i][0] if i < len(self._row_jobs)
                            else None, [])[-self.cloud_max:]
            for i in range(B)]
        J = 1 + max((len(p) for p in pools), default=0)
        dz = self._base_mu[0].shape[-1]
        dev, dt = self._base_mu[0].device, self._base_mu[0].dtype
        ref_mu = torch.zeros(B, J, dz, device=dev, dtype=dt)
        ref_lv = torch.zeros(B, J, dz, device=dev, dtype=dt)
        valid = torch.zeros(B, J, device=dev, dtype=torch.bool)
        for i in range(B):
            ref_mu[i, 0] = self._base_mu[i]
            ref_lv[i, 0] = self._base_lv[i]
            valid[i, 0] = True
            for k, (m, lv) in enumerate(pools[i]):
                ref_mu[i, 1 + k] = m.to(dev, dt)
                ref_lv[i, 1 + k] = lv.to(dev, dt)
                valid[i, 1 + k] = True
        self._ref_mu, self._ref_lv, self._ref_valid = ref_mu, ref_lv, valid
        # 2) once-per-retry commit: rows whose (scene, try) has not been
        #    seen before are at their chunk 0 -- their fresh anchor joins
        #    the scene cloud for later chunks and later retries, and the
        #    scene's started-high-water advances for the start-gate.
        for i in range(B):
            key = (self._row_jobs[i] if i < len(self._row_jobs)
                   else (None, None))
            if key[0] is None or key in self._committed:
                continue
            self._cloud.setdefault(key[0], []).append(
                (self._base_mu[i].detach().clone(),
                 self._base_lv[i].detach().clone()))
            self._committed.add(key)
            self._started[key[0]] = max(self._started.get(key[0], -1),
                                        int(key[1]))
        return None

    def try_started_gate(self, init_idx, try_idx) -> bool:
        """Start-gate consumed by rollout_vec's job-gate dispatch (review
        P0-1 fix): try k of a scene is admissible once tries < k have
        STARTED (committed their chunk-0 anchor), not once they finished.
        This enforces the j-axis ordering the cloud semantics needs --
        retry k's chunk-0 pool provably contains retries < k's anchors --
        at zero slot-parallelism cost (a finished-serialization gate would
        idle slots whenever scenes < n_envs)."""
        return (int(try_idx) == 0
                or self._started.get(int(init_idx), -1) >= int(try_idx) - 1)

    def reset(self):
        """Base reset (clear s̄_t / z caches) + drop the per-chunk pool. The
        per-scene CLOUD survives: it is cross-retry state on a planner
        instance that lives for the whole rollout process."""
        super().reset()
        self._ref_mu = None
        self._ref_lv = None
        self._ref_valid = None

    # ------------------------------------------------------------------ #
    # the per-row soft-min over the pool (B,)
    # ------------------------------------------------------------------ #
    def _cloud_rows(self, mu: torch.Tensor, logvar: torch.Tensor,
                    x0_hat: torch.Tensor):
        """Returns the per-row climbed scalar ``S`` (B,) (graph-connected
        zeros on the guard path -- direct calls before select_z, mirroring
        ``_kl_rows``' None-baseline contract; unreachable via the public
        rollout path).

        Elementwise the (B, J) KL matrix matches ``_kl_rows`` per (row,
        reference column): 0.5*sum_d[(mu-m_j)^2/var_j + var/var_j - 1 -
        (logvar-lv_j)]. The masked soft-min uses logsumexp with -inf
        logits on padded columns (exp(-inf)=0: padding contributes neither
        value nor gradient; column 0 is always valid)."""
        zero = x0_hat.flatten(1).sum(dim=1).to(mu.dtype) * 0.0
        if (self._ref_mu is None or self._ref_lv is None
                or self._ref_valid is None
                or self._ref_mu.shape[0] != mu.shape[0]):
            return zero
        var = torch.exp(logvar).unsqueeze(1)                    # (B, 1, dz)
        var0 = torch.exp(self._ref_lv)                          # (B, J, dz)
        kl = 0.5 * (((mu.unsqueeze(1) - self._ref_mu) ** 2 / var0)
                    + (var / var0) - 1.0
                    - (logvar.unsqueeze(1) - self._ref_lv)).sum(dim=-1)  # (B,J)
        logits = (-kl / self.cloud_tau).masked_fill(
            ~self._ref_valid, float("-inf"))
        return -self.cloud_tau * torch.logsumexp(logits, dim=1)  # (B,)

    # ------------------------------------------------------------------ #
    # override: swap the climbed scalar, keep everything else verbatim
    # ------------------------------------------------------------------ #
    def _kl_backward(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                     current_obs=None):
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        s = self._cloud_rows(mu, logvar, x0_hat)
        g = torch.autograd.grad(s.sum(), trajectory)[0]
        # telemetry tick (mirrors the base [kl-telemetry] cadence).
        self._crep_calls += 1
        if self._ref_valid is not None:
            self._pool_acc += float(self._ref_valid.shape[1])
        _s_mean = s.detach().mean()
        self._s_acc = (_s_mean if self._s_acc is None
                       or self._s_acc.device != _s_mean.device
                       else self._s_acc + _s_mean)
        if self._crep_calls % 2500 == 0:
            print(f"[crep-telemetry] calls={self._crep_calls} "
                  f"mean_pool={self._pool_acc / self._crep_calls:.1f} "
                  f"mean_S={float(self._s_acc) / self._crep_calls:.4g} "
                  f"tau={self.cloud_tau} cloud={len(self._cloud)}scenes",
                  flush=True)
        return s, g

    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        """Read-only view of the cloudrep row cost (NO commit, NO pool
        rebuild), for monitoring / diagnostics; the rollout hot path uses
        guided_step. Mirrors its row_losses (-clamp(S, kappa))."""
        s_bar_t = self._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        s = self._cloud_rows(mu, logvar, x0_hat)
        nll = -torch.clamp(s, max=float(self.cap))
        if reduction == "mean":
            return nll.mean()
        if reduction == "sum":
            return nll.sum()
        raise ValueError(reduction)
