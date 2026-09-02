"""Orbit guidance: constrained control on the kappa-shell (user 2026-08-31).

Research frame: ``idea/escape_coverage_research.md`` (约束控制章). The entropy
cost's climb is ARGMAX control -- every retry converges onto one ridge of the
J^T.Lambda.J landscape (narrow cone). Particle guidance widens the ensemble by
mutual repulsion; orbit instead changes the SINGLE-trajectory control law:
once a retry reaches the escape shell {KL >= kappa} it STAYS there and tours
the shell, instead of continuing to push radially outward.

Derivation chain (math session, msg 42):

  1. KL-control / path-integral control (Kappen 2005; Todorov 2006; Dvijotham
     UAI 2011): a diffusion process under KL-regularized control has an HJB
     equation that linearizes under the log transform; the optimal control
     field is u*(x) ∝ grad log V(x).
  2. Classifier guidance's injected grad f is the greedy (argmax) approximation
     of grad log V. Swapping the peak-shaped reward (maximize f) for a
     plateau-shaped one (REACH {f >= kappa}) turns argmax control into
     CONSTRAINT control -- same variational family as the entropy cost
     (paper narrative: "from argmax control to constrained control").
  3. Constrained-manifold sampling: Zappa Holmes-Cerfon & Goodman, "Monte
     Carlo on Manifolds" (2018); Holmes-Cerfon 2024; constrained Langevin with
     Lagrange multipliers (Leimkuhler & Matthews).
  4. Two-phase injection, decided per row at EVERY guided denoise step
     (f = UNCAPPED KL(q(z|s,a) || q(z|s,a^DP)), kappa = atypical cap):
       (i)  f < kappa - delta : the standard climb, verbatim
            (eta * sqrt(1-abar_t) * grad f) -- identical to atypical;
       (ii) f >= kappa - delta: constrained dynamics --
            - tangential noise xi_perp (the grad-f normal component projected
              out): direction coverage comes from touring the level set;
            - Newton feedback -lam*(f-kappa)*grad f / ||grad f||^2, pinning f
              at kappa from either side (never pushes past the shell, so the
              particle-repulsion overdose cliff cannot occur by construction).
  5. The Newton step is reparametrization-invariant: computed with
     g = df/dx_t it equals the x0_hat-space Newton step (the 1/sqrt(abar)
     Jacobian of pred_original_sample cancels between the numerator and the
     ||g||^2 denominator) -- under the frozen-eps (affine x0_hat in x_t)
     approximation; the actual graph also differentiates through the UNet
     Jacobian, so the equivalence is approximate to the same order as every
     other injection here (the climb included). The tangential noise is
     scaled by sqrt(1-abar_t) to match the injection convention (anneals
     like the DDPM process noise; sigma_orb is its own dose knob,
     INDEPENDENT of eta).
  6. Phase 2 REPLACES the climb on its rows (the feedback itself keeps
     climbing below kappa -- -lam*(f-kappa) is positive along grad f when
     f < kappa), so there is no double dose at the hand-over. Note the
     hand-over is DIRECTIONAL, not force, continuity: at the boundary the
     injected magnitude jumps from eta*sqrt(1-abar)*||grad f|| to
     lam*|f-kappa|/||grad f||, and the Newton term deliberately carries no
     sqrt(1-abar_t) annealing (scale-free step) -- so late-denoise feedback
     is relatively stronger than the vanishing climb/noise. Dose calibration
     should read the telemetry's mean|fb| and mean|noise| separately.

No-op sentinel: (orbit_lam=0, orbit_sigma=0, orbit_delta=0) is bit-identical
to --guide atypical (phase-2 rows then have zero capped-climb gradient anyway,
no noise is drawn, and the extra backward consumes no RNG).
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import torch

from scout.guidance.entropy_costs import (AtypicalCostPlanner, _enc_forward,
                                          _kl_rows)
from scout.guidance.planner import ScoutPlanner


def orbit_displacement(kl: torch.Tensor, g: torch.Tensor, kappa: float,
                       lam: float, delta: float, sigma: float,
                       noise_scale: float = 1.0,
                       xi_override: Optional[torch.Tensor] = None,
                       fb_clamp: str = "none"):
    """Pure phase-2 math (unit-tested directly): given the per-row UNCAPPED
    cost ``kl`` (B,) and its per-row gradient w.r.t. the trajectory ``g``
    (B, T, Da) -- block-diagonal, so row slices are per-row gradients of the
    summed backward -- return ``(disp, phase2, stats)``:

      disp    (B, T, Da) displacement, zero on phase-1 rows;
      phase2  (B,) float mask (1 = constrained dynamics, 0 = keep the climb);
      stats   (fb_norm, noise_norm): per-row L2 norms of the two parts
              (detached; telemetry only).

    ``sigma=0`` draws NO random numbers (bit-identity with atypical holds).
    ``xi_override`` (B, T, Da) replaces the i.i.d. ``randn`` draw with a
    caller-supplied per-row direction (sector mode): it is projected against
    the CURRENT row normal exactly like the i.i.d. draw, so a fixed vector
    yields a persistent great-circle walk on the shell instead of a random
    walk. ``None`` -> the original i.i.d. draw (bit-identical).

    ``fb_clamp`` (user 2026-09-02, option C): "soft" soft-clamps the Newton
    residual  (kl - kappa) -> delta * tanh((kl - kappa)/delta)  so the
    feedback pull saturates at lam*delta/||g|| far off the shell instead of
    growing with the shell-saturated KL (chain VIB KL distributions drift
    right -- late-round rows sit far above kappa and the unbounded residual
    turned the feedback into large per-step jerks: sqR5 fb=0.55, canR2
    fb=0.61 vs the healthy 0.24-0.33). In-band (|kl-kappa| << delta) tanh is
    linear to O(x^3/delta^2) -- the cold-start regime is untouched
    analytically. The tangential NOISE is additionally restricted to the
    band [kappa-delta, kappa+delta] (off-band rows are not on the shell;
    touring there is meaningless) -- the randn is still DRAWN and the
    component zeroed afterwards, so the RNG stream stays identical to the
    other modes (bit-comparability across arms). "none" (default) is
    bit-identical legacy.
    """
    kl_d = kl.detach()
    p2 = kl_d >= (float(kappa) - float(delta))
    gnorm2 = (g.detach() ** 2).flatten(1).sum(dim=1)             # (B,)
    small = gnorm2 < 1e-16            # flat rows: Newton direction undefined
    safe = gnorm2.clamp(min=1e-16)
    if str(fb_clamp) == "soft":
        resid = float(delta) * torch.tanh(
            (kl_d - float(kappa)) / float(delta))
    elif str(fb_clamp) == "none":
        resid = kl_d - float(kappa)
    else:
        raise ValueError(f"fb_clamp must be 'none' or 'soft', got "
                         f"{fb_clamp!r}")
    fb_coeff = torch.where(small, torch.zeros_like(kl_d),
                           -float(lam) * resid / safe)
    fb = fb_coeff[:, None, None] * g                               # (B,T,Da)
    noise = torch.zeros_like(g)
    if float(sigma) > 0.0:
        # flat rows: ghat exactly 0 -> UNPROJECTED (full) noise; near-flat
        # rows above the threshold keep their (unit) direction (review P2).
        ghat = torch.where(small[:, None, None], torch.zeros_like(g),
                           g / safe.sqrt()[:, None, None])
        if xi_override is None:
            xi = torch.randn(g.shape, device=g.device, dtype=g.dtype)
        else:
            if tuple(xi_override.shape) != tuple(g.shape):
                raise ValueError(
                    f"sector xi shape {tuple(xi_override.shape)} != batch g "
                    f"{tuple(g.shape)} -- row jobs desynced from replan batch")
            xi = xi_override.detach().to(device=g.device, dtype=g.dtype)
        dot = (xi * ghat).flatten(1).sum(dim=1)
        noise = float(noise_scale) * float(sigma) * (xi - dot[:, None, None] * ghat)
        if str(fb_clamp) == "soft":
            band = ((kl_d >= float(kappa) - float(delta))
                    & (kl_d <= float(kappa) + float(delta)))
            noise = noise * band.to(noise.dtype)[:, None, None]
    mask = p2.to(g.dtype)[:, None, None]
    disp = (fb + noise) * mask
    stats = ((fb_coeff.abs() * safe.sqrt()).detach(),
             noise.detach().flatten(1).norm(dim=1))
    return disp, p2.to(g.dtype), stats


class OrbitCostPlanner(ScoutPlanner):
    """Entropy cost + two-phase constrained control (orbit).

    The phase-1 climb (per-chunk KL-to-own-intent, capped) and the phase-2
    constrained update are computed TOGETHER by :meth:`orbit_step` -- the
    duck-typed hook consumed by policy.py's injection line -- from ONE
    encoder forward and ONE backward through the shared graph (merged
    2026-09-01; the pre-merge code ran a capped compute_loss backward PLUS a
    second uncapped forward+backward, both traversing the UNet, per guided
    denoise step). Rows are INDEPENDENT (no inter-particle coupling) -- no
    group lock, i.i.d. retries exactly as atypical.
    """

    def __init__(self, scout_vib, bridge=None, obs_adapter=None,
                 cap: float = 10.0,
                 orbit_lam: float = 0.5, orbit_delta: float = 0.25,
                 orbit_sigma: float = 0.25,
                 orbit_sector: str = "iid", orbit_sector_seed: int = 42,
                 orbit_noise_anneal: float = 1.0,
                 orbit_climb: str = "grad", orbit_ray_seed: int = 42,
                 orbit_grad_norm: bool = False,
                 orbit_round: int = 1,
                 orbit_sigma_decay: float = 1.0,
                 orbit_fb_clamp: str = "none"):
        super().__init__(scout_vib, bridge=bridge, z=None,
                         obs_adapter=obs_adapter)
        self._att = AtypicalCostPlanner(scout_vib, bridge=bridge,
                                        obs_adapter=obs_adapter, cap=cap)
        self.orbit_lam = float(orbit_lam)
        # guard: kappa - delta <= 0 would put EVERY row (KL~0 included) in
        # phase 2 and turn the run into unprojected noise-everything; clamp
        # delta just under kappa instead (review P1).
        if float(cap) - float(orbit_delta) <= 0.0:
            orbit_delta = max(float(cap) - 1e-6, 0.0)
            print(f"[orbit] WARNING: orbit_delta >= cap "
                  f"({orbit_delta} >= {cap}); clamped to {orbit_delta} "
                  f"-- phase 2 would otherwise swallow every row",
                  flush=True)
        self.orbit_delta = float(orbit_delta)
        self.orbit_sigma = float(orbit_sigma)
        # eta-dimless mode (2026-09-02, orbit-hparam-dev): normalize the
        # climb gradient by the LIVE-CLIMB MEAN per-row gradient norm
        # (rows with kl < cap - delta and norm > 1e-4; see orbit_step for
        # the guards) before the policy's guidance_scale multiplies it. The
        # injected climb becomes
        #   eta_tilde * sqrt(1-abar_t) * (g / g_med)
        # i.e. a fixed ACTION-SPACE displacement per step, so one eta_tilde
        # transfers across tasks whose VIB gradient scales differ (tool_hang
        # needed eta=12 vs 3.0 on square/can -- a per-task hand calibration
        # the live-climb mean absorbs automatically). Telemetry: the policy's
        # mean_inject (batch-flattened F-norm of the injection) becomes
        # eta_tilde * sqrt(1-abar_t) * ||g||/g_med-scaled -- on the task used
        # for calibration it lands at the legacy value when
        # eta_tilde = eta_legacy * g_med_task. OFF (default) = bit-identical
        # legacy injection. Phase 2 is untouched: the Newton term carries its
        # own ||grad||^2 normalization and the tangent noise is sigma-scaled,
        # both already dimensionless w.r.t. the gradient scale.
        self.orbit_grad_norm = bool(orbit_grad_norm)
        # Round-dependent sigma ceiling (user order 2026-09-02): the chain's
        # retrained VIB co-adapts to the guidance's rescued trajectories, so
        # the DP/VIB stack settles onto the rescue ridge -- late rounds the
        # tangential noise only kicks retries OFF that ridge (orbit rescue
        # 36 -> 17 -> 12 -> 14 -> 0 on square while atypical stays 22 at the
        # same trio). Effective ceiling:
        #     sigma_eff(round) = orbit_sigma * orbit_sigma_decay ** (round-1)
        # decay=1.0 (default) = round-independent = bit-identical legacy.
        # Compose with --orbit-noise-anneal p for the per-DENOISE-step
        # exponent (noise carries (1-abar_t)^(p/2), already shipped as B3).
        if int(orbit_round) < 1:
            raise ValueError(f"orbit_round must be >= 1, got {orbit_round!r}")
        if not (0.0 < float(orbit_sigma_decay) <= 1.0):
            raise ValueError(f"orbit_sigma_decay must be in (0, 1], got "
                             f"{orbit_sigma_decay!r}")
        self.orbit_round = int(orbit_round)
        self.orbit_sigma_decay = float(orbit_sigma_decay)
        self.orbit_sigma_eff = float(orbit_sigma) * (
            self.orbit_sigma_decay ** (self.orbit_round - 1))
        # fb soft-clamp (user 2026-09-02, option C): saturate the Newton
        # residual at delta (tanh) + restrict the tangential noise to the
        # band -- see orbit_displacement. "none" (default) = bit-identical.
        if str(orbit_fb_clamp) not in ("none", "soft"):
            raise ValueError(f"orbit_fb_clamp must be 'none' or 'soft', "
                             f"got {orbit_fb_clamp!r}")
        self.orbit_fb_clamp = str(orbit_fb_clamp)
        if 0.0 < self.orbit_sigma_eff < 1e-12:
            # the round schedule switched the noise OFF numerically -- snap
            # to exact 0.0 so orbit_displacement draws NO randn (a residual
            # 1e-19 magnitude would still shift the trajectory RNG stream;
            # check 18c). Guard above requires orbit_sigma > 0 to reach here.
            self.orbit_sigma_eff = 0.0
        # sector mode (B2, 2026-08-31 beat-SOE campaign): "iid" = per-step
        # i.i.d. tangent noise (original, bit-identical default); "det" =
        # per-(init, try) DETERMINISTIC direction vector, cached for the
        # whole retry -- retries tour different great circles on the kappa
        # shell (stratified angular coverage) instead of random walking a
        # shared distribution.
        if str(orbit_sector) not in ("iid", "det"):
            raise ValueError(f"orbit_sector must be 'iid' or 'det', got "
                             f"{orbit_sector!r}")
        self.orbit_sector = str(orbit_sector)
        self.orbit_sector_seed = int(orbit_sector_seed)
        # B3 (beat-SOE campaign): tangent-noise annealing exponent on the
        # sqrt(1-abar_t) scale -- 1.0 = original (bit-identical), >1 decays
        # the noise faster through the denoise trajectory (jerk lever).
        if float(orbit_noise_anneal) <= 0.0:
            raise ValueError(f"orbit_noise_anneal must be > 0, got "
                             f"{orbit_noise_anneal!r}")
        self.orbit_noise_anneal = float(orbit_noise_anneal)
        # B4 ray mode (2026-09-01, user-designated after the mix failed stage
        # 2): "grad" = phase-1 climb keeps the steepest direction for EVERY
        # retry (original, bit-identical default); "ray" = retry 0 (gamma_0)
        # keeps the steepest climb verbatim, retry k >= 1 climbs along a FIXED
        # unit direction u_k from a deterministic max-min design -- the
        # rank-1 magnitude-restored field v = ||g||*sgn(<g,u_k>)*u_k keeps
        # df/dt = ||g||*|<ghat,u_k>| >= 0 (monotone for any terrain) while
        # making retries independent direction draws (wide explore). See
        # idea/escape_coverage_research.md section 7.
        if str(orbit_climb) not in ("grad", "ray"):
            raise ValueError(f"orbit_climb must be 'grad' or 'ray', got "
                             f"{orbit_climb!r}")
        self.orbit_climb = str(orbit_climb)
        self.orbit_ray_seed = int(orbit_ray_seed)
        self._row_jobs: Optional[List] = None   # [(state, init_idx, try_idx)]
        self._sector_vec = {}                   # (init_idx, try_idx) -> cpu tensor
        self._sector_warned = False
        # ray mode (B4): deterministic max-min direction design, keyed by
        # try_idx only (global design -- every scene's retry k climbs the SAME
        # u_k; per-scene keying would need 100x the cache for no coverage
        # gain). Cached per (try_idx, shape); generated from a dedicated CPU
        # Generator (does NOT follow the global rescue-seed reseed -- pass
        # --orbit-ray-seed to decorrelate confirmation runs, same class as
        # sector=det).
        self._ray_design: Optional[List] = None # try_idx -> design vector
        self._ray_stack: Optional[torch.Tensor] = None  # on-device (K,T,Da)
        self._ray_stack_key: Optional[tuple] = None
        self._ray_warned = False
        # ray telemetry: |cos(g,u_k)| sum + rotated-row count, both
        # device-side; host sync only on the print tick (the c36e69f lesson:
        # per-call/per-row .item() serializes the stream thousands of times
        # per rollout).
        self._ray_acc: Optional[torch.Tensor] = None
        self._ray_cnt: Optional[torch.Tensor] = None
        self._ray_calls = 0
        # telemetry (dose calibration reads this): device-side accumulators,
        # host sync only on the print tick (the pg-telemetry pattern).
        self._orb_calls = 0          # orbit_step invocations
        self._orb_rows = 0           # rows seen (host int -- shape, no sync)
        self._orb_p2_rows = 0        # rows that entered phase 2 (print tick)
        self._p2_acc: Optional[torch.Tensor] = None    # device sum of p2
        self._fb_acc: Optional[torch.Tensor] = None    # sum per-row |fb| norm
        self._noise_acc: Optional[torch.Tensor] = None  # sum per-row noise norm
        # eta-dimless: last divisor + running mean (debug/probe surface)
        self._last_g_med: Optional[torch.Tensor] = None
        self._gmed_acc: Optional[torch.Tensor] = None
        # fb soft-clamp diagnostics (device-side; sync on print tick)
        self._sat_acc: Optional[torch.Tensor] = None
        self._gsh_n_acc: Optional[torch.Tensor] = None
        self._gsh_s_acc: Optional[torch.Tensor] = None

    @property
    def p2_rows(self) -> int:
        """Phase-2 row count so far (one host sync -- tests/probes only;
        production reads it off the orbit-telemetry print tick)."""
        return int(self._p2_acc) if self._p2_acc is not None else 0

    # -- context plumbing (rollout_vec._replan hooks) ---------------------- #
    def set_row_context(self, init_ids: Sequence):
        pass    # rows are independent; hook fires regardless (atypical idem)

    def set_row_jobs(self, jobs):
        """Per-replan row jobs ``[(state, init_idx, try_idx), ...]`` (engine
        calls this whenever the planner exposes it). Sector mode keys its
        deterministic direction cache on (init_idx, try_idx)."""
        self._row_jobs = list(jobs)

    def _sector_xi(self, g: torch.Tensor) -> Optional[torch.Tensor]:
        """Per-row deterministic direction stack for sector='det', or None
        (caller falls back to the i.i.d. draw). CPU generators -> device
        copy; cached per (init_idx, try_idx) so the vector persists across
        denoise steps AND chunks of the retry."""
        if self.orbit_sector != "det":
            return None
        jobs = self._row_jobs
        if not jobs:
            if not self._sector_warned:
                print("[orbit] WARNING: sector='det' but no row jobs from "
                      "the engine -- falling back to i.i.d. tangent noise",
                      flush=True)
                self._sector_warned = True
            return None
        rows = []
        shape = g.shape[1:]                     # (T, Da)
        for j in jobs:
            key = (int(j[1]), int(j[2]))
            v = self._sector_vec.get(key)
            if v is None or tuple(v.shape) != tuple(shape):
                seed = ((self.orbit_sector_seed * 1_000_003
                         + key[0] * 100_07 + key[1] * 7_919) & 0x7FFFFFFF)
                gen = torch.Generator()
                gen.manual_seed(seed)
                v = torch.randn(shape, generator=gen)
                self._sector_vec[key] = v
            rows.append(v)
        return torch.stack(rows).to(device=g.device, dtype=g.dtype)

    def _ray_design_dirs(self, n_dirs: int, shape) -> List:
        """Deterministic unit-vector design in the (T, Da) climb space:
        normalized Gaussian candidates + greedy max-min-angle sieve
        (section-7 'normalized Gaussian + max-min sieve' option; Fibonacci /
        t-design are low-dim luxuries). Design indices map to retries k>=1
        (k=0 is gamma_0 = the gradient itself). Cached; regenerated if the
        replan shape changes (should never happen within a run)."""
        if (self._ray_design is not None
                and len(self._ray_design) >= n_dirs
                and tuple(self._ray_design[0].shape) == tuple(shape)):
            return self._ray_design[:n_dirs]
        gen = torch.Generator()
        gen.manual_seed(int(self.orbit_ray_seed))
        cand = torch.randn((256,) + tuple(shape), generator=gen)
        cand = cand / cand.flatten(1).norm(dim=1).clamp(
            min=1e-12)[:, None, None]
        d = cand.flatten(1)                       # (256, T*Da), unit rows
        chosen = [int(torch.randint(d.shape[0], (1,), generator=gen))]
        for _ in range(n_dirs - 1):
            sims = d @ d[chosen].t()              # cosine to each chosen
            chosen.append(int(sims.max(dim=1).values.argmin()))
        self._ray_design = [cand[i] for i in chosen]
        return self._ray_design

    def ray_rotate(self, cond_grad: torch.Tensor) -> torch.Tensor:
        """Climb-direction hook (duck-typed, consumed by policy.py right
        before the injection): retries k >= 1 climb along the fixed unit
        direction u_k with the magnitude-restored rank-1 field

            v_i = ||g_i|| * sgn(<ghat_i, u_k>) * u_k

        (identity: df/dt = ||g||*|<ghat, u>| >= 0 -- monotone on any terrain;
        full injection strength preserved). Retry 0 (gamma_0) and rows
        without row-jobs keep g verbatim; per-row norm is preserved exactly
        so the mean_inject dose telemetry stays comparable across modes.
        Phase-2 rows are unaffected semantically (policy.py zeroes their
        climb via _keep and swaps in the constrained update). Fully
        vectorized + device-side telemetry (no per-row host sync)."""
        if self.orbit_climb != "ray":
            return cond_grad
        jobs = self._row_jobs
        if not jobs:
            if not self._ray_warned:
                print("[orbit] WARNING: climb='ray' but no row jobs from the "
                      "engine -- keeping the gradient climb everywhere",
                      flush=True)
                self._ray_warned = True
            return cond_grad
        if len(jobs) != cond_grad.shape[0]:
            raise ValueError(f"ray row jobs {len(jobs)} != batch "
                             f"{cond_grad.shape[0]} -- replan batch desynced")
        self._ray_calls += 1
        g = cond_grad
        gnorm = g.detach().flatten(1).norm(dim=1)             # (B,)
        max_try = max(int(j[2]) for j in jobs)
        if max_try < 1:
            return g                                        # all gamma_0
        shape = tuple(g.shape[1:])
        dirs = self._ray_design_dirs(max_try, shape)
        key = (max_try, g.device, g.dtype, shape)
        if self._ray_stack is None or self._ray_stack_key != key:
            self._ray_stack = torch.stack(list(dirs)).to(
                device=g.device, dtype=g.dtype)
            self._ray_stack_key = key
        U = self._ray_stack
        kidx = torch.tensor([max(int(j[2]) - 1, -1) for j in jobs],
                            device=g.device)                 # -1 = gamma_0
        rot = (kidx >= 0) & (gnorm > 0)                      # (B,)
        U_sel = U[kidx.clamp(min=0)]                         # (B,T,Da)
        dot = (g.detach() * U_sel).flatten(1).sum(dim=1)     # (B,)
        sgn = torch.where(dot >= 0, torch.ones_like(dot),
                          -torch.ones_like(dot))
        rotated = gnorm[:, None, None] * sgn[:, None, None] * U_sel
        out = torch.where(rot[:, None, None], rotated, g)
        with torch.no_grad():
            # empty selection (all gamma_0 / all flat) sums to 0 -- no host
            # sync anywhere off the print tick.
            cos_abs = (dot.abs() / gnorm.clamp(min=1e-12))[rot].sum()
            n_rot = rot.sum().to(g.dtype)
            self._ray_acc = (cos_abs if self._ray_acc is None
                             or self._ray_acc.device != cos_abs.device
                             else self._ray_acc + cos_abs)
            self._ray_cnt = (n_rot if self._ray_cnt is None
                             or self._ray_cnt.device != n_rot.device
                             else self._ray_cnt + n_rot)
        if self._ray_calls % 2500 == 0 and self._ray_cnt is not None:
            print(f"[ray-telemetry] calls={self._ray_calls} "
                  f"rotated_rows={int(self._ray_cnt)} "
                  f"mean|cos(g,u_k)|={float(self._ray_acc) / max(int(self._ray_cnt), 1):.4g}",
                  flush=True)
        return out

    def select_z(self, x0_hat: torch.Tensor, current_obs=None):
        """Per-chunk intent anchor (mu^0, sigma^0^2) -- delegated verbatim to
        the atypical planner (captured at the FIRST guided denoise step)."""
        return self._att.select_z(x0_hat, current_obs)

    def set_current_obs(self, current_obs):
        self._att.set_current_obs(current_obs)

    # -- the cost (phase 1, verbatim atypical) ------------------------------ #
    def compute_loss(self, x0_hat: torch.Tensor, current_obs=None,
                     reduction: str = "mean") -> torch.Tensor:
        return self._att.compute_loss(x0_hat, current_obs, reduction)

    # -- the merged guided step (phase 1 + phase 2) ------------------------- #
    def orbit_step(self, trajectory: torch.Tensor, x0_hat: torch.Tensor,
                   current_obs=None, noise_scale: float = 1.0):
        """ONE encoder forward + ONE backward through the shared graph serve
        BOTH guidance consumers (perf 2026-09-01, 方案一; replaces the
        pre-merge compute_loss backward + a second ``_encode_and_uncapped``
        forward + second backward -- the guided denoise loop used to run 2
        VIB forwards and 2 UNet-traversing backwards per step; py-spy on the
        running s233 square orbit chain put ~55% of wall clock inside
        guided_conditional_sample, kernel-launch bound).

        Returns ``(cond_grad, disp, phase2, row_losses)``:

          * ``cond_grad`` -- the CAPPED atypical climb gradient, exactly what
            ``-autograd.grad(compute_loss(..., reduction="sum"))`` yielded
            pre-merge. Equivalence: the cap's gradient mask is
            ``1[KL <= kappa]`` (torch clamp(max) backward PASSES gradient at
            input == max -- empirically confirmed, review P1-1), and
            applying it AFTER the uncapped backward is exact because rows
            are block-diagonal -- a row's vjp never mixes with other rows,
            so a row at/below kappa keeps its full uncapped gradient
            (bitwise: ``where`` returns ``g`` verbatim on the selected
            branch), and a row above kappa gets exact zeros (the old
            zero-upstream backward; ``where`` additionally avoids the
            ``0 * inf = NaN`` pathology of a post-multiply). Only
            sub-kappa rows' cond_grad survives the policy's ``_keep``
            anyway (phase-2 rows swap the climb for ``disp``), and those
            are precisely the uncapped rows.
          * ``disp, phase2`` -- the constrained update, computed from the
            UNCAPPED ``(kl, g)`` verbatim as the pre-merge ``orbit_update``
            did; see :func:`orbit_displacement`.
          * ``row_losses`` -- detached per-row capped cost
            ``-min(KL, kappa)`` (B,) so the policy's optional cost-curve
            keeps its historical atypical-equivalent scale without a second
            forward.

        RNG streams are untouched (neither backward nor the mask consumes
        randomness; the xi tangent draw keeps its position as the step's
        only global-RNG draw, after the backward exactly as before).

        ``noise_scale`` arriving from policy.py is the injection convention
        sqrt(1-abar_t). ``orbit_noise_anneal=p`` (B3, beat-SOE campaign)
        raises it to the p-th power, i.e. the tangential noise carries
        (1-abar_t)^(p/2) instead of (1-abar_t)^(1/2): p>1 suppresses
        late-denoise (fine action detail) noise harder -- the jerk-inflation
        lever. p=1.0 passes the scalar through UNTOUCHED (bit-identical,
        no floating-point pow on the hot path)."""
        if float(self.orbit_noise_anneal) != 1.0:
            noise_scale = float(noise_scale) ** float(self.orbit_noise_anneal)
        s_bar_t = self._att._resolve_s_bar_t(current_obs)
        a = _enc_forward(self, x0_hat)
        mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
        kl = _kl_rows(mu, logvar, self._att._base_mu, self._att._base_lv,
                      x0_hat)
        g = torch.autograd.grad(kl.sum(), trajectory)[0]
        cap_mask = (kl.detach() <= float(self._att.cap)).view(
            -1, *([1] * (g.dim() - 1)))
        cond_grad = torch.where(cap_mask, g, torch.zeros_like(g))
        if self.orbit_grad_norm:
            # eta-dimless: divide by the MEAN per-row gradient norm over the
            # LIVE CLIMB rows (detached scalar; consumes no RNG, so streams
            # stay aligned with the legacy path). Three guards (review
            # 2026-09-02 + 20k-call can telemetry: an all-batch median let
            # the shell rows -- phase 2, LARGE norms -- inflate the divisor
            # and starve the climb 3x on can where p2=58%):
            #   * climb-only: a row injects only when kl < cap - delta
            #     (phase 2 swaps the climb for the constrained update, and
            #     policy's _keep zeroes it) -- the divisor must describe
            #     the rows that actually inject. Handover-band rows
            #     [cap-delta, cap) are excluded too (review round 2).
            #   * roundoff exclusion: rows with norm <= 1e-4 (at-anchor
            #     first-step rows: KL==0 exactly, gradients ~1e-8 roundoff)
            #     are dropped from the MEAN as well as floored in the
            #     division -- otherwise they drag the divisor down and
            #     re-amplify noise.
            #   * NaN containment: map NaN row-norms to 0 so they drop out
            #     of the weighted mean; if EVERY climb row is NaN the count
            #     clamps to 1 and g_med -> 0 -> floor -> zero climb (safe
            #     no-injection direction), never batch-wide NaN.
            #   * MEAN, not median: after normalization the per-row norm
            #     ||g_i||/mean is bounded by B (mean >= ||g_i||/B), so a
            #     single finite outlier row can only cause batch-wide
            #     UNDER-injection (safe: the DP prior dominates, and the
            #     mean_g_med telemetry makes it visible). A median divisor
            #     would allow unbounded over-injection of rows below it.
            # Real working-point row norms are O(0.1) (square/can chain
            # telemetry), three orders above the 1e-4 floor. ray_rotate
            # runs AFTER this in policy.py and preserves per-row norms, so
            # the normalized semantics survive the ray mode. Calibration
            # note (review P2): injected magnitudes are batch-coupled
            # through this statistic even though directions/RNG stay
            # row-independent.
            row_norms = g.detach().flatten(1).norm(dim=1)          # (B,)
            row_norms = torch.nan_to_num(row_norms, nan=0.0,
                                         posinf=0.0, neginf=0.0)
            live = ((kl.detach() < float(self._att.cap) - self.orbit_delta)
                    & (row_norms > 1e-4)).to(g.dtype)              # (B,)
            g_med = (row_norms * live).sum() / live.sum().clamp(min=1.0)
            cond_grad = cond_grad / g_med.clamp(min=1e-4)
            # Per-row cap at 3x nominal dose (2026-09-02, telemetry round):
            # the divisor describes EARLY-COMB climb rows (small norms);
            # handover-band rows [cap-delta, cap) keep nonzero cond_grad
            # with LARGER norms and would normalize to 10-50x nominal --
            # they do not inject (policy's _keep zeroes them) but they
            # dominate the injection telemetry and any future consumer
            # (ray_rotate). Clamp every normalized row to <= 3x so the
            # statistic, the telemetry and the (possible) injection all
            # stay in the eta_tilde band; normal rows (ratio ~1) are
            # untouched. Bit-identity of the OFF path is unaffected.
            row_ratio = cond_grad.detach().flatten(1).norm(dim=1)
            row_scale = (3.0 / row_ratio.clamp(min=1e-12)).clamp(max=1.0)
            cond_grad = cond_grad * row_scale.view(-1, *([1] * (g.dim() - 1)))
            self._last_g_med = g_med.clamp(min=1e-4).detach()
            self._gmed_acc = (self._last_g_med
                              if self._gmed_acc is None
                              or self._gmed_acc.device != self._last_g_med.device
                              else self._gmed_acc + self._last_g_med)
        disp, p2, (fb_n, noise_n) = orbit_displacement(
            kl, g, kappa=self._att.cap, lam=self.orbit_lam,
            delta=self.orbit_delta, sigma=self.orbit_sigma_eff,
            noise_scale=noise_scale,
            xi_override=(self._sector_xi(g) if self.orbit_sigma_eff > 0.0
                         else None),
            fb_clamp=self.orbit_fb_clamp)
        # telemetry (norms masked to phase-2 rows -- the dose numbers must
        # describe what was actually injected, not the pre-mask values).
        # Device-side accumulation only; the host sync happens on the print
        # tick (review P2: a per-call .item() costs ~3.7k stream
        # serializations per guided rollout).
        self._orb_calls += 1
        self._orb_rows += int(p2.shape[0])
        if self.orbit_fb_clamp != "none":
            # diagnostics for the soft clamp: rows saturated past the band
            # (|kl-kappa| >= 2*delta, where tanh(x/delta) >= 0.964) and the
            # mean ||g|| of phase-2 rows -- the shell-saturation readout the
            # fb dose analysis needs (lam*delta/||g||_shell is the soft
            # asymptotic pull). Device-side accumulators, host sync on the
            # print tick only.
            with torch.no_grad():
                _kl = kl.detach()
                _gn = g.detach().flatten(1).norm(dim=1)
                _p2m = p2.detach() > 0.5
                _sat = ((_kl >= float(self._att.cap) + 2.0 * self.orbit_delta)
                        & _p2m).to(g.dtype).sum()
                _gsh_n = _p2m.to(g.dtype).sum()
                _gsh_s = (_gn * _p2m.to(g.dtype)).sum()
                self._sat_acc = (_sat if self._sat_acc is None
                                 or self._sat_acc.device != _sat.device
                                 else self._sat_acc + _sat)
                self._gsh_n_acc = (_gsh_n if self._gsh_n_acc is None
                                   or self._gsh_n_acc.device != _gsh_n.device
                                   else self._gsh_n_acc + _gsh_n)
                self._gsh_s_acc = (_gsh_s if self._gsh_s_acc is None
                                   or self._gsh_s_acc.device != _gsh_s.device
                                   else self._gsh_s_acc + _gsh_s)
        for attr, val in (("_fb_acc", (fb_n * p2).sum()),
                          ("_noise_acc", (noise_n * p2).sum()),
                          ("_p2_acc", p2.sum())):
            cur = getattr(self, attr)
            if cur is None or cur.device != val.device:
                setattr(self, attr, val.detach().clone())
            else:
                setattr(self, attr, cur + val.detach())
        if self._orb_calls % 2500 == 0:
            self._orb_p2_rows = int(self._p2_acc)
            n = max(self._orb_p2_rows, 1)
            print(f"[orbit-telemetry] calls={self._orb_calls} "
                  f"p2_rows={self._orb_p2_rows}/{self._orb_rows} "
                  f"mean|fb|/p2row={float(self._fb_acc) / n:.4g} "
                  f"mean|noise|/p2row={float(self._noise_acc) / n:.4g}"
                  + ("" if self._gmed_acc is None
                     else f" mean_g_med={float(self._gmed_acc) / self._orb_calls:.4g}")
                  + ("" if self.orbit_fb_clamp == "none"
                     else (f" fbclamp={self.orbit_fb_clamp}"
                           f" sat_rows={int(self._sat_acc) if self._sat_acc is not None else 0}"
                           f" g_shell={float(self._gsh_s_acc) / max(int(self._gsh_n_acc), 1) if self._gsh_s_acc is not None else 0:.4g}")),
                  flush=True)
        row_losses = -torch.clamp(kl.detach(), max=float(self._att.cap))
        return cond_grad, disp, p2, row_losses
