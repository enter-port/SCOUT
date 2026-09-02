"""Hermetic dummy verify for SCOUT guidance wiring (task verify steps 1-4).

No robomimic / pytorch3d / LPB stack needed. Builds:

  - a dummy noise-prediction ``model`` (returns ``torch.randn_like`` of the
    trajectory -- enough to drive the DDPM scheduler, the exact values are
    irrelevant for the guidance-mechanism checks);
  - a real :class:`~scout.model.scout_vib.ScoutVIB` on a tiny mock ResNet
    encoder (same pattern as ``scout/train_vib.py:_MockResNetEncoder``);
  - a real :class:`diffusers.DDPMScheduler`;
  - a :class:`ScoutPlanner` with :class:`IdentityBridge` (both action spaces
    equal in the dummy);
  - a :class:`ScoutPolicy` shell via ``__new__`` (bypasses the LPB ``__init__``,
    which needs robomimic); the policy attrs (``model``, ``noise_scheduler``,
    ``num_inference_steps``, ``kwargs``, SCOUT planner attrs) are set by hand.

Checks (task verify):
  1. ``ScoutPlanner.compute_loss`` returns a scalar differentiable in ``x0_hat``.
  2. ``ScoutPolicy.guided_conditional_sample``:
       (a) ``classifier_guidance=False``  -> T_unguided;
       (b) ``guidance_scale=0``           -> == T_unguided (guidance off = no-op);
       (c) ``guidance_scale>0``           -> != T_unguided (guidance bites).
  3. Per-step cost curve over the guided steps is non-increasing on average /
     final < initial (sign of ``-autograd.grad`` and ``+scale*cond`` correct).
  4. Dummy {image, proprio} obs path works end-to-end.
  5. 1/B scaling-bug regression (fixed 2026-08-21): grad(sum-reduced cost)
     == B x grad(mean-reduced); a row alone (B=1) reproduces its batched
     gradient; the guided path wires ``reduction="sum"`` (spy).
  6. Expert z-bank mode (2026-08-21): ``select_z`` returns the argmin-NLL
     bank row; the policy selects z* exactly once per denoise loop (every
     action chunk -- NOT once per trajectory) and the guided trajectory
     differs from unguided; explore mode (checks 1-5, plain planner) is
     untouched.
  7-9. Particle guidance (2026-08-30): pg_start=never == atypical
     bit-identical; repulsion semantics; pg_start gate counters.
  10-12. Orbit guidance (2026-08-31): (lam, sigma, delta)=(0,0,0) ==
     atypical bit-identical; orbit_displacement pure-math unit tests
     (phase mask / Newton projection identity / damped overshoot formula /
     tangential orthogonality / flat-gradient guard); policy-path
     integration (per-step counter, determinism, forced far/equal
     baselines through the real encoder).
  16. Merged single-backward orbit_step (perf 方案一+二, 2026-09-01):
     bit-identical to the pre-merge two-backward algorithm (planner +
     policy level, telemetry parity); vectorized atypical row core equals
     the historical per-row loop (values + gradients, incl. missing
     baselines).

orbit-dev note (2026-09-01): this branch carries ONLY the orbit line
extracted from entropy-random-dev (cherry-picks of f639e4b/c36e69f/
d293f1a/283ef17/e4d7cc4/4211d97 onto main). Particle guidance and the
rand-cost registry are NOT on this branch -- checks 7-9 (particle) are
absent by construction and the check numbering keeps their slots empty
for cross-branch comparability of the check names.

Run:
    python -m scout.guidance._verify
"""

from __future__ import annotations

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scout.guidance.planner import ScoutPlanner
from scout.guidance.policy import ScoutPolicy, _LPB_AVAILABLE, _IMPORT_ERROR
from scout.model.encoder import StateEncoder
from scout.model.scout_vib import ScoutVIB
from scout.normalizer import IdentityBridge


# --------------------------------------------------------------------------- #
# dummy components
# --------------------------------------------------------------------------- #
class _DummyModel(nn.Module):
    """Noise-prediction stand-in for the LPB ``ConditionalUnet1D``.

    Same call signature ``model(trajectory, t, local_cond=, global_cond=)`` and
    returns Gaussian noise of the trajectory's shape (ε-prediction). Stateless
    beyond global RNG, so per-step output is reproducible by re-seeding
    ``torch.manual_seed`` before each run.
    """

    def forward(self, trajectory, t, local_cond=None, global_cond=None):
        return torch.randn_like(trajectory)


class _MockResNetEncoder(nn.Module):
    """Tiny stand-in for LPB ``ResNetEncoder`` (mirrors ``train_vib.py``).

    Duck-typed: ``emb_dim=512``; ``forward({view: (B,T,3,H,W)}) ->
    {view: (B,T,1,512)}``. Has parameters so the freeze check is meaningful.
    """

    def __init__(self, view_names):
        super().__init__()
        self.view_names = list(view_names)
        self.emb_dim = 512
        self.proj = nn.Conv2d(3, 512, kernel_size=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        from einops import rearrange

        out = {}
        for v in self.view_names:
            imgs = x[v]
            b = imgs.shape[0]
            imgs = rearrange(imgs, "b t ... -> (b t) ...")
            feat = self.avgpool(self.proj(imgs))     # (B*T, 512, 1, 1)
            feat = feat.flatten(1).unsqueeze(1)      # (B*T, 1, 512)
            out[v] = rearrange(feat, "(b t) p d -> b t p d", b=b)
        return out


# --------------------------------------------------------------------------- #
# harness
# --------------------------------------------------------------------------- #
def _build_harness(
    seed: int = 233,
    batch_size: int = 4,
    horizon: int = 8,
    action_dim: int = 7,
    style_dim: int = 16,
    proprio_dim: int = 10,
    view_names=("agentview", "robot0_eye_in_hand"),
    num_train_timesteps: int = 100,
    num_inference_steps: int = 10,
):
    """Build (policy, planner, scout_vib, current_obs, cond_data, cond_mask,
    global_cond) for the verify runs."""
    torch.manual_seed(seed)

    E_s = StateEncoder(
        resnet_encoder=_MockResNetEncoder(view_names),
        view_names=list(view_names),
        proprio_dim=proprio_dim,
        proprio_emb_dim=64,
    )
    scout_vib = ScoutVIB(
        action_dim=action_dim, E_s=E_s, style_dim=style_dim, beta=1.0e-3
    ).eval()

    planner = ScoutPlanner(scout_vib=scout_vib, bridge=IdentityBridge())

    # ScoutPolicy shell (LPB __init__ skipped -- no robomimic). When robomimic
    # IS present ScoutPolicy is a real nn.Module subclass, so Module.__init__
    # must run before any Module attribute assignment (else "cannot assign module
    # before Module.__init__"). On the no-robomimic host ScoutPolicy bases to
    # `object` and this branch is skipped.
    policy = ScoutPolicy.__new__(ScoutPolicy)
    if _LPB_AVAILABLE:
        torch.nn.Module.__init__(policy)
    policy.model = _DummyModel()
    policy.noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        prediction_type="epsilon",
        beta_schedule="scaled_linear",
    )
    policy.num_inference_steps = num_inference_steps
    policy.kwargs = {}
    policy.scout_planner = None
    policy.guidance_start_timestep = None
    policy.guidance_scale = None
    policy.initialize_scout_planner(
        planner=planner,
        guidance_start_timestep=num_train_timesteps,  # gate (a): guide on every step
        guidance_scale=1.0,
    )

    # dummy obs (E_s format -- one T=1 frame).
    current_obs = {
        "visual": {
            v: torch.randn(batch_size, 1, 3, 128, 128) for v in view_names
        },
        "proprio": torch.randn(batch_size, 1, proprio_dim),
    }

    # empty inpaint (LPB obs_as_global_cond=True path): cond_data zeros, mask all False.
    cond_data = torch.zeros(batch_size, horizon, action_dim)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    global_cond = None  # unused by _DummyModel

    return policy, planner, scout_vib, current_obs, cond_data, cond_mask, global_cond


def _run_sample(policy, cond_data, cond_mask, global_cond, current_obs,
                z, seed, classifier_guidance, guidance_scale,
                return_cost_curve=False):
    """Reset all RNG, (re)set planner state, then run one guided sample."""
    torch.manual_seed(seed)
    gen = torch.Generator(device=cond_data.device).manual_seed(seed)
    # force the planner's per-call state (z always pre-set; current_obs re-encoded)
    policy.scout_planner.reset()
    policy.guidance_scale = float(guidance_scale)
    return policy.guided_conditional_sample(
        cond_data, cond_mask,
        local_cond=None, global_cond=global_cond,
        generator=gen,
        classifier_guidance=classifier_guidance,
        current_obs=current_obs,
        z=z,
        return_cost_curve=return_cost_curve,
    )


# --------------------------------------------------------------------------- #
# checks
# --------------------------------------------------------------------------- #
def check_planner_compute_loss(policy, planner, scout_vib, current_obs, horizon, action_dim):
    """Verify step 1: compute_loss returns scalar, differentiable in x0_hat."""
    B = current_obs["proprio"].shape[0]
    x0_hat = torch.randn(B, horizon, action_dim, requires_grad=True)
    planner.reset()
    planner.set_z(torch.randn(B, scout_vib.style_dim))
    planner.set_current_obs(current_obs)

    loss = planner.compute_loss(x0_hat, current_obs)
    assert loss.dim() == 0, f"compute_loss must return a scalar; got shape {loss.shape}"
    grad = torch.autograd.grad(loss, x0_hat)[0]
    assert grad.shape == x0_hat.shape, (
        f"grad shape {grad.shape} != x0_hat shape {x0_hat.shape}"
    )
    assert torch.isfinite(grad).all(), "grad has NaN/Inf"
    assert grad.abs().sum() > 0, "grad is exactly zero (cost not wired to x0_hat)"
    print(f"[check 1] compute_loss scalar={loss.item():.4f}, "
          f"grad|sum|={grad.abs().sum().item():.4f}, differentiable: OK")


def check_guidance_onoff(policy, planner, scout_vib, current_obs,
                         cond_data, cond_mask, global_cond, seed=233):
    """Verify step 2: guidance-off matches unguided; guidance-on differs."""
    B = cond_data.shape[0]
    horizon, action_dim = cond_data.shape[1], cond_data.shape[2]
    # one fixed z across the three runs (so z is not a confound).
    torch.manual_seed(seed)
    z_fixed = torch.randn(B, scout_vib.style_dim)

    # (a) unguided (classifier_guidance=False).
    t_unguided = _run_sample(
        policy, cond_data, cond_mask, global_cond, current_obs,
        z=z_fixed, seed=seed,
        classifier_guidance=False, guidance_scale=0.0,
    )

    # (b) classifier_guidance=True but guidance_scale=0 -> must equal t_unguided.
    t_scale0 = _run_sample(
        policy, cond_data, cond_mask, global_cond, current_obs,
        z=z_fixed, seed=seed,
        classifier_guidance=True, guidance_scale=0.0,
    )
    max_diff_zero = (t_unguided - t_scale0).abs().max().item()
    assert max_diff_zero < 1e-5, (
        f"scale=0 should be a no-op but max|diff|={max_diff_zero:.2e}"
    )

    # (c) classifier_guidance=True, guidance_scale>0 -> must differ.
    t_guided = _run_sample(
        policy, cond_data, cond_mask, global_cond, current_obs,
        z=z_fixed, seed=seed,
        classifier_guidance=True, guidance_scale=5.0,
    )
    max_diff_guided = (t_unguided - t_guided).abs().max().item()
    assert max_diff_guided > 1e-3, (
        f"scale>0 should change the trajectory but max|diff|={max_diff_guided:.2e}"
    )

    print(f"[check 2] unguided vs scale=0: max|diff|={max_diff_zero:.2e} (no-op OK)")
    print(f"[check 3] unguided vs scale=5: max|diff|={max_diff_guided:.2e} (bites OK)")


def check_cost_curve(policy, planner, scout_vib, current_obs,
                     cond_data, cond_mask, global_cond, seed=233):
    """Verify step 3 (cost direction): final cost < initial cost across guided
    denoise steps (sign of ``-autograd.grad`` and ``+scale*cond`` correct)."""
    B = cond_data.shape[0]
    torch.manual_seed(seed)
    z_fixed = torch.randn(B, scout_vib.style_dim)

    _, curve = _run_sample(
        policy, cond_data, cond_mask, global_cond, current_obs,
        z=z_fixed, seed=seed,
        classifier_guidance=True, guidance_scale=5.0,
        return_cost_curve=True,
    )
    assert len(curve) >= 2, f"need >=2 guided steps for a curve; got {len(curve)}"
    head, tail = curve[0], curve[-1]
    # strictly monotonic is too strong with a random model + scheduler; the
    # sign-correctness check is "cost went DOWN overall".
    assert tail < head, (
        f"cost curve should fall (sign correct); got head={head:.4f} -> "
        f"tail={tail:.4f} (rising means flipped sign)"
    )
    print(f"[check 4] cost curve ({len(curve)} steps): "
          f"head={head:.4f} -> tail={tail:.4f} "
          f"(delta={tail-head:.4f}); falls: OK")
    # also report how many of the steps were decreases (robustness signal).
    decreases = sum(1 for a, b in zip(curve[:-1], curve[1:]) if b <= a)
    print(f"           per-step decreases: {decreases}/{len(curve)-1}")


def check_grad_batch_invariance(policy, planner, scout_vib, current_obs,
                                cond_data, cond_mask, seed=233):
    """Verify step 5 (1/B scaling-bug regression, fixed 2026-08-21): the
    per-row guidance gradient must be INDEPENDENT of the batch size B.

    Pre-fix, the guided path mean-reduced the cost over B, so every row's
    injected force was divided by B (effective guidance = guidance_scale/B;
    idea/guidance_batch_scaling_bug.md). Checks:

      (a) algebra: grad of the sum-reduced cost == B x grad of the mean-reduced
          cost (exact for the block-diagonal Jacobian);
      (b) row isolation: row i's gradient from a B=4 sum-reduced batched call
          equals its gradient from a B=1 call with that row's obs/z alone --
          i.e. a row receives the same force whether alone or batched;
      (c) wiring: the policy's guided path actually passes reduction="sum"
          (spy on planner.compute_loss during a guided sample).
    """
    B = current_obs["proprio"].shape[0]
    horizon, action_dim = cond_data.shape[1], cond_data.shape[2]
    torch.manual_seed(seed)
    z = torch.randn(B, scout_vib.style_dim)

    # (a) grad(sum) == B * grad(mean) on identical inputs.
    x0 = torch.randn(B, horizon, action_dim, requires_grad=True)
    planner.reset()
    planner.set_z(z.clone())
    planner.set_current_obs(current_obs)
    loss_sum = planner.compute_loss(x0, current_obs, reduction="sum")
    g_sum = torch.autograd.grad(loss_sum, x0, retain_graph=False)[0]

    x0b = x0.detach().clone().requires_grad_(True)
    planner.set_z(z.clone())                      # same z; s̄_t already cached
    loss_mean = planner.compute_loss(x0b, current_obs, reduction="mean")
    g_mean = torch.autograd.grad(loss_mean, x0b)[0]

    max_ratio_err = ((g_sum - B * g_mean).abs().max()
                     / (B * g_mean).abs().max()).item()
    assert torch.allclose(g_sum, B * g_mean, rtol=1e-4, atol=1e-6), (
        f"grad(sum) != B*grad(mean): max rel err {max_ratio_err:.2e}"
    )

    # (b) row i alone (B=1, mean==sum) must reproduce its batched gradient.
    def _slice_obs(obs, i):
        return {
            "visual": {v: x[i:i + 1] for v, x in obs["visual"].items()},
            "proprio": obs["proprio"][i:i + 1],
        }

    worst = 0.0
    for i in range(B):
        x0_row = x0[i:i + 1].detach().clone().requires_grad_(True)
        planner.reset()
        planner.set_z(z[i:i + 1].clone())
        planner.set_current_obs(_slice_obs(current_obs, i))
        loss_row = planner.compute_loss(x0_row, _slice_obs(current_obs, i))
        g_row = torch.autograd.grad(loss_row, x0_row)[0]
        worst = max(worst, (g_row - g_sum[i:i + 1]).abs().max().item())
    assert worst < 1e-5, (
        f"row-alone gradient deviates from batched sum-reduced gradient "
        f"(max abs diff {worst:.2e}) -- per-row force is still B-dependent"
    )

    # (c) spy: the guided path must request reduction="sum" on every call.
    recorded = []
    orig = planner.compute_loss

    def _spy(x0_hat, obs=None, **kw):
        recorded.append(kw.get("reduction", "mean"))
        return orig(x0_hat, obs, **kw)

    planner.compute_loss = _spy
    try:
        _run_sample(policy, cond_data, cond_mask, None, current_obs,
                    z=z.clone(), seed=seed,
                    classifier_guidance=True, guidance_scale=1.0)
    finally:
        planner.compute_loss = orig
    assert recorded and all(r == "sum" for r in recorded), (
        f"guided path must call compute_loss(reduction='sum'); got {recorded}"
    )

    print(f"[check 5] 1/B bug regression: grad(sum)==B*grad(mean) "
          f"(rel err {max_ratio_err:.1e}); row-alone == batched "
          f"(max diff {worst:.1e}); policy wiring reduction="
          f"{set(recorded)} over {len(recorded)} guided steps: OK")


def check_expert_guidance(policy, planner, scout_vib, current_obs,
                          cond_data, cond_mask, seed=233):
    """Verify step 6: expert z-bank mode (user 2026-08-21).

    (a) ``select_z`` returns, per batch row, the bank entry with the lowest
        NLL under the query ``q(z|s_bar, x0_hat)`` (verified exactly: the
        query crafted from bank row j's own (s,a) must re-select row j);
    (b) the policy's expert path selects z* exactly ONCE per denoise loop
        (per action chunk) via ``set_z`` -- plus the pre-loop ``set_z(None)``
        hygiene clear -- and every z* is a bank row;
    (c) the expert-guided trajectory differs from the unguided one.
    """
    from scout.guidance.expert_bank import ScoutExpertPlanner

    horizon, action_dim = cond_data.shape[1], cond_data.shape[2]
    B = current_obs["proprio"].shape[0]
    torch.manual_seed(seed)

    # bank: K style-vectors, each the encoder mean of batch-row 0's obs with
    # a distinct (dummy, 1-step) action chunk.
    obs0 = {"visual": {v: x[0:1] for v, x in current_obs["visual"].items()},
            "proprio": current_obs["proprio"][0:1]}
    K = 6
    A = torch.randn(K, 1, action_dim)                     # (K, 1, per_step)
    with torch.no_grad():
        s0 = scout_vib.encode(obs0)
        bank = torch.stack([
            scout_vib.vib_enc(s0, A[j:j + 1].reshape(1, -1))[0][0]
            for j in range(K)])                            # (K, D)

    expert = ScoutExpertPlanner(scout_vib, z_bank=bank)
    policy.initialize_scout_planner(
        planner=expert,
        guidance_start_timestep=policy.noise_scheduler.config.num_train_timesteps,
        guidance_scale=5.0,
    )

    # (a) argmin-NLL selection: query row 0 rebuilt from bank row 2's action.
    x0_q = torch.randn(B, horizon, action_dim)
    x0_q[0, 0] = A[2, 0]                                  # n_steps=1 chunk
    z_star = expert.select_z(x0_q, current_obs)
    assert torch.allclose(z_star[0], bank[2], atol=1e-5), (
        "select_z must re-select the bank row the query was built from"
    )
    for i in range(B):                                    # every z* is a bank row
        assert any(torch.allclose(z_star[i], bank[j], atol=1e-6)
                   for j in range(K)), f"row {i} z* not in bank"

    # (b) policy wiring: exactly one non-None set_z per denoise loop.
    set_z_calls = []
    orig_set_z = expert.set_z
    expert.set_z = lambda z: (set_z_calls.append(
        None if z is None else z.clone()), orig_set_z(z))
    try:
        t_expert = _run_sample(policy, cond_data, cond_mask, None, current_obs,
                               z=None, seed=seed,
                               classifier_guidance=True, guidance_scale=5.0)
    finally:
        expert.set_z = orig_set_z
    nonnone = [z for z in set_z_calls if z is not None]
    assert set_z_calls[0] is None, "pre-loop must clear stale z (set_z(None))"
    assert len(nonnone) == 1, (
        f"one z* selection per denoise loop expected; got {len(nonnone)} "
        f"(calls: {[None if z is None else tuple(z.shape) for z in set_z_calls]})"
    )
    assert all(any(torch.allclose(nonnone[0][b], bank[j], atol=1e-6)
                   for j in range(K)) for b in range(B)), "some z* not in bank"

    # (c) guidance bites: expert-guided != unguided at scale>0.
    t_unguided = _run_sample(policy, cond_data, cond_mask, None, current_obs,
                             z=None, seed=seed, classifier_guidance=False,
                             guidance_scale=0.0)
    diff = (t_expert - t_unguided).abs().max().item()
    assert diff > 1e-3, f"expert guidance must bite; max|diff|={diff:.2e}"

    print(f"[check 6] expert z-bank: argmin-NLL selection exact; "
          f"set_z(non-None) x{len(nonnone)} per loop (z* in bank); "
          f"guided vs unguided max|diff|={diff:.2e} (bites OK)")


def check_orbit_guidance(policy, scout_vib, current_obs,
                         cond_data, cond_mask, global_cond, seed=233):
    """Checks 10-12 (orbit guidance, user 2026-08-31, math session):

    (10) no-op sentinel: (lam, sigma, delta) = (0, 0, 0) is BIT-IDENTICAL to
         --guide atypical (rows at/above kappa carry zero capped-climb
         gradient anyway; sigma=0 draws no noise; the second backward
         consumes no RNG). cap=0.01 forces rows across the boundary so the
         masking branch is exercised, not just the RNG-free path.
    (11) orbit_displacement pure math on a hand-built quadratic bowl
         (KL = ||x||^2, grad = 2x -- analytic at every point):
           - phase mask: rows below kappa-delta get EXACTLY zero;
           - first-order projection identity  g.fb = -lam*(kl-kappa);
           - lambda=1 lands on  kappa + (kl-kappa)^2/(4*kl)  (the damped-
             Newton overshoot formula of one quadratic step);
           - tangential noise orthogonal to g on phase-2 rows, masked rows
             stay exactly zero;
           - flat-gradient guard (g=0): zero feedback, finite unprojected
             noise (no division blow-up).
    (12) policy path through the real mock encoder:
           - orbit_step fires exactly once per guided denoise step;
           - same-seed determinism, and the ACTIVE orbit (sigma>0) differs
             from its no-op partner (machinery bites);
           - forced-FAR baseline -> phase 2 fires for every row, nonzero
             displacement; anchor-equal input -> phase 2 never fires.
    """
    from scout.guidance.orbit_costs import OrbitCostPlanner, orbit_displacement
    from scout.guidance.entropy_costs import AtypicalCostPlanner

    gst = policy.noise_scheduler.config.num_train_timesteps

    # ---- (10) no-op sentinel == atypical, bit-for-bit --------------------- #
    t_out = {}
    for name, pl in (
            ("atypical", AtypicalCostPlanner(scout_vib, bridge=IdentityBridge(),
                                             cap=0.01)),
            ("orbit-off", OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                                           cap=0.01, orbit_lam=0.0,
                                           orbit_delta=0.0, orbit_sigma=0.0))):
        policy.initialize_scout_planner(planner=pl,
                                        guidance_start_timestep=gst,
                                        guidance_scale=1.0)
        t_out[name] = _run_sample(policy, cond_data, cond_mask, None,
                                  current_obs, z=None, seed=seed,
                                  classifier_guidance=True,
                                  guidance_scale=1.0)
    diff10 = (t_out["atypical"] - t_out["orbit-off"]).abs().max().item()
    assert diff10 == 0.0, (
        f"(10) orbit no-op sentinel must be bit-identical to atypical; "
        f"max|diff|={diff10:.3e}")

    # ---- (11) pure-math unit tests ---------------------------------------- #
    kappa, delta, lam = 2.5, 0.25, 0.5
    B, T, D = 4, 2, 3
    # r^2 = 0.64 (< 2.25, phase 1) | 2.2801 (phase 2, below shell) | 3.24,
    # 4.84 (phase 2, above shell)
    r = torch.tensor([0.8, 1.51, 1.8, 2.2], dtype=torch.float64)
    x = torch.zeros(B, T, D, dtype=torch.float64)
    x[:, 0, 0] = r
    kl = (x ** 2).flatten(1).sum(dim=1)
    g = 2.0 * x
    disp0, p2, (fb_n, _) = orbit_displacement(kl, g, kappa, lam, delta,
                                              sigma=0.0)
    assert float(p2[0]) == 0.0 and disp0[0].abs().max().item() == 0.0, (
        "(11) rows below kappa-delta must be exactly phase 1")
    assert all(float(p2[i]) == 1.0 for i in (1, 2, 3)), (
        "(11) rows at/above kappa-delta must be phase 2")
    # first-order projection identity: g . fb == -lam * (kl - kappa)
    dot = (g * disp0).flatten(1).sum(dim=1)
    for i in (1, 2, 3):
        assert abs(float(dot[i]) + lam * float(kl[i] - kappa)) < 1e-9, (
            f"(11) Newton projection identity broken on row {i}: "
            f"{float(dot[i])} vs {-lam * float(kl[i] - kappa)}")
    # lambda=1 one quadratic step: ||x + fb||^2 == kappa + (kl-kappa)^2/(4kl)
    disp1, _, _ = orbit_displacement(kl, g, kappa, 1.0, delta, sigma=0.0)
    new_kl = ((x + disp1) ** 2).flatten(1).sum(dim=1)
    expect = kappa + (kl - kappa) ** 2 / (4.0 * kl)
    assert torch.allclose(new_kl[1:], expect[1:], rtol=1e-9), (
        f"(11) damped-Newton overshoot formula mismatch: "
        f"{new_kl.tolist()} vs {expect.tolist()}")
    # tangential noise: deterministic fb (sigma=0) minus the sigma>0 call
    # isolates the noise part; it must be orthogonal to g on phase-2 rows,
    # and the phase-1 row stays exactly zero.
    disp_n, _, _ = orbit_displacement(kl, g, kappa, lam, delta, sigma=0.3,
                                      noise_scale=0.7)
    noise_part = disp_n - disp0
    assert disp_n[0].abs().max().item() == 0.0, (
        "(11) noise must not leak into phase-1 rows")
    dotn = (g * noise_part).flatten(1).sum(dim=1)
    gn = g.flatten(1).norm(dim=1)
    nn = noise_part.flatten(1).norm(dim=1)
    for i in (1, 2, 3):
        assert abs(float(dotn[i])) < 1e-6 * float(gn[i] * nn[i]), (
            f"(11) tangential noise not orthogonal to grad on row {i}")
    # flat-gradient guard: g = 0 -> no feedback blow-up, finite noise
    dispf, p2f, (fbf, _) = orbit_displacement(kl, torch.zeros_like(g), kappa,
                                              lam, delta, sigma=0.3)
    assert not torch.isnan(dispf).any(), "(11) flat-gradient guard NaN"
    assert float(fbf.abs().max()) == 0.0, "(11) flat rows must give zero fb"
    assert float(p2f.sum()) == 3.0 and dispf[1:].abs().max().item() > 0, (
        "(11) flat rows: phase 2 fires with unprojected noise")

    # ---- (12) policy-path integration ------------------------------------- #
    orb = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=0.01,
                           orbit_lam=0.5, orbit_delta=0.25, orbit_sigma=0.25)
    policy.initialize_scout_planner(planner=orb,
                                    guidance_start_timestep=gst,
                                    guidance_scale=1.0)
    t1 = _run_sample(policy, cond_data, cond_mask, None, current_obs,
                     z=None, seed=seed, classifier_guidance=True,
                     guidance_scale=1.0)
    assert orb._orb_calls == policy.num_inference_steps, (
        f"(12) expected {policy.num_inference_steps} orbit_step calls, "
        f"got {orb._orb_calls}")
    assert orb.p2_rows > 0, (
        "(12) cap=0.01 must push some rows into phase 2 on the mock path")
    t2 = _run_sample(policy, cond_data, cond_mask, None, current_obs,
                     z=None, seed=seed, classifier_guidance=True,
                     guidance_scale=1.0)
    assert torch.equal(t1, t2), "(12) same-seed orbit runs must be identical"
    assert (t1 - t_out["orbit-off"]).abs().max().item() > 0.0, (
        "(12) active orbit (sigma>0) must differ from its no-op partner")

    # forced-FAR baseline -> every row phase 2, nonzero displacement; the
    # anchor-equal input -> phase 2 never fires (real encoder Jacobian path).
    Bn = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]
    orb2 = OrbitCostPlanner(scout_vib, bridge=IdentityBridge())
    orb2.set_current_obs(current_obs)
    x0_seed = torch.randn(Bn, H, Ad)
    orb2.select_z(x0_seed, current_obs)
    orb2._att._base_mu = [m + 5.0 for m in orb2._att._base_mu]
    traj = torch.randn(Bn, H, Ad).requires_grad_(True)
    x0h = 2.0 * traj                      # any differentiable map traj -> x0_hat
    _cg2, disp, p2r, _rl2 = orb2.orbit_step(traj, x0h, current_obs,
                                            noise_scale=0.5)
    assert float(p2r.sum()) == float(Bn), (
        "(12) far baseline must put every row in phase 2")
    assert disp.abs().max().item() > 0.0, "(12) phase-2 displacement nonzero"
    orb3 = OrbitCostPlanner(scout_vib, bridge=IdentityBridge())
    orb3.set_current_obs(current_obs)
    traj3 = torch.randn(Bn, H, Ad).requires_grad_(True)
    x0h3 = 2.0 * traj3
    orb3.select_z(x0h3.detach(), current_obs)   # anchor AT the input -> kl ~ 0
    _cg3, disp3, p23, _rl3 = orb3.orbit_step(traj3, x0h3, current_obs,
                                             noise_scale=0.5)
    assert float(p23.sum()) == 0.0 and disp3.abs().max().item() == 0.0, (
        "(12) anchor-equal input must stay entirely in phase 1")

    print(f"[check 10] orbit no-op (lam=sigma=delta=0) == atypical "
          f"bit-identical (max|diff|={diff10:.1e})")
    print(f"[check 11] orbit math: phase mask exact; Newton projection "
          f"identity; damped overshoot formula; tangential orthogonality; "
          f"flat-gradient guard")
    # delta-guard (review P1): delta >= cap clamps to just under cap, so
    # tiny-cap configs cannot silently turn into noise-everything.
    _g = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=0.01,
                          orbit_delta=0.25)
    assert abs(_g.orbit_delta - 0.009999) < 1e-6, (
        f"(12) orbit_delta must clamp under cap; got {_g.orbit_delta}")
    print(f"[check 12] orbit policy path: {orb._orb_calls} calls "
          f"({policy.num_inference_steps}/sample), p2_rows="
          f"{orb.p2_rows}/{orb._orb_rows}, deterministic, far/equal "
          f"baselines behave as designed, delta clamp OK")


def check_orbit_sector(scout_vib, current_obs, cond_data, seed=233):
    """Check 13 (beat-SOE campaign B2, 2026-08-31): sector='det' replaces the
    i.i.d. tangent draw with a per-(scene, try) deterministic, cached
    direction; default 'iid' is untouched.

    (13a) determinism: two orbit_step calls with the same row jobs give
          the same xi_override (cache hit, no redraw);
    (13b) stratification: (init, try) pairs differ -> directions differ;
    (13c) projection: the deterministic xi is projected against the CURRENT
          row normal exactly like the i.i.d. draw (orthogonality on p2 rows);
    (13d) fallback: sector='det' WITHOUT engine jobs falls back to i.i.d.
          (warns once, no crash); invalid sector value raises at __init__.
    """
    torch.manual_seed(seed)
    from scout.guidance.orbit_costs import OrbitCostPlanner
    Bn = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]
    jobs = [(None, 3, j) for j in range(Bn - 1)] + [(None, 7, 4)]

    def _far_planner(**kw):
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                             cap=0.01, orbit_sigma=0.25, **kw)
        p.set_current_obs(current_obs)
        p.select_z(torch.randn(Bn, H, Ad), current_obs)
        p._att._base_mu = [m + 5.0 for m in p._att._base_mu]
        return p

    orb = _far_planner(orbit_sector="det")
    orb.set_row_jobs(jobs)
    xi1 = orb._sector_xi(torch.zeros(Bn, H, Ad))
    assert xi1 is not None and xi1.shape == (Bn, H, Ad), (
        "(13a) sector xi must be stacked with the batch shape")
    xi2 = orb._sector_xi(torch.zeros(Bn, H, Ad))     # cache hit
    assert torch.equal(xi1, xi2), "(13a) cached sector xi must be identical"
    # fresh planner, same jobs + seed -> same directions (cross-call repro)
    orb_b = _far_planner(orbit_sector="det", orbit_sector_seed=42)
    orb_b.set_row_jobs(jobs)
    assert torch.equal(orb_b._sector_xi(torch.zeros(Bn, H, Ad)), xi1), (
        "(13a) sector directions must reproduce across planner instances")
    # (13b) rows 0/1 = same scene different try -> different; 0/last =
    # different scene (7,4) -> different.
    assert not torch.equal(xi1[0], xi1[1]), "(13b) try 0 vs try 1 must differ"
    assert not torch.equal(xi1[0], xi1[-1]), "(13b) scene 3 vs scene 7 differ"
    # (13c) projection: orbit_displacement with xi_override -- the sigma>0
    # minus sigma=0 difference is the det-noise part; it must be orthogonal
    # to g on phase-2 rows, exactly like the i.i.d. draw (fb is along g).
    # PLUS the exact-value assertion: the noise part must equal the analytic
    # projection of xi1 (review P1: orthogonality alone is blind to the
    # override being ignored -- an i.i.d. draw is orthogonal too).
    from scout.guidance.orbit_costs import orbit_displacement
    torch.manual_seed(seed + 1)
    klq = (torch.randn(Bn) * 0.5 + 2.0).abs()      # all rows far above cap
    gq = torch.randn(Bn, H, Ad)
    disp_s0, _, _ = orbit_displacement(klq, gq, 0.01, 0.5, 0.009, sigma=0.0)
    disp_sd, _, _ = orbit_displacement(klq, gq, 0.01, 0.5, 0.009, sigma=0.25,
                                       noise_scale=0.5, xi_override=xi1)
    noise_part = disp_sd - disp_s0
    dotn = (gq * noise_part).flatten(1).sum(dim=1)
    gn = gq.flatten(1).norm(dim=1)
    nn = noise_part.flatten(1).norm(dim=1)
    for i in range(Bn):
        assert abs(float(dotn[i])) < 1e-5 * float(gn[i] * nn[i]), (
            f"(13c) det-mode noise must stay tangent (row {i})")
    gn2 = (gq.flatten(1) ** 2).sum(dim=1).clamp(min=1e-16)
    ghat_q = gq / gn2.sqrt()[:, None, None]
    dot_x = (xi1 * ghat_q).flatten(1).sum(dim=1)
    expected = 0.5 * 0.25 * (xi1 - dot_x[:, None, None] * ghat_q)
    assert torch.allclose(noise_part, expected.to(noise_part.dtype),
                          atol=1e-6), (
        "(13c) det-mode noise must equal the analytic projection of xi1 "
        "-- an ignored override (silent i.i.d. fallback) must fail here")
    # seed actually reaches the generator (review P2): seed 43 != seed 42
    orb_s43 = _far_planner(orbit_sector="det", orbit_sector_seed=43)
    orb_s43.set_row_jobs(jobs)
    xi43 = orb_s43._sector_xi(torch.zeros(Bn, H, Ad))
    assert not torch.equal(xi43, xi1), "(13c) sector seed 43 must differ"
    # and the full planner path still runs end-to-end in det mode
    traj = torch.randn(Bn, H, Ad).requires_grad_(True)
    _cgc, disp, p2r, _rlc = orb.orbit_step(traj, 2.0 * traj, current_obs,
                                           noise_scale=0.5)
    assert float(p2r.sum()) == float(Bn) and disp.abs().max().item() > 0, (
        "(13c) det-mode orbit_step must produce phase-2 displacement")
    # (13d) det WITHOUT jobs -> i.i.d. fallback, warns once, no crash
    orb_nojobs = _far_planner(orbit_sector="det")
    assert orb_nojobs._sector_xi(torch.zeros(Bn, H, Ad)) is None, (
        "(13d) no-jobs sector must fall back to None (i.i.d.)")
    try:
        OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                         orbit_sector="bogus")
        raise AssertionError("(13d) bogus sector must raise at __init__")
    except ValueError:
        pass
    print("[check 13] orbit sector: det directions deterministic per "
          "(scene,try), stratified, tangent-projected; no-jobs fallback + "
          "value guard OK")


def check_orbit_ray(policy, scout_vib, current_obs, cond_data, seed=233):
    """Check 15 (beat-SOE campaign B4, 2026-09-01): climb='ray' rotates the
    phase-1 climb of retries k>=1 onto fixed max-min design unit directions
    with magnitude restoration v = ||g||*sgn(<g,u_k>)*u_k.

    (15a) default 'grad': ray_rotate returns the input untouched;
    (15b) try 0 (gamma_0) row verbatim; try k row: per-row norm preserved
          EXACTLY and direction = +/-u_k with sign = sgn(<g,u_k>);
    (15c) flat rows (zero gradient) stay zero;
    (15d) monotonicity identity <v, g> = ||g||*|<ghat,u>| >= 0 on random
          gradients (the df/dt >= 0 certificate);
    (15e) design deterministic per seed (reproduces across instances),
          different seed differs; no-jobs -> warn-once fallback keeps g;
          invalid climb value raises at __init__;
    (15f) POLICY level: through guided_conditional_sample the _ray_fn hook
          actually rotates -- k>=1 rows differ between grad and ray planners
          (same seed/global RNG; cap=1e9 keeps every row phase 1) while the
          gamma_0 row stays bit-identical (guards the silently-ignored-
          rotation bug class, cf. check 13c's exact-value assertion).
    """
    torch.manual_seed(seed)
    from scout.guidance.orbit_costs import OrbitCostPlanner
    Bn = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]
    jobs = [(None, 3, j) for j in range(Bn - 1)] + [(None, 7, 4)]

    def _planner(**kw):
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                             cap=2.5, orbit_sigma=0.25, **kw)
        p.set_current_obs(current_obs)
        return p

    g = torch.randn(Bn, H, Ad)
    # (15a) default grad mode: untouched (same tensor object)
    orb0 = _planner()
    orb0.set_row_jobs(jobs)
    assert orb0.ray_rotate(g) is g, "(15a) grad mode must return input as-is"
    # (15b) ray mode
    orb = _planner(orbit_climb="ray")
    orb.set_row_jobs(jobs)
    out = orb.ray_rotate(g)
    gn = g.flatten(1).norm(dim=1)
    on = out.flatten(1).norm(dim=1)
    assert torch.equal(out[0], g[0]), "(15b) gamma_0 (try 0) row verbatim"
    dirs = orb._ray_design_dirs(max(int(j[2]) for j in jobs), (H, Ad))
    for r, j in enumerate(jobs):
        k = int(j[2])
        if k < 1:
            continue
        assert torch.allclose(on[r], gn[r], rtol=1e-6), (
            f"(15b) row {r} norm must be preserved exactly")
        u = dirs[k - 1]
        cos = float((g[r].flatten() @ u.flatten()) / gn[r])
        expect = gn[r] * (1.0 if cos >= 0.0 else -1.0) * u
        assert torch.allclose(out[r], expect, atol=1e-5), (
            f"(15b) row {r} must be the magnitude-restored design direction")
        if r != 0:
            cross = float(out[r].flatten() @ dirs[0].flatten()
                          / (on[r] * dirs[0].flatten().norm()))
            assert abs(cross) < 0.9 or k == 1, (
                "(15b) design directions should be near-orthogonal")
    # (15c) flat rows stay zero
    gflat = g.clone(); gflat[-1] = 0.0
    outf = orb.ray_rotate(gflat)
    assert float(outf[-1].abs().max()) == 0.0, "(15c) flat row stays zero"
    # (15d) monotonicity identity: <v, g> >= 0 on every rotated row
    dots = (out[:-1] * g[:-1]).flatten(1).sum(dim=1)
    assert bool((dots >= -1e-6).all()), "(15d) <v,g> must be >= 0 everywhere"
    # (15e) determinism / seeds / fallback / validation
    orb_b = _planner(orbit_climb="ray", orbit_ray_seed=42)
    orb_b.set_row_jobs(jobs)
    assert torch.equal(
        orb_b.ray_rotate(g), out), "(15e) ray must reproduce across instances"
    orb_s43 = _planner(orbit_climb="ray", orbit_ray_seed=43)
    orb_s43.set_row_jobs(jobs)
    assert not torch.equal(orb_s43.ray_rotate(g), out), (
        "(15e) ray seed 43 must differ")
    orb_nj = _planner(orbit_climb="ray")
    assert orb_nj.ray_rotate(g) is g, "(15e) no-jobs fallback keeps g verbatim"
    assert orb_nj.ray_rotate(g) is g, "(15e) fallback warns once, no crash"
    try:
        OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                         orbit_climb="bogus")
        raise AssertionError("(15e) bogus climb must raise at __init__")
    except ValueError:
        pass
    # (15f) policy level: the hook bites inside guided_conditional_sample
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    gst = policy.noise_scheduler.config.num_train_timesteps
    tp = {}
    for name, pl in (
            ("grad", OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                                      cap=1e9, orbit_lam=0.0,
                                      orbit_delta=0.0, orbit_sigma=0.0)),
            ("ray", OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                                     cap=1e9, orbit_lam=0.0,
                                     orbit_delta=0.0, orbit_sigma=0.0,
                                     orbit_climb="ray"))):
        policy.initialize_scout_planner(planner=pl,
                                        guidance_start_timestep=gst,
                                        guidance_scale=1.0)
        pl.set_row_jobs(jobs)
        tp[name] = _run_sample(policy, cond_data, cond_mask, None,
                               current_obs, z=None, seed=seed,
                               classifier_guidance=True, guidance_scale=1.0)
    d15 = (tp["grad"] - tp["ray"]).abs()
    assert float(d15[0].max()) == 0.0, (
        "(15f) gamma_0 row must stay bit-identical through the policy path")
    assert float(d15[1:].max()) > 0.0, (
        "(15f) k>=1 rows must differ -- a silently-ignored ray hook fails here")
    assert int(pl._ray_cnt) > 0, "(15f) ray telemetry must count rotated rows"
    print("[check 15] orbit ray: gamma_0 verbatim, try-k rows = norm-preserving "
          "+/-u_k (design det./seeded), flat rows zero, <v,g>>=0 identity, "
          "policy hook bites OK")


def check_orbit_anneal(scout_vib, current_obs, cond_data, seed=233):
    """Check 14 (beat-SOE campaign B3, 2026-08-31): orbit_noise_anneal=p
    raises the incoming sqrt(1-abar_t) noise scale to the p-th power.

    (14a) p=1.0 passes the scalar through untouched (bit-identical orbit
          output -- no pow on the hot path);
    (14b) p=2 squares the scale exactly: displacement's noise part scales
          by scale^2 vs p=1 with the same RNG stream;
    (14c) p<=0 raises at __init__.
    """
    torch.manual_seed(seed)
    from scout.guidance.orbit_costs import OrbitCostPlanner
    Bn = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]

    def far(p_val, **kw):
        torch.manual_seed(seed + 3)          # identical anchor across arms
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(),
                             cap=0.01, orbit_sigma=0.25,
                             orbit_noise_anneal=p_val, **kw)
        p.set_current_obs(current_obs)
        p.select_z(torch.randn(Bn, H, Ad), current_obs)
        p._att._base_mu = [m + 5.0 for m in p._att._base_mu]
        return p

    outs = {}
    for tag, pval in (("p1", 1.0), ("p2", 2.0)):
        pl = far(pval)
        torch.manual_seed(seed + 7)          # same stream for both arms
        traj = torch.randn(Bn, H, Ad).requires_grad_(True)
        _cg, disp, p2r, _rl = pl.orbit_step(traj, 2.0 * traj, current_obs,
                                            noise_scale=0.6)
        outs[tag] = (disp, p2r)
    d1, p2r = outs["p1"]; d2, _ = outs["p2"]
    assert float(p2r.sum()) == float(Bn), "(14) far baseline must be all p2"
    assert d1.abs().max().item() > 0, "(14) displacement must be nonzero"
    # (14a) bit-identity at p=1: rerun with a fresh planner, same stream
    pl_b = far(1.0)
    torch.manual_seed(seed + 7)
    traj_b = torch.randn(Bn, H, Ad).requires_grad_(True)
    _cgb, disp_b, _p2b, _rlb = pl_b.orbit_step(traj_b, 2.0 * traj_b,
                                               current_obs, noise_scale=0.6)
    assert torch.equal(d1, disp_b), "(14a) p=1 must be bit-identical"
    # (14b) p=2: fb identical in both arms (same kl/g, same RNG stream), so
    # d2 - fb = (0.36/0.6) * (d1 - fb) -> d2 == 0.6*d1 + 0.4*fb; fb is
    # isolated by a third run at noise_scale=0 (sigma term multiplies to 0).
    pl0 = far(1.0)
    torch.manual_seed(seed + 7)
    traj0 = torch.randn(Bn, H, Ad).requires_grad_(True)
    _cg0, disp0, _p20, _rl0 = pl0.orbit_step(traj0, 2.0 * traj0, current_obs,
                                             noise_scale=0.0)     # fb only (sigma=0)
    expect_d2 = 0.6 * d1 + 0.4 * disp0
    assert torch.allclose(d2, expect_d2, atol=1e-6), (
        "(14b) p=2 must square the scale exactly (d2 == s^2/s*d1+(1-s^2/s)fb)")
    # non-integer p: 0.5 -> scale^0.5 = sqrt(0.6); same algebra via fb
    pl_h = far(0.5)
    torch.manual_seed(seed + 7)
    traj_h = torch.randn(Bn, H, Ad).requires_grad_(True)
    _cgh, dh, _p2h, _rlh = pl_h.orbit_step(traj_h, 2.0 * traj_h, current_obs,
                                           noise_scale=0.6)
    r_h = 0.6 ** (0.5 - 1.0)     # noise ratio scale_h/scale_1 = s^(p-1)
    assert torch.allclose(dh, r_h * d1 + (1.0 - r_h) * disp0, atol=1e-6), (
        "(14b) non-integer p must follow scale**p exactly")
    # (14c) invalid p raises
    try:
        far(0.0)
        raise AssertionError("(14c) p=0 must raise at __init__")
    except ValueError:
        pass
    print("[check 14] orbit noise anneal: p=1 bit-identical; p=2 squares "
          "the sqrt(1-abar) scale exactly; p<=0 guard OK")


def check_orbit_merged(policy, scout_vib, current_obs, cond_data, seed=233):
    """Check 16 (perf 2026-09-01, 方案一+二 user-approved): the merged
    single-backward ``orbit_step`` is BIT-IDENTICAL to the pre-merge
    two-backward algorithm, and the vectorized atypical row core matches the
    historical per-row loop.

    (16a) planner level: on identical inputs (same RNG stream for the xi
          draw) orbit_step's (cond_grad, disp, p2, row_losses) equal the
          legacy path's exactly (torch.equal) -- cond_grad from a separate
          capped compute_loss backward (retain_graph) + a second uncapped
          row-loop forward and backward -- on a fixture that straddles the
          cap boundary AND both phases (baseline-at-query rows sit at KL=0;
          nudged rows sit far above kappa);
    (16b) policy level: same-seed guided samples through the real
          guided_conditional_sample are bit-identical between a merged
          planner and a legacy-subclass planner (guards RNG ordering and
          the injection-line rewiring);
    (16c) telemetry parity: the policy's mean/max injected-force
          accumulators advance identically on both arms (dose-calibration
          continuity);
    (16d) vectorized atypical rows: _encode_and_row_losses equals the
          per-row loop reference exactly (values AND gradients), including
          missing-baseline rows (graph-connected zeros) and a mixed list.
    """
    from scout.guidance.orbit_costs import (OrbitCostPlanner,
                                            orbit_displacement)
    from scout.guidance.entropy_costs import AtypicalCostPlanner, _enc_forward

    class _LegacyOrbit(OrbitCostPlanner):
        """Pre-merge reference algorithm, verbatim structure: capped
        compute_loss backward (retain_graph) + second uncapped row-loop
        forward and backward. Telemetry counters are NOT incremented (the
        comparison reads returned values and the policy-side accumulators
        only)."""

        def orbit_step(self, trajectory, x0_hat, current_obs=None,
                       noise_scale=1.0):
            if float(self.orbit_noise_anneal) != 1.0:
                noise_scale = float(noise_scale) ** float(self.orbit_noise_anneal)
            loss = self._att.compute_loss(x0_hat, current_obs,
                                          reduction="sum")
            cond_grad = -torch.autograd.grad(loss, trajectory,
                                             retain_graph=True)[0]
            s_bar_t = self._att._resolve_s_bar_t(current_obs)
            a = _enc_forward(self, x0_hat)
            mu, logvar = self.scout_vib.vib_enc(s_bar_t.detach(), a)
            rows = []
            for i in range(mu.shape[0]):
                if (i >= len(self._att._base_mu)
                        or self._att._base_mu[i] is None):
                    rows.append(x0_hat[i].sum() * 0.0)
                    continue
                m0, lv0 = self._att._base_mu[i], self._att._base_lv[i]
                var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
                kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                            + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
                rows.append(kl)
            kl = torch.stack(rows)
            g = torch.autograd.grad(kl.sum(), trajectory)[0]
            disp, p2, _ = orbit_displacement(
                kl, g, kappa=self._att.cap, lam=self.orbit_lam,
                delta=self.orbit_delta, sigma=self.orbit_sigma,
                noise_scale=noise_scale,
                xi_override=(self._sector_xi(g) if self.orbit_sigma > 0.0
                             else None))
            row_losses = -torch.clamp(kl.detach(), max=float(self._att.cap))
            return cond_grad, disp, p2, row_losses

    B = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]

    # ---- (16a) planner-level equivalence --------------------------------- #
    # cap=2.5 / delta=0.25 (standard): odd-row baselines captured AT the
    # query -> KL = 0 exactly (phase 1, uncapped); even rows nudged +3.0 ->
    # KL far above kappa (capped climb + phase 2). The fixture exercises the
    # cap mask boundary region and both phases; the [kappa-delta, kappa)
    # band itself is covered at the pure-math level by check 11 (both
    # algorithms compute band rows with identical formulas).
    merged = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5,
                              orbit_lam=0.5, orbit_delta=0.25,
                              orbit_sigma=0.25)
    legacy = _LegacyOrbit(scout_vib, bridge=IdentityBridge(), cap=2.5,
                          orbit_lam=0.5, orbit_delta=0.25, orbit_sigma=0.25)
    for p in (merged, legacy):
        p.set_current_obs(current_obs)
    torch.manual_seed(seed + 12)
    traj0 = torch.randn(B, H, Ad)
    merged.select_z(2.0 * traj0, current_obs)   # anchor AT the query
    legacy._att._base_mu = [m.clone() for m in merged._att._base_mu]
    legacy._att._base_lv = [v.clone() for v in merged._att._base_lv]
    with torch.no_grad():
        for i in range(0, B, 2):                # even rows far above kappa
            merged._att._base_mu[i] = merged._att._base_mu[i] + 3.0
            legacy._att._base_mu[i] = legacy._att._base_mu[i] + 3.0
    outs = {}
    for name, p in (("merged", merged), ("legacy", legacy)):
        torch.manual_seed(seed + 12)            # redraws traj0 bit-exactly;
        traj = torch.randn(B, H, Ad).requires_grad_(True)  # xi stream shared
        outs[name] = p.orbit_step(traj, 2.0 * traj, current_obs,
                                  noise_scale=0.6)
    cg_m, disp_m, p2_m, rl_m = outs["merged"]
    cg_l, disp_l, p2_l, rl_l = outs["legacy"]
    assert torch.equal(cg_m, cg_l), (
        f"(16a) merged cond_grad != legacy (max|diff|="
        f"{(cg_m - cg_l).abs().max().item():.3e})")
    assert torch.equal(disp_m, disp_l), "(16a) merged disp != legacy"
    assert torch.equal(p2_m, p2_l), "(16a) merged p2 != legacy"
    assert torch.equal(rl_m, rl_l), "(16a) merged row_losses != legacy"
    climb_zero = (cg_m.flatten(1).abs().sum(dim=1) == 0)
    assert int(climb_zero.sum()) > 0 and float(p2_m.sum()) > 0 \
        and float(p2_m.sum()) < float(B), (
        f"(16a) fixture must mix phases and capped rows; climb-zero rows="
        f"{int(climb_zero.sum())}, p2={float(p2_m.sum())}/{B}")

    # ---- (16b/16c) policy-level bit-identity + telemetry parity ---------- #
    gst = policy.noise_scheduler.config.num_train_timesteps
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    tp, dacc = {}, {}
    for name, Pl in (("merged", OrbitCostPlanner), ("legacy", _LegacyOrbit)):
        pl = Pl(scout_vib, bridge=IdentityBridge(), cap=2.5,
                orbit_lam=0.5, orbit_delta=0.25, orbit_sigma=0.25)
        policy.initialize_scout_planner(planner=pl,
                                        guidance_start_timestep=gst,
                                        guidance_scale=1.0)
        g0 = (policy._g_acc.clone()
              if getattr(policy, "_g_acc", None) is not None else None)
        n0 = getattr(policy, "_g_n", 0)
        tp[name] = _run_sample(policy, cond_data, cond_mask, None,
                               current_obs, z=None, seed=seed,
                               classifier_guidance=True, guidance_scale=1.0)
        dacc[name] = (None if (policy._g_acc is None or g0 is None)
                      else policy._g_acc - g0, policy._g_n - n0)
    assert torch.equal(tp["merged"], tp["legacy"]), (
        "(16b) merged vs legacy must be bit-identical through the policy "
        "path (RNG ordering / injection rewiring)")
    dm, dl = dacc["merged"][0], dacc["legacy"][0]
    assert (dm is None and dl is None) or torch.equal(dm, dl), (
        "(16c) injected-force telemetry must advance identically")
    assert dacc["merged"][1] == dacc["legacy"][1], "(16c) step counts differ"

    # orbit arm's cost-curve branch (review P2-5b: previously untested --
    # a typo there would pass silently): curve length == guided steps, all
    # finite, values match the historical atypical-equivalent scale.
    pl = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5,
                          orbit_lam=0.5, orbit_delta=0.25, orbit_sigma=0.25)
    policy.initialize_scout_planner(planner=pl,
                                    guidance_start_timestep=gst,
                                    guidance_scale=1.0)
    _, curve = _run_sample(policy, cond_data, cond_mask, None,
                           current_obs, z=None, seed=seed,
                           classifier_guidance=True, guidance_scale=1.0,
                           return_cost_curve=True)
    assert len(curve) == policy.num_inference_steps, (
        f"(16b) orbit cost curve must have one entry per guided step; "
        f"got {len(curve)}")
    assert all(c == c and abs(c) != float("inf") for c in curve), (
        "(16b) orbit cost curve must be finite")
    assert max(abs(c) for c in curve) <= pl._att.cap + 1e-6, (
        "(16b) orbit cost curve values must live in [-cap, 0] "
        f"(atypical-equivalent scale); got {curve[:3]}")

    # ---- (16d) vectorized atypical rows vs the loop reference ------------- #
    att = AtypicalCostPlanner(scout_vib, bridge=IdentityBridge(), cap=0.05)
    att.set_current_obs(current_obs)
    torch.manual_seed(seed + 13)
    x0 = torch.randn(B, H, Ad)
    att.select_z(x0, current_obs)

    def _loop_reference(planner, x0_hat):
        s_bar_t = planner._resolve_s_bar_t(current_obs)
        a = _enc_forward(planner, x0_hat)
        mu, logvar = planner.scout_vib.vib_enc(s_bar_t.detach(), a)
        rows = []
        for i in range(mu.shape[0]):
            if (i >= len(planner._base_mu) or planner._base_mu[i] is None):
                rows.append(x0_hat[i].sum() * 0.0)
                continue
            m0, lv0 = planner._base_mu[i], planner._base_lv[i]
            var, var0 = torch.exp(logvar[i]), torch.exp(lv0)
            kl = 0.5 * (((mu[i] - m0) ** 2 / var0)
                        + (var / var0) - 1.0 - (logvar[i] - lv0)).sum()
            rows.append(-torch.clamp(kl, max=planner.cap))
        return torch.stack(rows)

    _, rows_v = att._encode_and_row_losses(x0, current_obs)
    assert torch.equal(torch.stack(rows_v), _loop_reference(att, x0)), (
        "(16d) vectorized atypical rows must equal the loop reference")
    # gradients through compute_loss vs the reference graph
    xa = x0.clone().requires_grad_(True)
    ga, = torch.autograd.grad(att.compute_loss(xa, current_obs,
                                               reduction="sum"), xa)
    xb = x0.clone().requires_grad_(True)
    gb, = torch.autograd.grad(_loop_reference(att, xb).sum(), xb)
    assert torch.equal(ga, gb), (
        f"(16d) gradients must equal the loop reference (max|diff|="
        f"{(ga - gb).abs().max().item():.3e})")
    # missing baselines -> graph-connected zeros (empty + mixed)
    att2 = AtypicalCostPlanner(scout_vib, bridge=IdentityBridge(), cap=0.05)
    att2.set_current_obs(current_obs)
    _, rows_e = att2._encode_and_row_losses(x0, current_obs)   # no select_z
    assert torch.equal(torch.stack(rows_e), torch.zeros(B)), (
        "(16d) empty-baseline rows must be exact zeros")
    att3 = AtypicalCostPlanner(scout_vib, bridge=IdentityBridge(), cap=0.05)
    att3.set_current_obs(current_obs)
    att3.select_z(x0, current_obs)
    att3._base_mu[1] = None
    _, rows3 = att3._encode_and_row_losses(x0, current_obs)
    stacked3 = torch.stack(rows3)
    assert torch.equal(stacked3, _loop_reference(att3, x0)) \
        and float(stacked3[1]) == 0.0, (
        "(16d) mixed-baseline rows must match the reference (None -> 0)")

    print(f"[check 16] merged orbit_step == legacy two-backward "
          f"bit-identical (planner + policy level, telemetry parity); "
          f"atypical rows vectorized == loop reference (values + grads)")


def check_orbit_eta_dimless(policy, scout_vib, current_obs, cond_data,
                            seed=233):
    """Check 17 (2026-09-02, orbit-hparam-dev): eta-dimless climb
    normalization.

    (17a) OFF (default) is deterministic on a fixture that mixes at-anchor,
          mid-band and above-cap rows (16a already pins OFF to the legacy
          two-backward reference);
    (17b) ON divides cond_grad by the LIVE-CLIMB MEAN per-row ||grad||
          (rows with kl < cap-delta and norm > 1e-4), recomputed
          INDEPENDENTLY in this check via _enc_forward/_kl_rows (a
          self-referential _last_g_med comparison would pass for ANY
          statistic definition -- review round 2 demonstrated the all-rows
          variant passing its own divisor); disp / p2 / row_losses are
          UNTOUCHED (phase 2 has no eta dependence), and at-anchor roundoff
          rows stay sub-dose after normalization (P0-1 regression);
    (17c) policy-level: normalization BITES at equal scale (trajectory
          differs, finite) -- full bit-identity is not assertable because
          g_med is a per-step online statistic; the planner-level identity
          (17b) + check 16b's RNG-path guarantee cover the wiring.
    """
    from scout.guidance.orbit_costs import OrbitCostPlanner

    B = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]

    torch.manual_seed(seed + 12)
    traj0 = torch.randn(B, H, Ad)

    def mk(norm_on):
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5,
                             orbit_lam=0.5, orbit_delta=0.25,
                             orbit_sigma=0.25, orbit_grad_norm=norm_on)
        p.set_current_obs(current_obs)
        p.select_z(2.0 * traj0, current_obs)      # B baseline anchors
        return p

    def armed(p):
        with torch.no_grad():
            for i in range(0, B, 2):              # even rows far above kappa
                p._att._base_mu[i] = p._att._base_mu[i] + 3.0
            if B > 1:                             # row 1: mid-band (0<KL<cap)
                p._att._base_mu[1] = p._att._base_mu[1] + 0.4
        return p

    outs = {}
    for name, on in (("off1", False), ("off2", False), ("on", True)):
        p = armed(mk(on))
        torch.manual_seed(seed + 12)
        traj = torch.randn(B, H, Ad).requires_grad_(True)
        outs[name] = p.orbit_step(traj, 2.0 * traj, current_obs,
                                  noise_scale=0.6)
        outs[name + "_planner"] = p
    cg_off1, disp_off1, p2_off1, rl_off1 = outs["off1"]
    cg_off2, _, _, _ = outs["off2"]
    cg_on, disp_on, p2_on, rl_on = outs["on"]
    # (17a) OFF determinism == pre-change semantics (16a already pins OFF to
    # the legacy two-backward reference; here two fresh OFF planners agree).
    assert torch.equal(cg_off1, cg_off2), "(17a) OFF must be deterministic"
    # fixture sanity (review P1-3): row 1 must be a LIVE climb row -- mid-band
    # KL with an O(1) gradient, not at-anchor roundoff -- otherwise the
    # division below is verified on numerical noise only.
    assert cg_off1[1].abs().max().item() > 1e-3, (
        "(17a) fixture row 1 must carry a real mid-band climb gradient")
    # (17b) ON == OFF / g_med where g_med is the LIVE-CLIMB MEAN row norm,
    # recomputed INDEPENDENTLY here (self-referencing the planner's
    # _last_g_med would pass for any statistic -- the review's WrongStat
    # probe passed its own divisor). Rebuild (kl, g) on the identical
    # fixture input through the same encoder path.
    from scout.guidance.entropy_costs import _enc_forward, _kl_rows
    p_on = outs["on_planner"]
    torch.manual_seed(seed + 12)
    traj_r = torch.randn(B, H, Ad).requires_grad_(True)
    s_bar = p_on._att._resolve_s_bar_t(current_obs)
    a_r = _enc_forward(p_on, 2.0 * traj_r)
    mu_r, lv_r = p_on.scout_vib.vib_enc(s_bar.detach(), a_r)
    kl_r = _kl_rows(mu_r, lv_r, p_on._att._base_mu, p_on._att._base_lv,
                    2.0 * traj_r)
    g_r = torch.autograd.grad(kl_r.sum(), traj_r)[0]
    norms_r = g_r.detach().flatten(1).norm(dim=1)
    norms_r = torch.nan_to_num(norms_r, nan=0.0, posinf=0.0, neginf=0.0)
    live_r = ((kl_r.detach() < float(p_on._att.cap) - p_on.orbit_delta)
              & (norms_r > 1e-4)).to(norms_r.dtype)
    g_med = ((norms_r * live_r).sum()
             / live_r.sum().clamp(min=1.0)).clamp(min=1e-4)
    assert torch.allclose(outs["on_planner"]._last_g_med, g_med, atol=1e-6), (
        "(17b) planner's stored divisor must equal the independent "
        "live-climb-mean recomputation")
    assert torch.allclose(cg_on, cg_off1 / g_med, atol=1e-6), (
        "(17b) ON must divide cond_grad by the live-climb mean row ||grad||")
    # roundoff amplification guard (review P0-1): a batch of PURE at-anchor
    # rows (KL==0 exactly, roundoff grads ~1e-8) must stay near-zero after
    # normalization -- the 1e-4 floor prevents the divisor from amplifying
    # noise into a full-dose kick.
    p_anchor = mk(False)                          # NO arming: every row at-anchor
    p_dim = mk(True)
    torch.manual_seed(seed + 12)
    traj_a = torch.randn(B, H, Ad).requires_grad_(True)
    cga, _, _, _ = p_anchor.orbit_step(traj_a, 2.0 * traj_a, current_obs,
                                       noise_scale=0.6)
    torch.manual_seed(seed + 12)
    traj_d = torch.randn(B, H, Ad).requires_grad_(True)
    cgd, _, _, _ = p_dim.orbit_step(traj_d, 2.0 * traj_d, current_obs,
                                    noise_scale=0.6)
    assert cgd.abs().max().item() < 1e-3, (
        f"(17b) at-anchor roundoff amplified after normalization "
        f"(max|cg|={cgd.abs().max().item():.3e} vs legacy "
        f"{cga.abs().max().item():.3e})")
    assert torch.equal(disp_on, disp_off1), \
        "(17b) phase-2 disp must not depend on the normalization"
    assert torch.equal(p2_on, p2_off1) and torch.equal(rl_on, rl_off1), \
        "(17b) p2/row_losses must not depend on the normalization"

    # (17c) policy-level: normalization BITES (trajectory differs from the
    # OFF arm at equal scale) and the dose telemetry stays finite/positive.
    # Full-trajectory bit-identity at eta_tilde = eta*g_med is NOT asserted:
    # g_med is a PER-STEP live-climb mean (by design -- the online-adaptive
    # semantics), so a fixed scale cannot reproduce the OFF arm step-by-step
    # unless every step's divisor is identical; the planner-level identity
    # (17b) plus check 16b's policy RNG-path guarantee cover the wiring.
    gst = policy.noise_scheduler.config.num_train_timesteps
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    tp = {}
    for name, on, sc in (("off", False, 0.7), ("on", True, 0.7)):
        pl = armed(mk(on))
        policy.initialize_scout_planner(planner=pl,
                                        guidance_start_timestep=gst,
                                        guidance_scale=sc)
        tp[name] = _run_sample(policy, cond_data, cond_mask, None,
                               current_obs, z=None, seed=seed,
                               classifier_guidance=True, guidance_scale=sc)
    assert not torch.equal(tp["on"], tp["off"]), (
        "(17c) normalization must change the trajectory at equal scale")
    assert torch.isfinite(tp["on"]).all(), "(17c) ON arm produced NaN/Inf"
    print(f"[check 17] eta-dimless: OFF deterministic; ON == OFF/g_med "
          f"(disp/p2 untouched, g_med={float(g_med):.4g}); "
          f"policy-level bite + finite telemetry OK")


def check_orbit_sigma_schedule(scout_vib, current_obs, cond_data, seed=233):
    """Check 18 (2026-09-02, orbit-hparam-dev): round-dependent sigma
    ceiling  sigma_eff = sigma * decay**(round-1).

    (18a) round=1 / decay=1.0 (defaults) is BIT-IDENTICAL to the legacy
          planner on a mixed-phase fixture (same RNG stream);
    (18b) round=2 / decay=0.5 halves ONLY the tangential-noise component
          exactly: disp(r2) == 0.5*disp(legacy) + 0.5*fb, where fb is
          isolated by a sigma=0 run (the check-14b algebra);
    (18c) a large round drives sigma_eff to numerical zero -> NO randn is
          drawn (disp equals the sigma=0 arm bit-for-bit) -- the sigma>0
          guard in orbit_displacement holds;
    (18d) invalid arguments raise at __init__ (round < 1, decay outside
          (0,1]).
    """
    from scout.guidance.orbit_costs import OrbitCostPlanner

    B = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]

    torch.manual_seed(seed + 12)
    traj0 = torch.randn(B, H, Ad)

    def mk(sigma=0.25, **kw):
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5,
                             orbit_lam=0.5, orbit_delta=0.25,
                             orbit_sigma=sigma, **kw)
        p.set_current_obs(current_obs)
        p.select_z(2.0 * traj0, current_obs)
        with torch.no_grad():
            for i in range(0, B, 2):
                p._att._base_mu[i] = p._att._base_mu[i] + 3.0
            if B > 1:
                p._att._base_mu[1] = p._att._base_mu[1] + 0.4
        return p

    def run(p, ns=0.6):
        torch.manual_seed(seed + 18)
        traj = torch.randn(B, H, Ad).requires_grad_(True)
        return p.orbit_step(traj, 2.0 * traj, current_obs, noise_scale=ns)

    # (18a) defaults bit-identical to a planner constructed without the new
    # kwargs at all
    a1 = run(mk())
    a2 = run(mk(orbit_round=1, orbit_sigma_decay=1.0))
    assert torch.equal(a1[0], a2[0]) and torch.equal(a1[1], a2[1]), (
        "(18a) round=1/decay=1 must be bit-identical to the legacy planner")
    # (18b) round=2/decay=0.5 -> noise halved exactly (check-14b algebra:
    # fb isolated with sigma=0, which shares the RNG stream trivially).
    r2 = run(mk(orbit_round=2, orbit_sigma_decay=0.5))
    fb_only = run(mk(sigma=0.0))
    assert torch.allclose(r2[1], 0.5 * a1[1] + 0.5 * fb_only[1], atol=1e-6), (
        "(18b) round=2/decay=0.5 must scale ONLY the noise component by 0.5")
    # (18c) round large -> sigma_eff ~ 0 -> no randn drawn
    p_zero = mk(sigma=0.0)
    p_big = mk(orbit_round=60, orbit_sigma_decay=0.5)   # 0.25*0.5**59 ~ 0
    assert p_big.orbit_sigma_eff == 0.0, (
        f"(18c) numerical-zero sigma_eff must snap to EXACT 0.0, got "
        f"{p_big.orbit_sigma_eff!r} -- otherwise the randn draw still fires "
        f"and shifts the trajectory RNG stream")
    out_zero = run(p_zero)
    out_big = run(p_big)
    assert torch.equal(out_zero[1], out_big[1]), (
        "(18c) numerical-zero sigma_eff must not consume randn (disp equal "
        "to the sigma=0 arm)")
    # (18d) guards
    for bad in (dict(orbit_round=0), dict(orbit_sigma_decay=0.0),
                dict(orbit_sigma_decay=1.5)):
        try:
            mk(**bad)
            raise AssertionError(f"(18d) {bad} must raise at __init__")
        except ValueError:
            pass
    print("[check 18] sigma round-schedule: defaults bit-identical; "
          "round=2/decay=0.5 halves only the noise; numerical-zero draws no "
          "randn; guards OK")


def _probe_row1_offset(scout_vib, current_obs, cond_data, target_kl,
                       seed=233, kappa=2.5):
    """Scan the row-1 anchor offset so its KL lands in-band (~target_kl)
    -- a vacuous in-band assertion was shipped once (review P1): without
    this, no fixture row sat in [kappa-delta/2 band] and the approximation
    check silently skipped."""
    from scout.guidance.entropy_costs import _enc_forward, _kl_rows
    from scout.guidance.orbit_costs import OrbitCostPlanner
    B = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]
    torch.manual_seed(seed + 12)
    traj0 = torch.randn(B, H, Ad)
    for off in (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.5, 0.6):
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=kappa)
        p.set_current_obs(current_obs)
        p.select_z(2.0 * traj0, current_obs)
        with torch.no_grad():
            p._att._base_mu[1] = p._att._base_mu[1] + off
        torch.manual_seed(seed + 19)
        t = torch.randn(B, H, Ad)
        s_bar = p._att._resolve_s_bar_t(current_obs)
        a = _enc_forward(p, 2.0 * t)
        mu, lv = p.scout_vib.vib_enc(s_bar.detach(), a)
        kl = _kl_rows(mu, lv, p._att._base_mu, p._att._base_lv, 2.0 * t)
        k1 = float(kl[1].detach())
        if abs(k1 - target_kl) <= 0.12:
            return off
    raise AssertionError("no row-1 offset lands in-band; fixture drift")


def check_orbit_fb_clamp(scout_vib, current_obs, cond_data, seed=233):
    """Check 19 (2026-09-02, orbit-hparam-dev): fb soft-clamp (user option C)
    -- Newton residual (kl-kappa) -> delta*tanh((kl-kappa)/delta), tangential
    noise masked to the band [kappa-delta, kappa+delta].

    (19a) fb_clamp='none' (default) is bit-identical to a planner constructed
          without the kwarg (same RNG stream, mixed-phase fixture);
    (19b) soft fb equals the analytic formula per row (independent
          recomputation), and in-band rows (|kl-kappa| <= delta/2) match the
          none-arm's fb within the tanh linearity bound (rel err <=
          (x/delta)^2/3 + eps ~ 9%);
    (19c) far-off-shell rows (|kl-kappa| >= 8*delta) have fb norm ==
          lam*delta/||g|| exactly (tanh saturated to 1 within float);
    (19d) noise band: off-band rows (kl > kappa+delta) carry ZERO noise in
          the soft arm while band rows keep it; the RNG stream is IDENTICAL
          across modes (randn is drawn then zeroed -- compare torch RNG
          state after one orbit_step in each mode);
    (19e) invalid fb_clamp raises at __init__ / orbit_displacement.
    """
    from scout.guidance.orbit_costs import (OrbitCostPlanner,
                                            orbit_displacement)
    from scout.guidance.entropy_costs import _enc_forward, _kl_rows

    B = current_obs["proprio"].shape[0]
    H, Ad = cond_data.shape[1], cond_data.shape[2]
    KAP, DEL, LAM = 2.5, 0.25, 0.5
    _ROW1_OFF = _probe_row1_offset(scout_vib, current_obs, cond_data,
                                   KAP - DEL / 2, seed)

    torch.manual_seed(seed + 12)
    traj0 = torch.randn(B, H, Ad)

    def mk(_sigma=0.25, **kw):
        p = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=KAP,
                             orbit_lam=LAM, orbit_delta=DEL,
                             orbit_sigma=_sigma, **kw)
        p.set_current_obs(current_obs)
        p.select_z(2.0 * traj0, current_obs)
        with torch.no_grad():
            for i in range(0, B, 2):            # even rows far above kappa
                p._att._base_mu[i] = p._att._base_mu[i] + 3.0
            if B > 2:                           # row 2: just above the band
                p._att._base_mu[2] = p._att._base_mu[2] + 3.0 + DEL * 8
            if B > 1:                           # row 1: IN-BAND row
                p._att._base_mu[1] = p._att._base_mu[1] + _ROW1_OFF
        return p

    def run(p, ns=0.6):
        torch.manual_seed(seed + 19)
        traj = torch.randn(B, H, Ad).requires_grad_(True)
        return p.orbit_step(traj, 2.0 * traj, current_obs, noise_scale=ns)

    # (19a) none == planner constructed without the kwarg
    a_none_default = run(mk())
    a_none_kw = run(mk(orbit_fb_clamp="none"))
    assert torch.equal(a_none_default[1], a_none_kw[1]), (
        "(19a) fb_clamp='none' must be bit-identical to the default")

    # soft arm + independent recomputation of (kl, g) on the same fixture
    p_soft = mk(orbit_fb_clamp="soft")
    out_soft = run(p_soft)
    disp_soft, p2_soft = out_soft[1], out_soft[2]
    torch.manual_seed(seed + 19)
    traj_r = torch.randn(B, H, Ad).requires_grad_(True)
    s_bar = p_soft._att._resolve_s_bar_t(current_obs)
    a_r = _enc_forward(p_soft, 2.0 * traj_r)
    mu_r, lv_r = p_soft.scout_vib.vib_enc(s_bar.detach(), a_r)
    kl_r = _kl_rows(mu_r, lv_r, p_soft._att._base_mu, p_soft._att._base_lv,
                    2.0 * traj_r)
    g_r = torch.autograd.grad(kl_r.sum(), traj_r)[0]
    kl_d = kl_r.detach()
    gn2 = g_r.detach().flatten(1).norm(dim=1) ** 2
    safe = gn2.clamp(min=1e-16)
    nonflat = gn2 >= 1e-16

    # (19b) analytic formula, row-wise: fb = lam*delta*tanh((kl-kappa)/delta)
    #       / ||g||^2 * g, masked by p2
    ref_coeff = torch.where(nonflat, -LAM * DEL * torch.tanh(
        (kl_d - KAP) / DEL) / safe, torch.zeros_like(kl_d))
    ref_fb = ref_coeff[:, None, None] * g_r.detach()
    ref_disp_fb = ref_fb * p2_soft.detach()[:, None, None]
    # isolate fb from the soft disp via a sigma=0 soft run (shares RNG --
    # sigma=0 draws nothing)
    p_soft0 = mk(orbit_fb_clamp="soft", _sigma=0.0)
    out_soft0 = run(p_soft0)
    fb_isolated = out_soft0[1]                       # sigma=0 -> fb only
    assert torch.allclose(fb_isolated, ref_disp_fb, atol=1e-6), (
        "(19b) soft fb must equal the analytic tanh formula per row")
    # in-band approximation vs the none arm
    inband = (kl_d <= KAP) & (kl_d >= KAP - DEL / 2) & nonflat
    assert bool(inband.any()), (
        "(19b) fixture must contain an in-band row (anti-vacuity, review P1)")
    p_none0 = mk(_sigma=0.0)
    out_none0 = run(p_none0)
    fb_none = out_none0[1]
    rel = ((fb_isolated[inband] - fb_none[inband])
           .norm() / fb_none[inband].norm().clamp(min=1e-12))
    assert float(rel) <= 0.10, (
        f"(19b) in-band soft-vs-none rel err {float(rel):.3f} > 10% "
        f"(tanh linearity broken)")

    # (19c) far-off-shell rows: fb norm == lam*delta/||g||
    far = (kl_d >= KAP + 8 * DEL) & nonflat
    if bool(far.any()):
        got = fb_isolated[far].flatten(1).norm(dim=1)
        want = (LAM * DEL / gn2[far].clamp(min=1e-16).sqrt())
        assert torch.allclose(got, want, atol=1e-5), (
            "(19c) saturated fb pull must equal lam*delta/||g|| exactly")

    # (19d) noise band: off-band rows carry zero noise; RNG stream identical
    offband = (kl_d > KAP + DEL)
    bandkeep = (kl_d >= KAP - DEL) & (kl_d <= KAP + DEL)
    noise_soft = (disp_soft - fb_isolated)
    if bool(offband.any()):
        assert float(noise_soft[offband].abs().max()) == 0.0, (
            "(19d) off-band rows must carry ZERO noise in the soft arm")
    assert bool(bandkeep.any()) and float(
        noise_soft[bandkeep].abs().max()) > 0.0, (
        "(19d) in-band rows must KEEP their noise (band-keep branch)")
    st0 = torch.get_rng_state()
    run(mk(orbit_fb_clamp="none"))
    st_none = torch.get_rng_state()
    torch.set_rng_state(st0)
    run(mk(orbit_fb_clamp="soft"))
    st_soft = torch.get_rng_state()
    assert torch.equal(st_none, st_soft), (
        "(19d) RNG stream must be identical across fb_clamp modes")

    # (19e) guards
    try:
        mk(orbit_fb_clamp="band")
        raise AssertionError("(19e) fb_clamp='band' must raise")
    except ValueError:
        pass
    # P2-1: force the telemetry tick print path (2500-call threshold is
    # never reached by the suite otherwise) -- must not crash.
    p_tick = mk(orbit_fb_clamp="soft")
    p_tick._orb_calls = 2499
    run(p_tick)
    print("[check 19] fb soft-clamp: none bit-identical; soft == analytic "
          "tanh formula; in-band ~legacy (<=10%); saturated pull == "
          "lam*delta/||g||; off-band noise zeroed with RNG stream preserved; "
          "band-keep exercised; telemetry tick OK; guards OK")
def main():
    print("=" * 60)
    print("SCOUT guidance wiring -- hermetic dummy verify")
    print(f"LPB base DP import: {'OK' if _LPB_AVAILABLE else 'MISSING'}"
          + (f" ({_IMPORT_ERROR.__class__.__name__}: {_IMPORT_ERROR})"
             if not _LPB_AVAILABLE else ""))
    print("=" * 60)

    harness = _build_harness()
    policy, planner, scout_vib, current_obs, cond_data, cond_mask, global_cond = harness
    horizon, action_dim = cond_data.shape[1], cond_data.shape[2]

    check_planner_compute_loss(policy, planner, scout_vib, current_obs,
                               horizon, action_dim)
    check_guidance_onoff(policy, planner, scout_vib, current_obs,
                         cond_data, cond_mask, global_cond)
    check_cost_curve(policy, planner, scout_vib, current_obs,
                     cond_data, cond_mask, global_cond)
    check_grad_batch_invariance(policy, planner, scout_vib, current_obs,
                                cond_data, cond_mask)
    # check 6 replaces the policy's planner with the expert one -- run LAST.
    check_expert_guidance(policy, planner, scout_vib, current_obs,
                          cond_data, cond_mask)
    # orbit guidance checks (10-12) replace the planner as well; run last.
    check_orbit_guidance(policy, scout_vib, current_obs,
                         cond_data, cond_mask, global_cond)
    # sector mode check (13) needs no policy (planner-level only).
    check_orbit_sector(scout_vib, current_obs, cond_data)
    # noise-anneal check (14), planner-level only.
    check_orbit_anneal(scout_vib, current_obs, cond_data)
    check_orbit_ray(policy, scout_vib, current_obs, cond_data)
    # merged single-backward equivalence (16) -- perf 方案一+二, 2026-09-01.
    check_orbit_merged(policy, scout_vib, current_obs, cond_data)
    # eta-dimless normalization (17) -- orbit-hparam-dev, 2026-09-02.
    check_orbit_eta_dimless(policy, scout_vib, current_obs, cond_data)
    # round-dependent sigma ceiling (18) -- orbit-hparam-dev, 2026-09-02.
    check_orbit_sigma_schedule(scout_vib, current_obs, cond_data)
    # fb soft-clamp (19) -- orbit-hparam-dev, 2026-09-02.
    check_orbit_fb_clamp(scout_vib, current_obs, cond_data)

    print("-" * 60)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
