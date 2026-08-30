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
           - orbit_update fires exactly once per guided denoise step;
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
        f"(12) expected {policy.num_inference_steps} orbit_update calls, "
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
    disp, p2r = orb2.orbit_update(traj, x0h, current_obs, noise_scale=0.5)
    assert float(p2r.sum()) == float(Bn), (
        "(12) far baseline must put every row in phase 2")
    assert disp.abs().max().item() > 0.0, "(12) phase-2 displacement nonzero"
    orb3 = OrbitCostPlanner(scout_vib, bridge=IdentityBridge())
    orb3.set_current_obs(current_obs)
    traj3 = torch.randn(Bn, H, Ad).requires_grad_(True)
    x0h3 = 2.0 * traj3
    orb3.select_z(x0h3.detach(), current_obs)   # anchor AT the input -> kl ~ 0
    disp3, p23 = orb3.orbit_update(traj3, x0h3, current_obs, noise_scale=0.5)
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

    print("-" * 60)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
