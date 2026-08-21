"""ScoutPolicy: LPB base DP + SCOUT classifier guidance (scout_design.md §4).

Subclasses the LPB fork :class:`DiffusionUnetHybridImagePolicy` (the SOE base
DP, ported top-level under ``diffusion_policy/``) and overrides **only**
``guided_conditional_sample`` -- swapping the LPB NN-to-demo cost for SCOUT's
VIB re-encoding gap and dropping the OOD gate. The LPB fork file itself is
untouched (subclass-only -- task verify step 4).

What changes vs LPB ``guided_conditional_sample``
(``diffusion_policy/policy/diffusion_unet_hybrid_image_policy.py:212-271``):

  1. **Cost**: LPB's ``self.planner.compute_loss`` (NN-distance to demo latents)
     -> :meth:`scout.guidance.planner.ScoutPlanner.compute_loss`
     (``mean_t [−log q_θ(z|s̄_t, a_t)]`` -- Gaussian NLL of the sampled skill
     latent z under the encoder's diagonal Gaussian; user decision 2026-08-14).
  2. **Gate**: LPB's
     ``if classifier_guidance and t < self.guidance_start_timestep and
     current_cost > self.threshold:``
     -> ``if classifier_guidance and t < self.guidance_start_timestep:``.
     The OOD gate (b) (``current_cost > self.threshold``) is dropped per design
     §4 (SCOUT explores actively; no expert-latent NN distance). The pre-loop
     ``compute_current_reward`` + ``correct_num`` bookkeeping is also dropped.
  3. **z + s̄_t management**: the skill latent is fixed once per inference call
     (sampled with a dedicated RNG stream or passed in by the caller) and
     pre-loaded onto the planner; ``s̄_t`` is pre-encoded once (cached across
     denoise steps) -- design §1, §4 ("z 整段定住", "s̄_t 定住").

The loop body (inpaint conditioning, ``trajectory.detach().requires_grad_()``,
``model_output = model(...)``, ``x̂_0 = scheduler.step(...).
pred_original_sample``, ``cond = -autograd.grad(loss, trajectory)[0]``,
``scale = guidance_scale * sqrt(1 - alphas_cumprod[t])``,
``trajectory = trajectory.detach() + scale*cond``,
``scheduler.step(...).prev_sample``) is verbatim LPB -- task step 2.

.. note:: The LPB base DP import requires ``robomimic`` (not installable on the
   Windows verify host). The parent class is therefore imported via try/except:
   when robomimic is missing, ``ScoutPolicy`` falls back to ``object`` as a
   bare shell -- enough to bind :meth:`guided_conditional_sample` and drive the
   hermetic dummy verify (which builds the model/scheduler/planner by hand via
   ``__new__``). With robomimic installed (training env), ``ScoutPolicy`` is a
   true subclass and inherits the full LPB interface (``predict_action`` etc.).
"""

from __future__ import annotations

from typing import Optional

import torch

try:  # pragma: no cover - exercised only in the training env (robomimic present)
    from diffusion_policy.policy.diffusion_unet_hybrid_image_policy import (
        DiffusionUnetHybridImagePolicy,
    )

    _LPB_AVAILABLE = True
    _IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # robomimic / hydra missing -> dummy-verify host
    DiffusionUnetHybridImagePolicy = object  # type: ignore[misc,assignment]
    _LPB_AVAILABLE = False
    _IMPORT_ERROR = e

from scout.guidance.planner import ScoutPlanner


class ScoutPolicy(DiffusionUnetHybridImagePolicy):
    """LPB base DP with SCOUT classifier guidance plugged in.

    Construction (training env): build the LPB base DP via the LPB ``__init__``
    (needs robomimic/hydra + shape_meta etc.), then attach a
    :class:`ScoutPlanner` through :meth:`initialize_scout_planner`. The planner
    carries the frozen :class:`~scout.model.scout_vib.ScoutVIB` (``E_s`` + VIB
    encoder) and the action-space bridge.

    Dummy verify (no robomimic): construct via ``ScoutPolicy.__new__(ScoutPolicy)``
    and set ``model``, ``noise_scheduler``, ``num_inference_steps``, ``kwargs``
    plus the SCOUT attrs by hand; the inherited LPB ``__init__`` is skipped.
    """

    def __init__(
        self,
        *args,
        scout_planner: Optional[ScoutPlanner] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # mirror LPB's planner / guidance attrs (None here so attribute access
        # never raises; properly set by initialize_scout_planner).
        self.scout_planner: Optional[ScoutPlanner] = scout_planner
        self.guidance_start_timestep: Optional[int] = None
        self.guidance_scale: Optional[float] = None

    # ------------------------------------------------------------------ #
    # SCOUT planner attachment (LPB analogue: initialize_planner)
    # ------------------------------------------------------------------ #
    def initialize_scout_planner(
        self,
        planner: ScoutPlanner,
        guidance_start_timestep: int,
        guidance_scale: float,
    ):
        """Attach the SCOUT planner + guidance hyper-params. Must be called
        before ``guided_conditional_sample(..., classifier_guidance=True)``."""
        self.scout_planner = planner
        self.guidance_start_timestep = int(guidance_start_timestep)
        self.guidance_scale = float(guidance_scale)

    # ------------------------------------------------------------------ #
    # guidance -- overrides LPB (subclass only; LPB fork file untouched)
    # ------------------------------------------------------------------ #
    def guided_conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        classifier_guidance: bool = False,
        current_obs=None,
        text_latents=None,
        z: Optional[torch.Tensor] = None,
        return_cost_curve: bool = False,
        **kwargs,
    ):
        """LPB loop body verbatim, with SCOUT cost + gate (b) dropped.

        Args:
            classifier_guidance: turn guidance on (design §4).
            current_obs: encoded once to ``s̄_t`` and held fixed across the loop
                (passed straight through to ``scout_planner.compute_loss``).
            text_latents: kept in the signature for LPB interface parity; SCOUT
                stage-1 has no language, so it is **not** injected into
                ``current_obs`` (LPB's behaviour would mutate the obs dict,
                which is wrong for E_s's ``{"visual","proprio"}`` format).
            z: optional skill latent ``(B, style_dim)`` fixed across the loop.
                If ``None`` and ``classifier_guidance`` is on, sampled from a
                dedicated RNG stream (NOT the ``generator`` -- so trajectory
                init + DDPM stochasticity are unaffected; pass ``z`` explicitly
                for reproducibility, e.g. E2 z-set sweep).
            return_cost_curve: if True, also return the per-step cost curve
                ``(loss.item()`` at each guided step); design §5 E2-③
                "cost-direction" check (curve should fall -- a rising curve
                means the sign is flipped).

        Returns:
            ``trajectory`` if ``return_cost_curve`` is False, else
            ``(trajectory, cost_curve)``.
        """
        if classifier_guidance:
            if self.scout_planner is None:
                raise RuntimeError(
                    "classifier_guidance=True but no scout_planner is attached"
                    " (call initialize_scout_planner first)."
                )
            if self.guidance_start_timestep is None:
                raise RuntimeError("guidance_start_timestep not set.")

        model = self.model
        scheduler = self.noise_scheduler

        # 0. init trajectory ~ N(0, I) (LPB L229-233).
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        cost_curve: list[float] = []
        # Throwaway RNG for the gate's x̂_0 read-back: ``scheduler.step`` draws
        # ``variance_noise`` (diffusers DDPMScheduler.step L84) as a side effect
        # even when only ``pred_original_sample`` is read, and LPB's call passes
        # no generator -> the draw falls through to the *global* RNG. With LPB's
        # OOD gate this was rare; SCOUT drops that gate (design §4) and enters
        # every step, so the spurious draw would desync the model-output stream
        # and break the ``scale=0 == unguided`` invariant (task verify step 2).
        # ``pred_original_sample`` is deterministic (the noise only enters
        # ``prev_sample``), so routing the side-effect draw to a throwaway
        # generator preserves LPB's call syntax and value exactly while keeping
        # the main ``generator`` + global RNG streams clean.
        _gate_gen = torch.Generator(device=condition_data.device)
        if classifier_guidance:
            # fix z for the whole loop (design §1: "z 整段定住"). Sampled WITHOUT
            # the `generator` so the trajectory / DDPM RNG stream is unchanged
            # <-> guidance_scale=0 reproduces the unguided trajectory exactly
            # (task verify step 2). Caller can pass z explicitly for full
            # reproducibility (E2 z-set sweep).
            B = condition_data.shape[0]
            # z precedence: explicit ``z`` arg  >  caller-locked ``planner.z``
            # (per-trajectory skill, set via planner.set_z and held across ALL
            # chunks of one rollout -- scout_design §1 "z 整段定住", distinct
            # from SOE which resamples z per chunk)  >  fresh z~N(0,I) (per-chunk,
            # SOE-style). ``planner.reset()`` between calls unlocks -> per-chunk.
            if z is None:
                z = self.scout_planner.z
            if z is None:
                z = torch.randn(
                    size=(B, self.scout_planner.scout_vib.style_dim),
                    device=condition_data.device,
                    dtype=condition_data.dtype,
                )
            else:
                z = z.to(device=condition_data.device, dtype=condition_data.dtype)
            self.scout_planner.set_z(z)
            # pre-encode s̄_t once (design §4: "s̄_t 定住") -- cached across
            # denoise steps; compute_loss reuses it instead of re-running the
            # frozen ResNet every step.
            if current_obs is not None:
                self.scout_planner.set_current_obs(current_obs)

        for t in scheduler.timesteps:
            # 1. apply conditioning (inpaint) -- LPB L246.
            trajectory[condition_mask] = condition_data[condition_mask]
            trajectory = trajectory.detach().requires_grad_()

            # 2. predict model noise ε_θ(x_t, t) -- LPB L250-251.
            model_output = model(
                trajectory, t, local_cond=local_cond, global_cond=global_cond
            )

            # --- SCOUT guidance: LPB L253-259 loop body, gate (b) dropped --- #
            if classifier_guidance and t < self.guidance_start_timestep:
                # cost on the one-step clean-action estimate x̂_0 (NOT on ε).
                # ``generator=_gate_gen`` routes the spurious variance-noise
                # draw off the main stream (see comment near _gate_gen decl).
                x0_hat = scheduler.step(
                    model_output, t, trajectory, generator=_gate_gen
                ).pred_original_sample
                # reduction="sum": rows are block-diagonal independent, so the
                # gradient of the SUMMED cost gives each row its full unscaled
                # gradient -- the injected force no longer depends on how many
                # envs share this replan call. Pre-fix the mean reduction
                # divided every row's force by B (effective guidance =
                # guidance_scale/B, B = concurrent envs of the moment); see
                # idea/guidance_batch_scaling_bug.md.
                loss = self.scout_planner.compute_loss(
                    x0_hat, current_obs, reduction="sum"
                )
                cond_grad = -torch.autograd.grad(loss, trajectory)[0]
                grad_scale = self.guidance_scale * (
                    1.0 - scheduler.alphas_cumprod[t]
                ).sqrt()
                trajectory = trajectory.detach() + grad_scale * cond_grad
                if return_cost_curve:
                    # log per-row mean (historical scale of the E2 metric),
                    # not the B-aggregated sum.
                    cost_curve.append(float(loss.item()) / x0_hat.shape[0])

            # 3. DDPM reverse step x_t -> x_{t-1} -- LPB L262-266.
            trajectory = scheduler.step(
                model_output, t, trajectory, generator=generator, **kwargs
            ).prev_sample

        # finally make sure conditioning is enforced -- LPB L269.
        trajectory[condition_mask] = condition_data[condition_mask]

        if return_cost_curve:
            return trajectory, cost_curve
        return trajectory
