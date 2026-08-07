import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Dict
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scout.policy.diffusion_modules.conditional_unet1d import ConditionalUnet1D
from scout.policy.diffusion_modules.mask_generator import LowdimMaskGenerator
from scout.policy.exploration import apply_modal_level_exploration

class CustomDDIMScheduler(DDIMScheduler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    # not changed yet
    def _get_variance(self, timestep, prev_timestep):
        # print("Using custom variance")
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev
        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

        return variance


class DiffusionUNetPolicy(nn.Module):
    def __init__(self,
            action_dim,
            horizon,
            n_obs_steps,
            obs_feature_dim,
            num_inference_steps=20,
            diffusion_step_embed_dim=256,
            down_dims=(256,512),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            noise_scheduler_type="ddim",

            # parameters passed to step
            **kwargs):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale,
            remove_upsample=horizon==1
        )

        # create noise scheduler
        if noise_scheduler_type in ["ddim", "DDIM"]:
            self.noise_scheduler = DDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon"
            )
        elif noise_scheduler_type in ["custom_ddim", "CustomDDIM"]:
            self.noise_scheduler = CustomDDIMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                set_alpha_to_one=True,
                steps_offset=0,
                prediction_type="epsilon"
            )
        elif noise_scheduler_type in ["ddpm", "DDPM"]:
            self.noise_scheduler = DDPMScheduler(
                num_train_timesteps=100,
                beta_start=0.0001,
                beta_end=0.02,
                beta_schedule="squaredcos_cap_v2",
                clip_sample=True,
                variance_type="fixed_small",
                prediction_type="epsilon"
            )
        else:
            raise ValueError(f"Unsupported noise scheduler type {noise_scheduler_type}")

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        self.n_obs_steps = n_obs_steps
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

        self.enable_exploration = False
        self.tau1 = 0.0
        self.tau2 = 1.0
        self.noise_scale = 0.01
        self.lowdim_size = 0
        self.noise_scale_for_lowdim = 0.01

        self.enable_exploration_debug = False

        self.enable_action_noise = False
        self.action_noise_scale = 0.05

        self.enable_cfg = False
        self.cfg_scale = 1.0
        self.cfg_model:ConditionalUnet1D = None

    # ========= inference  ============
    def conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            return_intermediate=False,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        if return_intermediate:
            action_list = []

        if self.enable_exploration and self.enable_exploration_debug:
            # test what if noise is static and not decaying
            global_cond_debug = global_cond + torch.randn_like(global_cond) * self.noise_scale

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            if return_intermediate:
                action_list.append(trajectory.clone())

            # 2. predict model output
            if self.enable_exploration:
                if not self.enable_exploration_debug:
                    global_cond_image, global_cond_lowdim = global_cond.split([self.obs_feature_dim - self.lowdim_size, self.lowdim_size], dim=-1)
                    _global_cond_image = apply_modal_level_exploration(
                        global_cond_image, t, scheduler.config.num_train_timesteps,
                        tau1=self.tau1, tau2=self.tau2, noise_scale=self.noise_scale
                    )
                    _global_cond_lowdim = apply_modal_level_exploration(
                        global_cond_lowdim, t, scheduler.config.num_train_timesteps,
                        tau1=self.tau1, tau2=self.tau2, noise_scale=self.noise_scale_for_lowdim
                    )
                    _global_cond = torch.cat([_global_cond_image, _global_cond_lowdim], dim=-1)
                else:
                    _global_cond = global_cond_debug
            else:
                _global_cond = global_cond

            model_output = model(trajectory, t,
                local_cond=local_cond, global_cond=_global_cond)
            if self.enable_cfg:
                assert self.cfg_model is not None, "cfg_model must be set if enable_cfg is True"
                # apply classifier-free guidance
                model_output_cfg = self.cfg_model(trajectory, t,
                    local_cond=local_cond, global_cond=None)
                model_output = model_output_cfg + self.cfg_scale * (model_output - model_output_cfg)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        if self.enable_action_noise:
            trajectory += torch.randn_like(trajectory) * self.action_noise_scale

        if return_intermediate:
            action_list.append(trajectory.clone())
            return trajectory, action_list

        return trajectory

    def guided_conditional_sample(self,
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            classifier_guidance=False,
            s_bar_t=None, z=None, vib_enc=None, bridge=None,
            guidance_scale=1.0, guidance_start_timestep=0,
            return_cost_curve=False,
            # keyword arguments to scheduler.step
            **kwargs
        ):
        """LPB-style classifier-guided denoising with SCOUT cost
        (scout_design.md §4, mirroring LPB
        ``diffusion_unet_hybrid_image_policy.py:212-271`` verbatim).

        Differences from :meth:`conditional_sample`:
          * opens grad on ``trajectory`` (``x_t``) each step and backprops the
            SCOUT cost on the one-step clean-action estimate ``x̂_0`` to steer
            ``x_t`` -- ``x̂_0`` is *not* modified, only ``x_t`` (LPB invariant).
          * **no SIME / CFG / debug branches** -- guidance is orthogonal to
            SIME; if you need both, wire them as a future wrapper.

        Differences from LPB:
          * **drops the OOD gate (b)** ``current_cost > threshold`` (SCOUT is
            exploration -- every chunk-step is guided; design §4 门控).
          * cost = SCOUT VIB re-encoding gap ``mean_t ‖z − μ(s̄_t, a_t)‖²``
            (``scout.guidance.cost.scout_cost``), not LPB's NN-to-demo distance.

        Args for guidance (only used when ``classifier_guidance`` is True):
          s_bar_t  : ``(B, s_latent_dim)`` encoded current obs (fixed across chunk).
          z        : ``(B, style_dim)`` sampled skill latent (fixed across chunk).
          vib_enc  : a :class:`scout.model.vib.VIBIBEncoder`-like callable.
          bridge   : :class:`scout.normalizer.ActionNormalizerBridge` (identity
                     in stage-1 -- DP and VIB both raw).
          guidance_scale          : η in ``η·√(1−ᾱ_t)``.
          guidance_start_timestep : gate (a) -- guide only when ``t < this`` (the
                     last K denoising steps; standard CG practice).

        When ``classifier_guidance=False`` (or ``guidance_scale=0`` with the
        identity bridge) the output *values* are identical to
        :meth:`conditional_sample` run on the same model + seed: the only
        per-step change is ``trajectory.detach().requires_grad_()``, which
        preserves values.
        """
        # local imports keep the ported DP importable without the guidance stack
        from scout.guidance.cost import scout_cost

        model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)

        scheduler.set_timesteps(self.num_inference_steps)

        cost_curve = [] if return_cost_curve else None

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. open grad on x_t, then predict ε_θ(x_t, t)
            trajectory = trajectory.detach().requires_grad_()
            model_output = model(trajectory, t,
                local_cond=local_cond, global_cond=global_cond)

            # 3. SCOUT classifier guidance -- gate (a) ONLY (OOD gate dropped).
            if classifier_guidance and t < guidance_start_timestep:
                x0_hat = scheduler.step(model_output, t, trajectory).pred_original_sample
                loss = scout_cost(x0_hat, s_bar_t, z, vib_enc, bridge)
                if cost_curve is not None:
                    cost_curve.append(float(loss.detach()))
                cond_grad = -torch.autograd.grad(loss, trajectory)[0]
                scale = guidance_scale * (1 - scheduler.alphas_cumprod[t]).sqrt()
                trajectory = trajectory.detach() + scale * cond_grad

            # 4. DDPM/DDIM reverse step (model_output from step 2; possibly
            #    steered trajectory from step 3).
            trajectory = scheduler.step(
                model_output, t, trajectory,
                generator=generator,
                **kwargs
                ).prev_sample

        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]

        if return_cost_curve:
            return trajectory, cost_curve
        return trajectory

    def predict_action(self, readout) -> Dict[str, torch.Tensor]:
        if readout is None:
            # unconditional diffusion
            assert self.obs_feature_dim == 0, "readout must be provided if obs_feature_dim > 0"
            B = 1  # use default batch size
            T = self.horizon
            Da = self.action_dim

            device = next(self.model.parameters()).device
            dtype = next(self.model.parameters()).dtype
        else:
            B = readout.shape[0]
            T = self.horizon
            Da = self.action_dim
            Do = self.obs_feature_dim
            To = self.n_obs_steps

            # build input
            device = readout.device
            dtype = readout.dtype
            obs_features = readout
            assert obs_features.shape[0] == B * To

        # condition through global feature
        local_cond = None
        global_cond = None
        if readout is not None:
            # reshape back to B, Do
            global_cond = obs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        action_pred = sample[...,:Da]

        return action_pred

    def predict_action_and_return_intermediate(self, readout):
        B = readout.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = readout.device
        dtype = readout.dtype
        obs_features = readout
        assert obs_features.shape[0] == B * To

        # condition through global feature
        local_cond = None
        global_cond = None
        # reshape back to B, Do
        global_cond = obs_features.reshape(B, -1)
        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sample, action_list = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            return_intermediate=True,
            **self.kwargs)
        action_pred = sample[...,:Da]
        return action_pred, action_list

    # ========= training  ============
    def compute_loss(self, readout, actions):
        batch_size = readout.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = actions
        cond_data = trajectory
        assert readout.shape[0] == batch_size * self.n_obs_steps
        # reshape back to B, Do
        global_cond = readout.reshape(batch_size, -1) # (B, T*C)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss

    def compute_weighted_loss(self, readout, actions, weights=None, weight_type="binary"):
        if readout is None:
            # unconditional diffusion
            assert self.obs_feature_dim == 0, "readout must be provided if obs_feature_dim > 0"
            batch_size = actions.shape[0]  # use actions to determine batch size
        else:
            batch_size = readout.shape[0]

        # handle different ways of passing observation
        local_cond = None
        global_cond = None
        trajectory = actions
        cond_data = trajectory
        if readout is not None:
            assert readout.shape[0] == batch_size * self.n_obs_steps
            # reshape back to B, Do
            global_cond = readout.reshape(batch_size, -1) # (B, T*C)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)

        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]

        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps,
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        if weights is not None:
            if weight_type=="binary":
                weights[weights<0] = 0
                weights[weights>0] = 1
            elif weight_type=="exp":
                weights = torch.exp(weights)
            elif weight_type=="sigmoid":
                weights = torch.sigmoid(weights)
            elif weight_type=="not_neg":
                weights[weights<0] = 0
            else:
                raise ValueError(f"Unsupported weight type {self.weight_type}")
            loss = loss * weights.unsqueeze(-1).type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        # loss = loss.mean()
        return loss
