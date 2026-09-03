"""Cross-branch golden equivalence probe (2026-09-01, orbit-dev extraction).

Self-contained (imports ONLY the package of the worktree it is run from --
resolved via cwd, so run as:  cd <worktree> && python ../golden_orbit_eq.py).
Builds a fixed mock harness and hashes fixed-seed guided trajectories for
four guide configurations. Two branches are behaviorally identical for the
orbit line iff all four hashes match.
"""
import hashlib
import os
import sys

sys.path.insert(0, os.getcwd())   # the worktree's scout package, not the script dir

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from scout.guidance.policy import ScoutPolicy, _LPB_AVAILABLE
from scout.model.encoder import StateEncoder
from scout.model.scout_vib import ScoutVIB
from scout.normalizer import IdentityBridge


class _DummyModel(nn.Module):
    def forward(self, trajectory, t, local_cond=None, global_cond=None):
        return torch.randn_like(trajectory)


class _MockResNetEncoder(nn.Module):
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
            feat = self.avgpool(self.proj(imgs))
            feat = feat.flatten(1).unsqueeze(1)
            out[v] = rearrange(feat, "(b t) p d -> b t p d", b=b)
        return out


def build(seed=233, B=4, horizon=8, action_dim=7, style_dim=16,
          proprio_dim=10, num_train_timesteps=100, num_inference_steps=10):
    torch.manual_seed(seed)
    E_s = StateEncoder(resnet_encoder=_MockResNetEncoder(
        ("agentview", "robot0_eye_in_hand")),
        view_names=["agentview", "robot0_eye_in_hand"],
        proprio_dim=proprio_dim, proprio_emb_dim=64)
    scout_vib = ScoutVIB(action_dim=action_dim, E_s=E_s, style_dim=style_dim,
                         beta=1.0e-3).eval()
    policy = ScoutPolicy.__new__(ScoutPolicy)
    if _LPB_AVAILABLE:
        torch.nn.Module.__init__(policy)
    policy.model = _DummyModel()
    policy.noise_scheduler = DDPMScheduler(
        num_train_timesteps=num_train_timesteps,
        prediction_type="epsilon", beta_schedule="scaled_linear")
    policy.num_inference_steps = num_inference_steps
    policy.kwargs = {}
    policy.scout_planner = None
    policy.guidance_start_timestep = None
    policy.guidance_scale = None
    current_obs = {
        "visual": {v: torch.randn(B, 1, 3, 128, 128)
                   for v in ("agentview", "robot0_eye_in_hand")},
        "proprio": torch.randn(B, 1, proprio_dim),
    }
    cond_data = torch.zeros(B, horizon, action_dim)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
    return policy, scout_vib, current_obs, cond_data, cond_mask


def run(policy, cond_data, cond_mask, current_obs, planner, seed,
        guidance_scale=1.0):
    policy.initialize_scout_planner(
        planner=planner, guidance_start_timestep=100,
        guidance_scale=float(guidance_scale))
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    policy.scout_planner.reset()
    return policy.guided_conditional_sample(
        cond_data, cond_mask, local_cond=None, global_cond=None,
        generator=gen, classifier_guidance=True, current_obs=current_obs,
        z=None)


def h(t):
    return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()[:16]


def main():
    from scout.guidance.entropy_costs import AtypicalCostPlanner
    from scout.guidance.orbit_costs import OrbitCostPlanner

    policy, scout_vib, obs, cond_data, cond_mask = build()

    # 0) unguided baseline (base DP path)
    torch.manual_seed(233)
    gen = torch.Generator().manual_seed(233)
    t_un = policy.guided_conditional_sample(
        cond_data, cond_mask, generator=gen, classifier_guidance=False)
    print(f"unguided      {h(t_un)}")

    # 1) atypical (shared cost path)
    att = AtypicalCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5)
    print(f"atypical      {h(run(policy, cond_data, cond_mask, obs, att, 233))}")

    # 2) orbit iid (production config of the s233 chain)
    orb = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5,
                           orbit_lam=0.5, orbit_delta=0.25, orbit_sigma=0.25)
    print(f"orbit_iid     {h(run(policy, cond_data, cond_mask, obs, orb, 233))}")

    # 3) orbit ray + sector=det (B2/B4 hooks with row jobs)
    jobs = [(None, 3, j) for j in range(3)] + [(None, 7, 4)]
    ray = OrbitCostPlanner(scout_vib, bridge=IdentityBridge(), cap=2.5,
                           orbit_lam=0.5, orbit_delta=0.25, orbit_sigma=0.25,
                           orbit_sector="det", orbit_sector_seed=42,
                           orbit_climb="ray", orbit_ray_seed=42)
    ray.set_row_jobs(jobs)
    print(f"orbit_ray_det {h(run(policy, cond_data, cond_mask, obs, ray, 233))}")

    print("torch", torch.__version__)


if __name__ == "__main__":
    main()
