# SCOUT 测试期 classifier guidance。
#
# 思路(见 idea/idea_notes.md §5):
#   z ~ N(0,I) 整段生成定住;在冻结 base DP 的去噪循环里,每步把
#       model_output += -guidance_scale * ∇_a Cost
#   其中 Cost(a, z | s) = || z - mu(s, a) ||_2 ,mu 来自训练好的 SCOUT 编码器。
#   即「在 DP 去噪方向上叠加负 Cost 梯度」,把动作往「编码回去正好等于 z」的方向推。
#
# 结构对齐 SOE 的 policy/diffusion.py:conditional_sample,区别仅在每步注入 guidance。
#
# 关键约定(低维设定):base DP 的 readout == low_dim 状态向量(MultiImageObsEncoder 对
#   low_dim key 直接拼接、排序),故 global_cond 直接取 state 向量;SCOUT 的 state 也按
#   同序拼接(transition_dataset 已排序 obs_keys),两端一致。

import os as _os
import sys as _sys

_HERE = _os.path.dirname(_os.path.abspath(__file__))
if _HERE not in _sys.path:
    _sys.path.insert(0, _HERE)
_SOE_SRC = _os.path.normpath(_os.path.join(_HERE, "..", "..", "SOE", "src"))
if _SOE_SRC not in _sys.path:
    _sys.path.insert(0, _SOE_SRC)

import json
import torch


def load_base_dp(dp_config_path, dp_ckpt_path, device):
    """用 base DP 的 config 重建 DP 模块,加载冻结权重。"""
    from policy.dp import DP  # SOE
    with open(dp_config_path, "r") as f:
        cfg = json.load(f)
    dp = DP(**cfg["policy"]["params"]).to(device)
    state_dict = torch.load(dp_ckpt_path, map_location=device)
    dp.load_state_dict(state_dict, strict=False)
    dp.eval()
    for p in dp.parameters():
        p.requires_grad_(False)
    print("base DP loaded from", dp_ckpt_path)
    return dp, cfg


def load_scout(scout_config_path, scout_ckpt_path, device):
    """重建 SCOUT 模块并加载权重(测试期只用编码器,但仍整模块加载)。"""
    from scout_policy import SCOUT
    with open(scout_config_path, "r") as f:
        cfg = json.load(f)
    scout = SCOUT(**cfg["policy"]["params"]).to(device)
    state_dict = torch.load(scout_ckpt_path, map_location=device)
    scout.load_state_dict(state_dict, strict=False)
    scout.eval()
    for p in scout.parameters():
        p.requires_grad_(False)
    print("SCOUT loaded from", scout_ckpt_path)
    return scout, cfg


@torch.no_grad()
def _dp_model_output(ad, trajectory, t, global_cond):
    """冻结 base DP 在当前带噪动作上的去噪输出(ε̂)。"""
    return ad.model(trajectory, t, local_cond=None, global_cond=global_cond)


def guided_sample(base_dp, scout, state, z, guidance_scale=1.0,
                  num_inference_steps=None, first_action_only=True,
                  return_cost_curve=False, init_noise=None):
    """
    classifier-guided 去噪采样。

    Args:
        base_dp: 冻结的 SOE DP(low_dim)。
        scout:   训好的 SCOUT(用其编码器的 mu)。
        state:   (B, state_dim) 低维状态向量(= base DP 的 readout)。
        z:       (B, style_dim) skill 目标,整段生成定住。
        guidance_scale: guidance 强度(可正可负;若 Cost 反而上升,请取负号)。
        first_action_only: Cost 只取 chunk 的第一个动作 a_t(与编码器训练口径一致)。
        return_cost_curve: 返回 Cost 随去噪步的曲线。
    Returns:
        action chunk (B, T, Da) [, cost_curve(list[float])]
    """
    ad = base_dp.action_decoder
    model = ad.model
    scheduler = ad.noise_scheduler
    T, Da = ad.horizon, ad.action_dim
    B = state.shape[0]
    device, dtype = state.device, state.dtype

    global_cond = state.reshape(B, -1)
    cond_data = torch.zeros((B, T, Da), device=device, dtype=dtype)
    cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

    if init_noise is None:
        init_noise = torch.randn((B, T, Da), device=device, dtype=dtype)
    trajectory = init_noise.clone()
    scheduler.set_timesteps(num_inference_steps or ad.num_inference_steps)

    cost_curve = []
    for t in scheduler.timesteps:
        trajectory[cond_mask] = cond_data[cond_mask]  # 无 inpaint,实际不变

        if guidance_scale != 0:
            # 在带 grad 的副本上算 ∇_trajectory Cost
            with torch.enable_grad():
                traj_g = trajectory.detach().requires_grad_(True)
                model_output = model(traj_g, t, local_cond=None, global_cond=global_cond)
                a = traj_g[:, 0, :] if first_action_only else traj_g.reshape(B, -1)
                mu, _, _ = scout.encode(state, a)        # mu(s, a),带 grad
                cost = torch.sum((z - mu).pow(2))         # 对 batch 与维求和
                grad = torch.autograd.grad(cost, traj_g)[0]
                model_output = model_output - guidance_scale * grad
                if return_cost_curve:
                    cost_curve.append((cost.detach().item() / max(B, 1)))
            model_output = model_output.detach()
        else:
            with torch.no_grad():
                model_output = model(trajectory, t, local_cond=None, global_cond=global_cond)

        trajectory = scheduler.step(model_output, t, trajectory.detach()).prev_sample

    trajectory[cond_mask] = cond_data[cond_mask]
    if return_cost_curve:
        return trajectory[..., :Da], cost_curve
    return trajectory[..., :Da]
