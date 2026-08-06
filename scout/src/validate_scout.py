# SCOUT go/no-go 验证(long_term_plan.md 阶段1 硬里程碑)。
#
# 要回答的问题:classifier guidance 能否把冻结 base DP 的动作推向不同的 skill、
# 产生有意义的探索(而非单纯噪声)?三个判据:
#   (1) 多样性:固定初始噪声、只变 z,guidance_scale>0 时动作随 z 明显发散;
#              而 guidance_scale=0 时动作几乎不变(z 不参与)。
#   (2) 一致性:把引导出的动作 a 编码回去,mu(s,a) 应接近目标 z(Cost 小)。
#   (3) Cost 下降:去噪过程中 Cost 应随步数下降(方向正确;若上升则 guidance_scale 取负)。
#
# 三条都满足 => guidance 机制成立 => GO。

import os
import sys
import json
import argparse

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SOE_SRC = os.path.normpath(os.path.join(_HERE, "..", "..", "SOE", "src"))
if _SOE_SRC not in sys.path:
    sys.path.insert(0, _SOE_SRC)

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from guided_sampler import load_base_dp, load_scout, guided_sample  # noqa: E402
from transition_dataset import ScoutTransitionDataset  # noqa: E402


def get_test_states(scout_cfg, eval_filter_key, num_states, device):
    params = dict(scout_cfg["dataset"]["params"])
    params["train_filter_key"] = eval_filter_key
    ds = ScoutTransitionDataset(**params)
    N = min(num_states, len(ds))
    idx = np.random.RandomState(0).choice(len(ds), N, replace=False)
    states = torch.stack([ds[int(i)]["state_t"] for i in idx]).to(device)
    return states, ds.state_dim


@torch.no_grad()
def encode_mu(scout, state, action):
    mu, _, _ = scout.encode(state, action)
    return mu


def run_guided(base_dp, scout, states, init_noise, zs, guidance_scale):
    """对一组 z(固定初始噪声)做 guided 采样,返回 actions (Z,N,T,Da) 与 z (Z,N,style)。"""
    actions = []
    for k in range(zs.shape[0]):
        a = guided_sample(base_dp, scout, states, zs[k],
                          guidance_scale=guidance_scale, init_noise=init_noise)
        actions.append(a)
    return torch.stack(actions)  # (Z,N,T,Da)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp_config", type=str, required=True)
    parser.add_argument("--dp_ckpt", type=str, required=True)
    parser.add_argument("--scout_config", type=str, required=True)
    parser.add_argument("--scout_ckpt", type=str, required=True)
    parser.add_argument("--num_states", type=int, default=64)
    parser.add_argument("--num_z", type=int, default=8)
    parser.add_argument("--guidance_scales", type=str, default="0,1,5,10,20")
    parser.add_argument("--eval_filter_key", type=str, default="valid")
    parser.add_argument("--out_dir", type=str, default="scout/out/validate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.out_dir, exist_ok=True)

    base_dp, _ = load_base_dp(args.dp_config, args.dp_ckpt, device)
    scout, scout_cfg = load_scout(args.scout_config, args.scout_ckpt, device)
    style_dim = scout.style_dim

    states, _ = get_test_states(scout_cfg, args.eval_filter_key, args.num_states, device)
    N = states.shape[0]
    print("num test states:", N, "style_dim:", style_dim)

    # 固定初始噪声 + 固定一组 z,跨 guidance_scale 公平对比
    g = torch.Generator(device=device).manual_seed(0)
    init_noise = torch.randn((N, base_dp.action_decoder.horizon, base_dp.action_dim),
                             device=device, generator=g)
    zs = torch.randn(args.num_z, N, style_dim, device=device, generator=g)

    scales = [float(s) for s in args.guidance_scales.split(",")]
    results = {"scales": scales, "diversity": [], "consistency_err": [], "guided_vs_baseline": []}

    baseline_actions = None
    cost_curve_ref = None
    actions_per_scale = {}
    for scale in scales:
        actions = run_guided(base_dp, scout, states, init_noise, zs, scale)  # (Z,N,T,Da)
        actions_per_scale[scale] = actions
        Z = actions.shape[0]
        flat = actions.reshape(Z, N, -1)
        # (1) 多样性:跨 z 的 std(每维),均值
        diversity = flat.std(dim=0).mean().item()
        # (2) 一致性:把引导动作编码回去,与目标 z 比
        with torch.no_grad():
            a0 = actions[:, :, 0, :]  # (Z,N,Da)
            errs = []
            for k in range(Z):
                mu_back = encode_mu(scout, states, a0[k])  # (N,style)
                errs.append((zs[k] - mu_back).pow(2).sum(dim=-1).sqrt().mean().item())
            consistency_err = float(np.mean(errs))
        # baseline(scale=0)动作
        if scale == scales[0]:
            baseline_actions = actions.mean(dim=0)  # (N,T,Da) ~ 同噪声下相同
        with torch.no_grad():
            gvb = (actions - baseline_actions.unsqueeze(0)).reshape(Z, N, -1).norm(dim=-1).mean().item()
        results["diversity"].append(diversity)
        results["consistency_err"].append(consistency_err)
        results["guided_vs_baseline"].append(gvb)
        print("scale={:>6.2f}  diversity={:.4f}  consistency_err={:.4f}  guided_vs_baseline={:.4f}".format(
            scale, diversity, consistency_err, gvb))

    # (3) Cost 曲线(用最大 scale)
    max_scale = max(s for s in scales if s != 0) if any(s != 0 for s in scales) else 1.0
    _, cost_curve = guided_sample(base_dp, scout, states[:min(8, N)], zs[0, :min(8, N)],
                                  guidance_scale=max_scale, init_noise=init_noise[:min(8, N)],
                                  return_cost_curve=True)

    # ---- 保存指标 ----
    results["cost_curve"] = cost_curve
    results["max_scale_for_cost"] = max_scale
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- 画图 ----
    # 图1:多样性 vs scale
    plt.figure(figsize=(5, 4))
    plt.plot(scales, results["diversity"], "-o", label="inter-z diversity")
    plt.xlabel("guidance_scale")
    plt.ylabel("action std across z")
    plt.title("Diversity vs guidance scale\n(应随 scale 上升;scale=0 时≈0)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "diversity_vs_scale.png"))
    plt.close()

    # 图2:一致性误差 vs scale
    plt.figure(figsize=(5, 4))
    plt.plot(scales, results["consistency_err"], "-o", color="C1")
    plt.xlabel("guidance_scale")
    plt.ylabel(r"$\|z-\mu(s,a_{guided})\|_2$")
    plt.title("Consistency error vs scale\n(应随 scale 下降 => 动作编码回去≈z)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "consistency_vs_scale.png"))
    plt.close()

    # 图3:Cost 随去噪步
    plt.figure(figsize=(5, 4))
    plt.plot(cost_curve, "-o", color="C2")
    plt.xlabel("denoising step")
    plt.ylabel("Cost")
    plt.title("Cost over denoising (scale={})\n(应下降;若上升 => guidance_scale 取负)".format(max_scale))
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "cost_over_steps.png"))
    plt.close()

    # 图4:某一状态、若干 z 的动作 chunk(前 action_dim 维)+ scale=0 基线
    s0_actions = actions_per_scale[max_scale][:, 0, :, :]  # (Z,T,Da)
    base0 = baseline_actions[0]  # (T,Da)
    plt.figure(figsize=(7, 4))
    t_axis = np.arange(s0_actions.shape[1])
    for k in range(min(args.num_z, 8)):
        plt.plot(t_axis, s0_actions[k, :, 0].cpu().numpy(), alpha=0.6)  # 画第0维
    plt.plot(t_axis, base0[:, 0].cpu().numpy(), "k--", linewidth=2, label="baseline (scale=0)")
    plt.xlabel("chunk step")
    plt.ylabel("action dim 0")
    plt.title("One state, different z (scale={})\n不同颜色=不同 z".format(max_scale))
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "actions_per_z.png"))
    plt.close()

    print("\n=== go/no-go 小结 ===")
    print("多样性:scale=0 -> {:.4f} , 最大 scale -> {:.4f}".format(
        results["diversity"][0], max(results["diversity"])))
    print("一致性:scale=0 -> {:.4f} , 最大 scale -> {:.4f}".format(
        results["consistency_err"][0], min(results["consistency_err"])))
    print("Cost 起止:{:.4f} -> {:.4f}".format(cost_curve[0], cost_curve[-1]))
    print("指标与图已保存到:", args.out_dir)


if __name__ == "__main__":
    main()
