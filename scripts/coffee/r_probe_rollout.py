"""Rollout-convention R probe (user idea, round 2, 2026-09-22).

R = eta * sqrt(1-alphabar_t) * |g_rollout| / |a|   measured at the FIRST
guided denoise step (t = gst-1, "gst 的下一步"), on a frozen core batch
B=128 built straight from the core hdf5 exactly the way the base-DP
training dataset reads it (images /255 -> [0,1] CHW; proprio raw; obs
stacked (t-1, t); actions aa->6d like RobomimicReplayImageDataset).

What is measured: the REAL guided denoise loop of ScoutPolicy
(predict_action_dyn_guided) with the task's KLCostPlanner (ATY) attached
at the campaign dose (eta0, kappa=2.5, gst from the eval config). The
planner's guided_step is wrapped to record, per denoise step:
noise_scale, elementwise mean|cond_grad|, per-row L2 mean, phase-2
fraction. The per-element injected magnitude at step k is
eta * ns_k * meanabs(g_k).

Note on scales: the injection lives in the DP-NORMALIZED action space;
|a| denominators are reported BOTH raw (abs_actions mean|.|, same number
as the first R table) and DP-normalized (the batch's own action chunks
pushed through dp.normalizer['action']), so the ratio is readable in
either convention.

Deterministic: batch indices fixed (demo-major, stride), obs center-crop
76 only inside the E_s adapter (val convention), torch.manual_seed(0)
before the sampling call.

usage (server):
  python scripts/coffee/r_probe_rollout.py \
      --eval-config configs/eval_can_entropy.yaml \
      --dp-ckpt <DP-base ckpt> --vib-ckpt <dyn-base ckpt> \
      --core-hdf5 <core hdf5> --eta0 3.0 --kappa0 2.5 \
      [--batch-size 128] --out <json>
"""
import argparse
import json
import os
import sys

sys.path.insert(0, ".")


def main():
    import h5py
    import numpy as np
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--dp-ckpt", required=True)
    ap.add_argument("--vib-ckpt", required=True)
    ap.add_argument("--core-hdf5", required=True)
    ap.add_argument("--eta0", type=float, required=True)
    ap.add_argument("--kappa0", type=float, default=2.5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    from scout.eval.factories import load_cfg, make_lpb_dp_factory, make_scout_vib_factory
    from scout.eval.rollout import make_action_bridge, make_obs_adapter
    from scout.guidance.entropy_costs import KLCostPlanner

    cfg = load_cfg(args.eval_config)
    view_names = list(cfg.eval.view_names)
    proprio_keys = list(cfg.eval.proprio_keys)
    gst = int(cfg.exploration.get("guidance_start_timestep", 50))

    # ---- frozen core batch straight from the hdf5 (dataset convention) -- #
    with h5py.File(args.core_hdf5, "r") as f:
        demos = sorted(f["data"].keys())
        obs_i, act_i = [], []
        need = args.batch_size
        stride = max(1, sum(len(f[f"data/{d}/abs_actions"]) for d in demos)
                     // (need * 4))
        for d in demos:
            g = f[f"data/{d}"]
            n = g["abs_actions"].shape[0]
            for t in range(1, n - 1, stride):
                if len(obs_i) >= need:
                    break
                o = {}
                for v in view_names:
                    # (2, H, W, C) uint8 -> float/255 -> CHW
                    img = np.stack([g[f"obs/{v}"][t - 1], g[f"obs/{v}"][t]])
                    o[v] = np.moveaxis(img, -1, 1).astype(np.float32) / 255.0
                for k in proprio_keys:
                    o[k] = np.stack([g[f"obs/{k}"][t - 1],
                                     g[f"obs/{k}"][t]]).astype(np.float32)
                obs_i.append(o)
                act_i.append(g["abs_actions"][t].astype(np.float32))
            if len(obs_i) >= need:
                break
    B = len(obs_i)
    obs_dict = {
        k: torch.as_tensor(np.stack([o[k] for o in obs_i])).to(dev)
        for k in list(view_names) + list(proprio_keys)}

    # actions: raw 7-dim (or 14) -> DP 10-dim (20) 6d-rot, like the dataset
    from diffusion_policy.model.common.rotation_transformer import (
        RotationTransformer,)
    rot = RotationTransformer("axis_angle", "rotation_6d")

    def to_dp_action(a):                     # (B, 7|14) -> (B, 10|20)
        a = np.asarray(a)
        dual = a.shape[-1] == 14
        if dual:
            a = a.reshape(-1, 2, 7)
        pos, rv, grip = a[..., :3], a[..., 3:6], a[..., 6:]
        r6 = rot.forward(rv)
        out = np.concatenate([pos, r6, grip], axis=-1).astype(np.float32)
        return out.reshape(-1, 20) if dual else out

    act_raw = np.stack(act_i)                                   # (B, 7|14)
    act_dp = to_dp_action(act_raw)                              # (B, 10|20)

    # ---- policy + planner (verbatim rollout_pipeline._attach_planner) --- #
    dp = make_lpb_dp_factory(dev)(args.dp_ckpt)
    dp.eval()
    cfg.vib.ckpt_path = args.vib_ckpt
    cfg.vib.base_dp_ckpt = args.dp_ckpt
    scout_vib = make_scout_vib_factory(cfg, dev)(args.vib_ckpt)
    scout_vib.eval()

    bridge = make_action_bridge(dp)
    base_adapt = make_obs_adapter(view_names, proprio_keys)

    def obs_adapter(current_obs):
        # my obs are already [0,1]; the rollout adapter expects env-scale
        # [0,255] and divides by 255 -> pre-multiply so the net E_s input is
        # the training-convention [0,1] + center crop 76.
        scaled = {k: (v * 255.0 if k in view_names else v)
                  for k, v in current_obs.items()}
        return base_adapt(scaled)

    planner = KLCostPlanner(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                            cap=args.kappa0, eta_dimless=False)

    recs = []
    orig_step = planner.guided_step

    def wrapped(traj, x0, obs, noise_scale):
        out = orig_step(traj, x0, obs, noise_scale)
        cond_grad, disp, p2, _rl = out
        with torch.no_grad():
            recs.append({
                "t_first_guided": None,          # filled by caller (k-th call)
                "ns": float(noise_scale),
                "magabs": float(cond_grad.abs().mean()),
                "rownorm": float(cond_grad.flatten(1).norm(dim=1).mean()),
                "p2frac": (float(p2.float().mean())
                           if p2 is not None else None),
                "disp_abs": (float(disp.abs().mean())
                             if disp is not None else None)})
        return out

    planner.guided_step = wrapped
    dp.initialize_scout_planner(planner, gst, args.eta0)

    torch.manual_seed(0)
    result = dp.predict_action_dyn_guided(obs_dict)

    # guided calls happen for t < gst in DESCENDING t order:
    # call k <-> t = gst-1-k
    ts = [gst - 1 - k for k in range(len(recs))]
    for k, r in enumerate(recs):
        r["t_first_guided"] = ts[k]
    first = recs[0] if recs else None

    # denominators
    meanabs_a_raw = float(np.abs(act_raw).mean())
    nact = dp.normalizer["action"]
    meanabs_a_norm = float(nact.normalize(
        torch.as_tensor(act_dp, dtype=torch.float32, device=dev)
    ).abs().mean())

    R = None
    if first is not None:
        inj = args.eta0 * first["ns"] * first["magabs"]     # per-element
        R = inj / meanabs_a_raw
        R_norm = inj / meanabs_a_norm
    else:
        R_norm = None

    out = {
        "method": "rollout_R",
        "eta0": args.eta0, "kappa0": args.kappa0, "gst": gst, "B": B,
        "meanabs_a_raw": meanabs_a_raw, "meanabs_a_norm": meanabs_a_norm,
        "first_guided_step": first,
        "R_raw": R, "R_norm": R_norm,
        "per_step": recs,
        "action_pred_meanabs": float(result["action"].abs().mean()),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[r-probe-rollout] B={B} gst={gst} steps={len(recs)}")
    if first is not None:
        print(f"[r-probe-rollout] t_first={first['t_first_guided']} "
              f"ns={first['ns']:.4g} magabs={first['magabs']:.4g} "
              f"p2frac={first['p2frac']} "
              f"R_raw={round(R, 4)} R_norm={round(R_norm, 4)}")
    print(f"[r-probe-rollout] saved -> {args.out}")


if __name__ == "__main__":
    main()
