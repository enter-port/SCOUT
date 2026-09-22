#!/usr/bin/env python3
"""thm2_cap_probe.py -- does the ATY guidance KL reach the kappa cap?

Loads seed-233 round-1 DP + dyn (beta=1e-5) and runs a REAL guided denoise on
the frozen core batch (B=128) at the round-1 calibrated dose eta, instrumenting
KLCostPlanner._kl_backward to capture the UNCAPPED per-row KL at every guided
step. Reports the fraction of rows / steps that sit at or above kappa=2.5.

Read-only: touches no campaign file; the only (optional) write is --out json.
Construction mirrors scripts/coffee/cfk_rcalib.py (same frozen batch, same
_attach_planner-equivalent planner mount).
"""
import argparse, json, sys
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
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--kappa", type=float, default=2.5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device("cuda")
    from scout.eval.factories import (load_cfg, make_lpb_dp_factory,
                                      make_scout_vib_factory)
    from scout.eval.rollout import make_action_bridge, make_obs_adapter
    from scout.guidance.entropy_costs import KLCostPlanner

    cfg = load_cfg(args.eval_config)
    view_names = list(cfg.eval.view_names)
    proprio_keys = list(cfg.eval.proprio_keys)
    gst = int(cfg.exploration.get("guidance_start_timestep", 50))

    # ---- frozen core batch (verbatim cfk_rcalib.py / r_probe_rollout.py) -- #
    with h5py.File(args.core_hdf5, "r") as f:
        demos = sorted(f["data"].keys())
        obs_i = []
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
                    img = np.stack([g[f"obs/{v}"][t - 1], g[f"obs/{v}"][t]])
                    o[v] = np.moveaxis(img, -1, 1).astype(np.float32) / 255.0
                for k in proprio_keys:
                    o[k] = np.stack([g[f"obs/{k}"][t - 1],
                                     g[f"obs/{k}"][t]]).astype(np.float32)
                obs_i.append(o)
            if len(obs_i) >= need:
                break
    B = len(obs_i)
    obs_dict = {
        k: torch.as_tensor(np.stack([o[k] for o in obs_i])).to(dev)
        for k in list(view_names) + list(proprio_keys)}

    dp = make_lpb_dp_factory(dev)(args.dp_ckpt)
    dp.eval()
    cfg.vib.ckpt_path = args.vib_ckpt
    cfg.vib.base_dp_ckpt = args.dp_ckpt
    scout_vib = make_scout_vib_factory(cfg, dev)(args.vib_ckpt)
    scout_vib.eval()

    bridge = make_action_bridge(dp)
    base_adapt = make_obs_adapter(view_names, proprio_keys)

    def obs_adapter(current_obs):
        scaled = {k: (v * 255.0 if k in view_names else v)
                  for k, v in current_obs.items()}
        return base_adapt(scaled)

    planner = KLCostPlanner(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                            cap=args.kappa, eta_dimless=False)
    dp.initialize_scout_planner(planner, gst, args.eta)

    # ---- instrument: capture uncapped per-row KL at every guided step ---- #
    steps = []  # list of dict: t (timestep), mean, max, n_ge_cap, frac_ge_cap
    orig_kb = planner._kl_backward

    def kb(trajectory, x0_hat, current_obs=None):
        kl, g = orig_kb(trajectory, x0_hat, current_obs)
        with torch.no_grad():
            k = kl.detach().flatten()          # (B,) uncapped per-row KL
            t = gst - 1 - len(steps)           # guided call k <-> t=gst-1-k
            steps.append({
                "t": t,
                "mean": float(k.mean()),
                "max": float(k.max()),
                "n_ge_cap": int((k >= args.kappa).sum()),
                "frac_ge_cap": float((k >= args.kappa).float().mean()),
            })
        return kl, g

    planner._kl_backward = kb
    torch.manual_seed(0)
    dp.predict_action_dyn_guided(obs_dict)

    n_steps = len(steps)
    all_kl = np.concatenate([np.array([s["mean"]]) for s in steps])  # placeholder
    # recompute a proper aggregate over raw rows is not stored; we have
    # per-step stats only (sufficient for the cap question)
    row_tot = sum(s["n_ge_cap"] for s in steps)
    rows_total = B * n_steps
    out = {
        "dp_ckpt": args.dp_ckpt, "vib_ckpt": args.vib_ckpt,
        "eta": args.eta, "kappa": args.kappa, "B": B, "gst": gst,
        "n_guided_steps": n_steps,
        "rows_total": rows_total,
        "rows_at_or_above_cap": row_tot,
        "row_cap_fraction": row_tot / rows_total if rows_total else 0.0,
        "steps_with_any_cap_row": sum(1 for s in steps if s["n_ge_cap"] > 0),
        "step_cap_fraction": (sum(1 for s in steps if s["n_ge_cap"] > 0)
                              / n_steps if n_steps else 0.0),
        "kl_mean_over_steps": float(np.mean([s["mean"] for s in steps])) if n_steps else 0.0,
        "kl_max_over_steps": float(max(s["max"] for s in steps)) if n_steps else 0.0,
        "per_step": steps,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "per_step"}, indent=1))
    print("---- per-step (t, mean KL, max KL, n>=cap) ----")
    for s in steps:
        print(f"t={s['t']:3d}  mean={s['mean']:.4f}  max={s['max']:.4f}  "
              f"n_ge_cap={s['n_ge_cap']:3d}/{B}")
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"[cap-probe] saved -> {args.out}")


if __name__ == "__main__":
    main()
