#!/usr/bin/env python3
"""thm2_c_calib.py -- C-anchored kappa calibration (seed-233 round-1 ckpt).

User-specified flow (2026-09-22/23), opposite direction to the R-anchored eta.
C definition v2 (user 2026-09-23): C_mean := step-averaged MEAN UNCAPPED KL
across rows -- i.e. the mean over all (guided step x row) of
KL(q(z|s,a) || q(z|s,a^0)), the "跨步均值 KL". (v1 was the row cap-hit
fraction, which measured κ-invariant on this ckpt and could not converge.)

  C_target is measured ONCE on the BASE ckpt (DP-base + dyn-base) at its own
  calibrated dose eta_base and kappa=2.5.

  (2a) probe (eta_i, kappa_{i-1}) -> C_mean_i;
  (2b) kappa_i' = kappa_{i-1} * C_mean_i / C_target;
  (2c) re-probe (eta_i, kappa_i'); if C_mean in [0.9,1.1]*C_target -> done,
       else continue from kappa_i', at most 2 repeats.

Read-only: writes only --out json.
"""
import argparse, json, sys
sys.path.insert(0, ".")


def main():
    import h5py
    import numpy as np
    import torch

    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--core-hdf5", required=True)
    ap.add_argument("--base-dp-ckpt", required=True)
    ap.add_argument("--base-vib-ckpt", required=True)
    ap.add_argument("--base-eta", type=float, required=True)
    ap.add_argument("--round-dp-ckpt", required=True)
    ap.add_argument("--round-vib-ckpt", required=True)
    ap.add_argument("--round-eta", type=float, required=True)
    ap.add_argument("--kappa0", type=float, default=2.5)
    ap.add_argument("--max-repeat", type=int, default=2)
    ap.add_argument("--band", type=float, default=0.1)   # +-10% of C_target
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dev = torch.device("cuda")
    from scout.eval.factories import (load_cfg, make_lpb_dp_factory,
                                      make_scout_vib_factory)
    from scout.eval.rollout import make_action_bridge, make_obs_adapter
    from scout.guidance.entropy_costs import KLCostPlanner

    # ---- frozen core batch (verbatim cfk_rcalib.py / thm2_cap_probe.py) --- #
    cfg0 = load_cfg(args.eval_config)
    view_names = list(cfg0.eval.view_names)
    proprio_keys = list(cfg0.eval.proprio_keys)
    gst = int(cfg0.exploration.get("guidance_start_timestep", 50))
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

    def measure_c(dp_ckpt, vib_ckpt, eta, kappa):
        """One guided denoise on the frozen batch -> C_mean (= mean over all
        (step,row) of the UNCAPPED KL; the guidance cap itself still bites at
        `kappa` -- it shapes the trajectory the KL is measured on). Same x_T
        draw every call (manual_seed(0))."""
        cfg = load_cfg(args.eval_config)
        cfg.vib.ckpt_path = vib_ckpt
        cfg.vib.base_dp_ckpt = dp_ckpt
        dp = make_lpb_dp_factory(dev)(dp_ckpt)
        dp.eval()
        scout_vib = make_scout_vib_factory(cfg, dev)(vib_ckpt)
        scout_vib.eval()
        bridge = make_action_bridge(dp)
        base_adapt = make_obs_adapter(view_names, proprio_keys)

        def obs_adapter(co):
            scaled = {k: (v * 255.0 if k in view_names else v)
                      for k, v in co.items()}
            return base_adapt(scaled)

        planner = KLCostPlanner(scout_vib, bridge=bridge,
                                obs_adapter=obs_adapter, cap=kappa,
                                eta_dimless=False)
        dp.initialize_scout_planner(planner, gst, eta)
        recs = []
        orig_kb = planner._kl_backward

        def kb(trajectory, x0_hat, current_obs=None):
            kl, g = orig_kb(trajectory, x0_hat, current_obs)
            with torch.no_grad():
                recs.append(float(kl.detach().mean()))
            return kl, g

        planner._kl_backward = kb
        torch.manual_seed(0)
        dp.predict_action_dyn_guided(obs_dict)
        C = float(np.mean(recs)) if recs else 0.0
        return C, len(recs)

    # ---- (0) C_target on the base ckpt @ base eta, kappa0 ---------------- #
    C_target, n_steps = measure_c(args.base_dp_ckpt, args.base_vib_ckpt,
                                  args.base_eta, args.kappa0)
    print(f"[C-calib] base C_target = {C_target:.6f} (eta={args.base_eta}, "
          f"kappa={args.kappa0}, {n_steps} steps)")

    # ---- (2a)(2b)(2c) iterate kappa on the round-1 ckpt ------------------- #
    kappa = args.kappa0
    hist = []
    for rep in range(args.max_repeat + 1):
        C, n = measure_c(args.round_dp_ckpt, args.round_vib_ckpt,
                         args.round_eta, kappa)
        hist.append({"stage": f"iter{rep}", "kappa": kappa, "C_mean": C})
        print(f"[C-calib] iter{rep}: kappa={kappa:.6f} C_mean={C:.6f} "
              f"({n} steps)")
        if (1 - args.band) * C_target <= C <= (1 + args.band) * C_target:
            print(f"[C-calib] converged at kappa={kappa:.6f} "
                  f"(C_mean={C:.6f} in [{ (1-args.band)*C_target:.6f}, "
                  f"{(1+args.band)*C_target:.6f}])")
            break
        if rep == args.max_repeat:
            print(f"[C-calib] max-repeat reached, last kappa={kappa:.6f} "
                  f"(C_mean={C:.6f})")
            break
        # ratio INVERTED (user-confirmed 2026-09-23): C (mean KL) is
        # INCREASING in kappa, so "C below target = cap too tight = ENLARGE
        # kappa" -> kappa' = kappa * C_target/C_mean. The literal v1 ratio
        # kappa*C_mean/C_target was designed for the (decreasing) hit-rate
        # metric and diverges for this one (2.5 -> 0.845 -> 0.110, C 3.35
        # -> 1.29 -> 0.25, empirically confirmed).
        kappa = kappa * C_target / C
        print(f"[C-calib]   -> kappa' = {kappa:.6f}")

    out = {
        "C_target": C_target,
        "C_target_base": {"dp": args.base_dp_ckpt, "vib": args.base_vib_ckpt,
                          "eta": args.base_eta, "kappa": args.kappa0},
        "kappa": kappa,
        "eta": args.round_eta,
        "band_abs": [(1 - args.band) * C_target, (1 + args.band) * C_target],
        "converged": bool(
            (1 - args.band) * C_target <= hist[-1]["C_mean"]
            <= (1 + args.band) * C_target),
        "history": hist,
        "rule": ("C_mean = mean over (step,row) of UNCAPPED KL (跨步均值KL); "
                 "kappa' = kappa * C_target/C_mean (inverted); "
                 "C_target = base C @ kappa0"),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"[C-calib] saved -> {args.out}")


if __name__ == "__main__":
    main()
