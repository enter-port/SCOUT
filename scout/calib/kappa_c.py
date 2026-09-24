#!/usr/bin/env python3
"""Calibrate kappa by matching mean uncapped KL to a fixed base reference.

User-specified C calibration flow (2026-09-22/23).
C definition v2 (user 2026-09-23): C_mean := step-averaged MEAN UNCAPPED KL
across rows -- i.e. the mean over all (guided step x row) of
KL(q(z|s,a) || q(z|s,a^0)), the "跨步均值 KL". (v1 was the row cap-hit
fraction, which measured κ-invariant on this ckpt and could not converge.)

  C_target is measured on the BASE ckpt (DP-base + dyn-base) at its own
  calibrated dose eta_base and base_kappa=2.5. The round's initial kappa
  can change without moving this target.

  (2a) probe (eta_i, kappa_{i-1}) -> C_mean_i;
  (2b) kappa_i' = kappa_{i-1} * C_target / C_mean_i;
  (2c) re-probe (eta_i, kappa_i'); if C_mean in [0.9,1.1]*C_target -> done,
       else continue from kappa_i', at most 2 repeats.

Optional --solver bracket bounds the search and bisects a measured bracket
instead of repeating ratio updates. --require-converged rejects a failed
calibration after saving its diagnostic JSON. Neither option changes the
definition of C or establishes that matching C improves rollout outcomes.

Read-only: writes only --out json.
"""
import argparse
import json
import math

from .core import prepare_core, load_model_pair, measure_guidance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--core-hdf5", required=True)
    ap.add_argument("--base-dp-ckpt", required=True)
    ap.add_argument("--base-vib-ckpt", required=True)
    ap.add_argument("--base-eta", type=float, required=True)
    ap.add_argument("--round-dp-ckpt", required=True)
    ap.add_argument("--round-vib-ckpt", required=True)
    ap.add_argument("--round-eta", type=float, required=True)
    ap.add_argument("--base-kappa", type=float, default=2.5,
                    help="fixed base-model C target cap")
    ap.add_argument("--kappa0", type=float, default=2.5,
                    help="initial cap for the round model")
    ap.add_argument("--max-repeat", type=int, default=2)
    ap.add_argument("--solver", choices=["ratio", "bracket"], default="ratio")
    ap.add_argument("--max-probes", type=int, default=12,
                    help="evaluation budget for the bracket solver")
    ap.add_argument("--kappa-min", type=float, default=1e-3)
    ap.add_argument("--kappa-max", type=float, default=100.0)
    ap.add_argument("--require-converged", action="store_true")
    ap.add_argument("--band", type=float, default=0.1)   # +-10% of C_target
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    if any(not math.isfinite(k) or k <= 0 for k in [args.base_kappa, args.kappa0]):
        ap.error("base-kappa and kappa0 must be finite and positive")
    if not 0 <= args.band < 1 or args.max_repeat < 0 or args.batch_size < 1:
        ap.error("require 0 <= band < 1, max-repeat >= 0, and batch-size >= 1")

    from scout.eval.factories import load_cfg
    from scout.guidance.entropy_costs import KLCostPlanner

    cfg0, obs_dict, _ = prepare_core(args.eval_config, args.core_hdf5,
                                    args.batch_size, scale_order=None)
    gst = int(cfg0.exploration.get("guidance_start_timestep", 50))

    def measure_c(dp_ckpt, vib_ckpt, eta, kappa):
        # Keep the original per-probe model reload and Torch-per-step reduction.
        cfg = load_cfg(args.eval_config)
        dp, vib, bridge, obs_adapter = load_model_pair(cfg, dp_ckpt, vib_ckpt)
        planner = KLCostPlanner(vib, bridge=bridge, obs_adapter=obs_adapter,
                                cap=kappa, eta_dimless=False)
        dp.initialize_scout_planner(planner, gst, eta)
        measurement = measure_guidance(dp, planner, obs_dict, eta,
                                       record_injection=False, step_mean_c=True)
        return measurement["C_mean"], measurement["n_steps"]

    # ---- (0) fixed C_target on the base ckpt ----------------------------- #
    C_target, n_steps = measure_c(args.base_dp_ckpt, args.base_vib_ckpt,
                                  args.base_eta, args.base_kappa)
    print(f"[C-calib] base C_target = {C_target:.6f} (eta={args.base_eta}, "
          f"kappa={args.base_kappa}, {n_steps} steps)")
    if not math.isfinite(C_target) or C_target <= 0:
        raise ValueError("base C_target must be positive")

    # ---- (2a)(2b)(2c) iterate kappa on the round-1 ckpt ------------------- #
    kappa = args.kappa0
    hist = []
    if args.solver == "bracket":
        from .search import solve_kappa

        def probe_round(cap):
            value, steps = measure_c(args.round_dp_ckpt, args.round_vib_ckpt,
                                     args.round_eta, cap)
            print(f"[C-calib] bracket probe: kappa={cap:.6f} C_mean={value:.6f} ({steps} steps)", flush=True)
            return value

        result = solve_kappa(probe_round, C_target, kappa, band=args.band,
                             max_probes=args.max_probes, lower=args.kappa_min,
                             upper=args.kappa_max)
        kappa, C, hist = result["kappa"], result["C_mean"], result["history"]
        selected_index = result["selected_index"]
        termination_reason = result["termination_reason"]
        for index, item in enumerate(hist):
            item["stage"] = f"probe{index}"
    else:
        termination_reason = "max_repeat"
        for rep in range(args.max_repeat + 1):
            C, n = measure_c(args.round_dp_ckpt, args.round_vib_ckpt,
                             args.round_eta, kappa)
            if not math.isfinite(C) or C <= 0:
                raise ValueError("round C_mean must be finite and positive")
            hist.append({"stage": f"iter{rep}", "kappa": kappa, "C_mean": C})
            print(f"[C-calib] iter{rep}: kappa={kappa:.6f} C_mean={C:.6f} ({n} steps)")
            if (1 - args.band) * C_target <= C <= (1 + args.band) * C_target:
                termination_reason = "target_band"
                break
            if rep == args.max_repeat:
                print(f"[C-calib] max-repeat reached, last kappa={kappa:.6f} (C_mean={C:.6f})")
                break
            # C generally increases with kappa, so increase kappa when C
            # is below target. This legacy update can overshoot repeatedly.
            kappa = kappa * C_target / C
            print(f"[C-calib]   -> kappa' = {kappa:.6f}")
        selected_index = len(hist) - 1

    out = {
        "C_target": C_target,
        "C_target_base": {"dp": args.base_dp_ckpt, "vib": args.base_vib_ckpt,
                          "eta": args.base_eta, "kappa": args.base_kappa},
        "kappa": kappa,
        "eta": args.round_eta,
        "band_abs": [(1 - args.band) * C_target, (1 + args.band) * C_target],
        "converged": bool((1 - args.band) * C_target <= C <= (1 + args.band) * C_target),
        "C_mean": C,
        "solver": args.solver,
        "selected_index": selected_index,
        "termination_reason": termination_reason,
        "history": hist,
        "rule": ("C_mean = mean over (step,row) of UNCAPPED KL (跨步均值KL); "
                 + ("kappa' = kappa * C_target/C_mean; " if args.solver == "ratio"
                    else "bounded bracketing and log-kappa bisection; ") +
                 "C_target = base C @ base_kappa"),
    }
    print(json.dumps(out, indent=1))
    if args.out:
        with open(args.out, "w") as stream:
            json.dump(out, stream, indent=1)
        print(f"[C-calib] saved -> {args.out}")
    if args.require_converged and not out["converged"]:
        raise SystemExit("C calibration did not converge; inspect the saved measurements")


if __name__ == "__main__":
    main()
