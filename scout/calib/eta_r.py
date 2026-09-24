"""R-anchored per-round (eta, kappa) calibration for COFFEE-MG-p4
(user-specified flow, 2026-09-22).

At the START of each ATY round, with the round's resolved DP + VIB ckpts
(walk-back applied by the round script BEFORE this runs):

  (1) frozen core batch, B=128 states (identical construction to
      r_probe_rollout.py: hdf5-read, images /255 CHW, obs stack (t-1,t));
  (2) measure R_mean with the PREVIOUS round's dose (eta_prev, kappa_prev)
      -- R_mean = mean over guided denoise steps of
      eta*sqrt(1-alphabar_t)*meanabs|grad| / meanabs|a_core|;
  (3) eta_i = eta_prev * target / R_mean    (target = 0.01);
  (4) re-evaluate R_mean at eta_i; success iff 0.009 <= R_mean <= 0.011,
      else repeat (2)(3) with the updated eta, at most 2 repeats
      (i.e. up to 3 updates / 4 evaluations).

kappa is CARRIED OVER unchanged (kappa_prev; nothing in the flow updates
it -- kappa_0 = 2.5 from the p1 grid verdict).

Deterministic: fixed batch indices, torch.manual_seed(0) before every
evaluation (same x_T draw across eta values -- the comparison is paired).

usage (server, .venv_mg):
  python -m scout.calib.eta_r \
      --eval-config configs/eval_coffee_entropy.yaml \
      --dp-ckpt <DP ckpt this round> --vib-ckpt <dyn ckpt this round> \
      --core-hdf5 <core hdf5> \
      --eta-prev 5.6 --kappa-prev 2.5 \
      [--target 0.01 --band-lo 0.009 --band-hi 0.011 --max-repeat 2] \
      --out <json>
"""
import argparse
import json
import sys

from .core import prepare_core, load_model_pair, measure_guidance


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-config", required=True)
    ap.add_argument("--dp-ckpt", required=True)
    ap.add_argument("--vib-ckpt", required=True)
    ap.add_argument("--core-hdf5", required=True)
    ap.add_argument("--eta-prev", type=float, required=True)
    ap.add_argument("--kappa-prev", type=float, required=True)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--band-lo", type=float, default=0.009)
    ap.add_argument("--band-hi", type=float, default=0.011)
    ap.add_argument("--max-repeat", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from scout.guidance.entropy_costs import KLCostPlanner

    cfg, obs_dict, meanabs_a_raw = prepare_core(
        args.eval_config, args.core_hdf5, args.batch_size, scale_order="stored")
    B = len(next(iter(obs_dict.values())))
    gst = int(cfg.exploration.get("guidance_start_timestep", 50))
    dp, scout_vib, bridge, obs_adapter = load_model_pair(cfg, args.dp_ckpt, args.vib_ckpt)

    planner = KLCostPlanner(scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                            cap=args.kappa_prev, eta_dimless=False)
    dp.initialize_scout_planner(planner, gst, args.eta_prev)

    def eval_rmean(eta):
        """One guided sampling pass at dose eta -> R_mean (per-step mean of
        eta*ns*meanabs|g| divided by meanabs|a_core|). Same x_T draw each
        call (manual_seed(0)) so successive etas are compared paired."""
        planner.cap = args.kappa_prev
        return measure_guidance(dp, planner, obs_dict, eta, meanabs_a_raw,
                                record_kl=False)["R_mean"]

    # ---- (2)(3)(4) iterative re-anchoring ---------------------------------- #
    eta = args.eta_prev
    R = eval_rmean(eta)                       # (2)
    hist = [{"stage": "prev", "eta": eta, "R_mean": R}]
    if R <= 0:
        print("[rcalib] FATAL: R_mean == 0 at eta_prev (dead dyn?)",
              file=sys.stderr)
        sys.exit(1)
    converged = False
    n_updates = 0
    for rep in range(args.max_repeat + 1):
        eta = eta * args.target / R           # (3)
        n_updates += 1
        R = eval_rmean(eta)                   # (4)
        hist.append({"stage": f"update{n_updates}", "eta": eta,
                     "R_mean": R})
        print(f"[rcalib] update {n_updates}: eta={eta:.6g} R_mean={R:.6g}")
        if args.band_lo <= R <= args.band_hi:
            converged = True
            break

    out = {
        "method": "rcalib", "B": B, "gst": gst,
        "eta": eta, "kappa": args.kappa_prev,
        "target": args.target, "band": [args.band_lo, args.band_hi],
        "converged": converged, "n_updates": n_updates,
        "eta_prev": args.eta_prev, "R_prev": hist[0]["R_mean"],
        "R_mean": R, "meanabs_a_raw": meanabs_a_raw,
        "history": hist,
        "rule": ("eta_i = eta_{i-1} * 0.01 / R_mean(eta_{i-1}); converge "
                 "iff R_mean in [0.009, 0.011]; max 2 repeats; kappa "
                 "carried over unchanged"),
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"[rcalib] done: eta={eta:.6g} kappa={args.kappa_prev} "
          f"R_mean={R:.6g} converged={converged} updates={n_updates}")
    print(f"[rcalib] saved -> {args.out}")


if __name__ == "__main__":
    main()
