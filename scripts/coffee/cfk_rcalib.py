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
  python scripts/coffee/cfk_rcalib.py \
      --eval-config configs/eval_coffee_entropy.yaml \
      --dp-ckpt <DP ckpt this round> --vib-ckpt <dyn ckpt this round> \
      --core-hdf5 <core hdf5> \
      --eta-prev 5.6 --kappa-prev 2.5 \
      [--target 0.01 --band-lo 0.009 --band-hi 0.011 --max-repeat 2] \
      --out <json>
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
    ap.add_argument("--eta-prev", type=float, required=True)
    ap.add_argument("--kappa-prev", type=float, required=True)
    ap.add_argument("--target", type=float, default=0.01)
    ap.add_argument("--band-lo", type=float, default=0.009)
    ap.add_argument("--band-hi", type=float, default=0.011)
    ap.add_argument("--max-repeat", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--out", required=True)
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

    # ---- (1) frozen core batch (verbatim r_probe_rollout.py) -------------- #
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

    with h5py.File(args.core_hdf5, "r") as f:
        vals = [f[f"data/{d}/abs_actions"][:] for d in f["data"].keys()]
    meanabs_a_raw = float(np.abs(np.concatenate(vals, 0)).mean())

    # ---- policy + planner (verbatim rollout_pipeline._attach_planner) ----- #
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
                            cap=args.kappa_prev, eta_dimless=False)
    dp.initialize_scout_planner(planner, gst, args.eta_prev)

    def eval_rmean(eta):
        """One guided sampling pass at dose eta -> R_mean (per-step mean of
        eta*ns*meanabs|g| divided by meanabs|a_core|). Same x_T draw each
        call (manual_seed(0)) so successive etas are compared paired."""
        dp.guidance_scale = eta
        planner.cap = args.kappa_prev
        recs = []
        orig_step = planner.guided_step

        def wrapped(traj, x0, obs, noise_scale):
            out = orig_step(traj, x0, obs, noise_scale)
            cond_grad, disp, p2, _rl = out
            with torch.no_grad():
                recs.append((float(noise_scale),
                             float(cond_grad.abs().mean())))
            return out

        planner.guided_step = wrapped
        try:
            torch.manual_seed(0)
            dp.predict_action_dyn_guided(obs_dict)
        finally:
            planner.guided_step = orig_step
        inj = np.array([eta * ns * mag for ns, mag in recs], dtype=float)
        return float(inj.mean() / meanabs_a_raw)

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
