"""Kappa from independent DP posterior KL median, followed by R calibration.

Eight unguided draws per core observation. Pool the directed
KL(final_i || anchor_j), i != j, across draws and observations; take p50.
This is a median, not the mean C used by kappa_c.
"""
import argparse
import json
import math
from pathlib import Path

from .core import (ROOT, IDENTITY_KEYS, TASKS, check_gpu, prepare_core,
                   load_model_pair, measure_guidance)


def diagonal_kl(mu, lv, ref_mu, ref_lv):
    """Directed diagonal-Gaussian KL, summed over latent dimensions."""
    import numpy as np

    mu, lv, ref_mu, ref_lv = [np.asarray(x, dtype=np.float64)
                               for x in (mu, lv, ref_mu, ref_lv)]
    difference = lv - ref_lv
    values = .5 * ((mu - ref_mu) ** 2 * np.exp(-ref_lv)
                   + np.expm1(difference) - difference).sum(-1)
    if not np.isfinite(values).all() or (values < -1e-10).any():
        raise ValueError("Invalid posterior divergence")
    return values


def posterior_kl_arrays(fm, fl, am, al):
    """Arrays [ordered independent pair, observation]; never pair different states."""
    import numpy as np

    arrays = [np.asarray(x, dtype=np.float64) for x in (fm, fl, am, al)]
    if arrays[0].ndim != 3 or any(a.shape != arrays[0].shape for a in arrays):
        raise ValueError("Expected matching [draw, observation, latent] posterior arrays")
    fm, fl, am, al = arrays
    samples = len(fm)
    if samples < 2:
        raise ValueError("At least two independent DP draws are required")
    pairs = [(i, j) for i in range(samples) for j in range(samples) if i != j]
    final_to_final = np.stack([diagonal_kl(fm[i], fl[i], fm[j], fl[j]) for i, j in pairs])
    final_to_anchor = np.stack([diagonal_kl(fm[i], fl[i], am[j], al[j]) for i, j in pairs])
    same_draw = np.stack([diagonal_kl(fm[i], fl[i], am[i], al[i]) for i in range(samples)])
    return final_to_final, final_to_anchor, same_draw


def posterior_diversity(directory, source, dp, vib, bridge, obs_adapter, obs, gst, samples):
    """Compare independent DP draws at the final clean estimate and intent anchor."""
    import numpy as np
    import torch
    from scout.guidance.entropy_costs import KLCostPlanner

    output = directory / "dp_diversity"
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    output.mkdir(exist_ok=True)
    last = []

    def capture(module, inputs, values):
        last[:] = [value.detach().clone() for value in values]

    handle = vib.vib_enc.register_forward_hook(capture)
    final_mu, final_lv, anchor_mu, anchor_lv = [], [], [], []
    try:
        for seed in range(samples):
            planner = KLCostPlanner(vib, bridge=bridge, obs_adapter=obs_adapter,
                                    cap=2.5, eta_dimless=False)
            dp.initialize_scout_planner(planner, gst, 0.0)
            last.clear()
            torch.manual_seed(seed)
            dp.predict_action_dyn_guided(obs)
            if len(last) != 2:
                raise ValueError("Expected posterior mean and log variance")
            final_mu.append(last[0].double().cpu().numpy())
            final_lv.append(last[1].double().cpu().numpy())
            anchor_mu.append(torch.stack(planner._base_mu).double().cpu().numpy())
            anchor_lv.append(torch.stack(planner._base_lv).double().cpu().numpy())
            print(f"[{source['task']}] unguided draw {seed + 1}/{samples}", flush=True)
    finally:
        handle.remove()
    fm, fl, am, al = [np.stack(values) for values in (final_mu, final_lv, anchor_mu, anchor_lv)]

    final_to_final, final_to_anchor, same_draw = posterior_kl_arrays(fm, fl, am, al)
    reference = directory / "diagnostics/natural.npz"
    if reference.exists():
        with np.load(reference) as data:
            expected = data["kl"][-1]
        if not np.allclose(same_draw[0], expected, rtol=1e-3, atol=2e-5):
            raise ValueError("Seed-zero final KL does not reproduce the existing natural probe")

    def describe(values):
        return dict(mean=float(values.mean()),
                    quantiles=dict(zip(["p10", "p25", "p50", "p75", "p90", "p95"],
                                       np.quantile(values, [.1, .25, .5, .75, .9, .95]).tolist())),
                    median_state_mean=float(np.median(values.mean(0))))

    summary = {key: source[key] for key in ("task", "dp_ckpt", "vib_ckpt", "core_hdf5")}
    summary.update(samples=samples, rng_seeds=list(range(samples)), B=len(fm[0]),
                   guidance_start=gst, eta=0, cost_sum_over_latent_dimensions=True,
                   final_to_final=describe(final_to_final),
                   independent_final_to_anchor=describe(final_to_anchor),
                   same_draw_final_to_anchor=describe(same_draw),
                   definition="Posterior at the last unguided clean estimate; anchor is the first guided-step estimate")
    np.savez_compressed(output / "samples.npz", final_mu=fm, final_logvar=fl,
                        anchor_mu=am, anchor_logvar=al, final_to_final=final_to_final,
                        independent_final_to_anchor=final_to_anchor, same_draw_final_to_anchor=same_draw)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[{source['task']}] DP divergence: {summary['final_to_final']}; "
          f"cross-anchor: {summary['independent_final_to_anchor']}", flush=True)

    return summary


def match_r(measure, eta, target_r, *, band=.1, max_probes=9):
    """The research dose loop: accept the initial point, or make <=8 updates.

    Return the last measured point even on nonconvergence so it can be saved
    for diagnosis. Callers must check the measured R before using it.
    target_r=None is a single diagnostic measurement (no calibration).
    """
    if max_probes < 1 or not 0 <= band < 1:
        raise ValueError("Require max_probes >= 1 and 0 <= band < 1")
    if target_r is not None and (not math.isfinite(target_r) or target_r <= 0):
        raise ValueError("R target must be finite and positive")
    history = []
    for iteration in range(max_probes):
        result = measure(eta)
        measured_r = result["R_mean"]
        history.append({"eta": eta, "R_mean": measured_r})
        if target_r is None or (1 - band) * target_r <= measured_r <= (1 + band) * target_r:
            break
        if not math.isfinite(measured_r) or measured_r <= 0:
            raise ValueError(f"Invalid dose reading: {measured_r}")
        print(f"[R-calib] probe {iteration}: eta={eta:.6g} R={measured_r:.6g}", flush=True)
        if iteration == max_probes - 1:
            break
        eta *= target_r / measured_r
    return eta, result, history


def calibrate_dose(source, dp, vib, bridge, obs_adapter, obs, gst, scale,
                   cap, eta, target_r=.01, *, band=.1, max_probes=9):
    """Fix kappa and measure/calibrate R using the original research loop."""
    import numpy as np
    from scout.guidance.entropy_costs import KLCostPlanner

    planner = KLCostPlanner(vib, bridge=bridge, obs_adapter=obs_adapter,
                            cap=cap, eta_dimless=False)
    dp.initialize_scout_planner(planner, gst, eta)

    def measure(dose):
        return measure_guidance(dp, planner, obs, dose, scale, record_noisy=True)

    eta, result, history = match_r(measure, eta, target_r, band=band, max_probes=max_probes)
    kl, inj, noisy = result["kl"], result["injection"], result["noisy_scale"]
    measured_r = result["R_mean"]
    stats = {key: source[key] for key in IDENTITY_KEYS}
    stats.update(
        eta=eta, kappa=cap, B=len(next(iter(obs.values()))), n_steps=len(kl), rng_seed=0,
        meanabs_a_raw=scale, R_mean=measured_r, R_target=target_r,
        R_converged=target_r is None or (1 - band) * target_r <= measured_r <= (1 + band) * target_r,
        calibration_history=history,
        R_noisy_mean=float(np.mean(inj / np.maximum(noisy, 1e-12))),
        C_mean=float(kl.mean()), cap_fraction=float((kl > cap).mean()),
        kl_quantiles=dict(zip(["p25", "p50", "p75", "p90", "p95", "p99"],
                             np.quantile(kl, [.25, .5, .75, .9, .95, .99]).tolist())),
        final_step_kl_quantiles=dict(zip(["p25", "p50", "p75", "p90", "p95", "p99"],
                                        np.quantile(kl[-1], [.25, .5, .75, .9, .95, .99]).tolist())),
        per_step=[{"index": i, "C_mean": float(k.mean()), "cap_fraction": float((k > cap).mean()),
                   "meanabs_injection": float(inj[i]), "meanabs_noisy": float(noisy[i])}
                  for i, k in enumerate(kl)])
    return stats, dict(kl=kl, injection=inj, noisy_scale=noisy)


def validate_calibration(calibration, metadata, cap, target_r=.01, band=.1):
    """Reject mismatched cached checkpoints or a dose outside the measured band."""
    for key in IDENTITY_KEYS:
        if calibration[key] != metadata[key]:
            raise ValueError(f"Dose calibration used a different {key}")
    if (calibration["kappa"] != cap or not calibration["R_converged"]
            or not math.isfinite(cap) or cap <= 0
            or calibration.get("R_target") != target_r
            or not (1 - band) * target_r <= calibration["R_mean"] <= (1 + band) * target_r
            or not math.isfinite(calibration["eta"]) or calibration["eta"] <= 0):
        raise ValueError("Dose calibration did not converge at predicted median KL")


def calibrate(args):
    """Calibrate from direct checkpoints or the existing research directory."""
    import numpy as np

    source = args.source.resolve() if args.source else None
    if source is not None:
        source.relative_to(ROOT / "experiments")
        args.out.resolve().relative_to(source)
        path = source / "checkpoints.json"
        if not path.exists():
            path = source / "diagnostics/natural.json"
        metadata = json.loads(path.read_text())
        directory = source
        eta = metadata.get("eta_initial")
        if eta is None and (source / "eta.json").exists():
            eta = json.loads((source / "eta.json").read_text())["eta"]
    else:
        metadata = dict(task=args.task, dp_ckpt=str(args.dp_ckpt.resolve()),
                        vib_ckpt=str(args.vib_ckpt.resolve()), core_hdf5=str(args.core_hdf5.resolve()))
        for key in ("dp_ckpt", "vib_ckpt", "core_hdf5"):
            if not Path(metadata[key]).is_file():
                raise FileNotFoundError(metadata[key])
        directory = args.out.with_suffix(".artifacts")
        if directory.exists():
            raise FileExistsError(directory)
        eta = None
    eta = args.eta_initial if args.eta_initial is not None else eta
    if eta is None or not math.isfinite(eta) or eta <= 0:
        raise ValueError("Provide a positive --eta-initial (or source eta_initial/eta.json)")
    if args.out.exists():
        raise FileExistsError(args.out)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    directory.mkdir(parents=True, exist_ok=True)
    models = None

    def setup():
        nonlocal models
        if models is None:
            check_gpu(args.gpu, args.gpu_uuid)
            cfg, obs, scale = prepare_core(
                args.eval_config or ROOT / f"configs/{metadata['task']}/eval.yaml",
                metadata["core_hdf5"])
            gst = int(cfg.exploration.get("guidance_start_timestep", 50))
            models = (*load_model_pair(cfg, metadata["dp_ckpt"], metadata["vib_ckpt"]), obs, gst, scale)
            if len(next(iter(obs.values()))) != 128:
                raise ValueError("KL median protocol requires 128 core observations")
        return models

    diversity_path = directory / "dp_diversity/summary.json"
    if not diversity_path.exists():
        dp, vib, bridge, obs_adapter, obs, gst, scale = setup()
        posterior_diversity(directory, metadata, dp, vib, bridge, obs_adapter, obs, gst, args.samples)
    diversity = json.loads(diversity_path.read_text())
    for key in IDENTITY_KEYS:
        if diversity[key] != metadata[key]:
            raise ValueError(f"DP diversity used a different {key}")
    if diversity["samples"] != args.samples or diversity["B"] != 128:
        raise ValueError("Different DP draw count or core batch size")
    cap = diversity["independent_final_to_anchor"]["quantiles"]["p50"]
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("Median KL must be finite and positive")
    reference = dict(kappa=cap, statistic="median independent final-to-anchor posterior KL",
                     source=str(diversity_path), R_target=args.target_r, samples=args.samples)
    reference_path = directory / "kl_median_reference.json"
    if reference_path.exists() and json.loads(reference_path.read_text()) != reference:
        raise ValueError("Existing median reference has different inputs")
    reference_path.write_text(json.dumps(reference, indent=2) + "\n")
    calibration_path = directory / f"diagnostics_rmatched/k{cap:g}.json"
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())
        # Explicit overrides must not silently reuse a different dose trajectory.
        history = calibration.get("calibration_history", [])
        if args.eta_initial is not None and (not history or history[0]["eta"] != eta):
            raise ValueError("Existing calibration used a different initial eta")
    else:
        dp, vib, bridge, obs_adapter, obs, gst, scale = setup()
        calibration, arrays = calibrate_dose(metadata, dp, vib, bridge, obs_adapter,
                                            obs, gst, scale, cap, eta, args.target_r)
        calibration_path.parent.mkdir(exist_ok=True)
        np.savez_compressed(calibration_path.with_suffix(".npz"), **arrays)
        calibration_path.write_text(json.dumps(calibration, indent=2) + "\n")
    out = dict(calibration, method="dp_kl_median", statistic=reference["statistic"],
               samples=args.samples, median_reference=str(reference_path),
               diversity_summary=str(diversity_path), calibration_path=str(calibration_path))
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    validate_calibration(calibration, metadata, cap, args.target_r)
    print(json.dumps({key: out[key] for key in ("task", "kappa", "eta", "R_mean", "R_converged")}, indent=2))
    print(f"Core calibration saved at {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="existing task research directory")
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--dp-ckpt", type=Path)
    parser.add_argument("--vib-ckpt", type=Path)
    parser.add_argument("--core-hdf5", type=Path)
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--eta-initial", type=float)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--target-r", type=float, default=.01)
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    direct = [args.task, args.dp_ckpt, args.vib_ckpt, args.core_hdf5]
    if (args.source and any(x is not None for x in direct)) or (not args.source and not all(direct)):
        parser.error("use --source OR all of --task --dp-ckpt --vib-ckpt --core-hdf5")
    if args.source and args.eval_config:
        parser.error("--eval-config is for direct checkpoint mode")
    if not args.source and args.eta_initial is None:
        parser.error("direct checkpoint mode requires --eta-initial")
    if args.samples < 2 or not math.isfinite(args.target_r) or args.target_r <= 0:
        parser.error("samples >= 2 and finite positive target-r required")
    if args.eta_initial is not None and (not math.isfinite(args.eta_initial) or args.eta_initial <= 0):
        parser.error("eta-initial must be finite and positive")
    calibrate(args)


if __name__ == "__main__":
    main()
