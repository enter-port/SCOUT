#!/usr/bin/env python3
"""Core calibration of R/P, joint R/C, or the original sequential R then C.

Conditional invariance: f -> c*f, kappa -> c*kappa, eta -> eta/c
leaves eta*grad(min(f,kappa)) unchanged. Whether one P transfers across
different learned cost shapes must be established by environment rollouts.
"""
import argparse
import json
import math
from pathlib import Path

from .core import ROOT, check_gpu, prepare_core, load_model_pair, measure_guidance
from .search import solve_kappa


def calibrate(args):
    source = args.source.resolve() if args.source else None
    if source is not None:
        source.relative_to(ROOT / "experiments")
        args.out.resolve().relative_to(source)
        metadata_path = source / "checkpoints.json"
        if not metadata_path.exists():
            metadata_path = source / "diagnostics/natural.json"
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = dict(task=args.task, dp_ckpt=str(args.dp_ckpt.resolve()),
                        vib_ckpt=str(args.vib_ckpt.resolve()), core_hdf5=str(args.core_hdf5.resolve()))
        for key in ("dp_ckpt", "vib_ckpt", "core_hdf5"):
            if not Path(metadata[key]).is_file():
                raise FileNotFoundError(metadata[key])
    task = metadata["task"]
    keys = ("task", "dp_ckpt", "vib_ckpt", "core_hdf5")
    if args.out.exists():
        raise FileExistsError(args.out)
    reference = None
    if args.c_reference:
        reference = json.loads(args.c_reference.read_text())
        if reference["task"] != task or reference["core_hdf5"] != metadata["core_hdf5"]:
            raise ValueError("C reference must use the same task and core")
        # Older measured diagnostics predate the convergence flag. Their
        # actual finite R measurement still establishes membership in the band.
        if (reference.get("R_converged") is False
                or not math.isfinite(reference["R_mean"])
                or abs(reference["R_mean"] / args.target_r - 1) > args.band):
            raise ValueError("C reference must itself satisfy the R band")
        if not math.isfinite(reference["C_mean"]) or reference["C_mean"] <= 0:
            raise ValueError("C reference must have positive finite C")
    # Reuse a core measurement only when both independently stated bands
    # hold. Select by proximity to P, never by rollout success outcomes.
    reusable = []
    for pattern in ("diagnostics/k*.json", "diagnostics_rmatched/k*.json", "calibration_k*.json"):
        for path in source.glob(pattern) if source is not None else []:
            d = json.loads(path.read_text())
            if any(d.get(key) != metadata[key] for key in keys):
                continue
            p = d["eta"] * d["kappa"]
            if (args.potential_cap is not None and abs(d["R_mean"] / args.target_r - 1) <= args.band
                    and abs(p / args.potential_cap - 1) <= args.band):
                reusable.append((abs(math.log(p / args.potential_cap)), str(path), d))
    if reusable and not args.no_reuse:
        _, path, d = min(reusable, key=lambda item: (item[0], item[1]))
        out = {key: metadata[key] for key in keys}
        out.update(eta=d["eta"], kappa=d["kappa"], R_mean=d["R_mean"],
                   R_target=args.target_r, R_converged=True,
                   potential_cap=d["eta"] * d["kappa"], potential_target=args.potential_cap,
                   band=args.band, reused_core_measurement=path,
                   termination_reason="existing_pair_within_both_bands", history=[])
        args.out.write_text(json.dumps(out, indent=2) + "\n")
        print(json.dumps(out, indent=2), flush=True)
        return

    if args.reuse_only:
        raise SystemExit("No existing core pair satisfies both bands")
    check_gpu(args.gpu, args.gpu_uuid)
    from scout.guidance.entropy_costs import KLCostPlanner

    cfg, obs, scale = prepare_core(
        args.eval_config or ROOT / f"configs/eval_{task}_entropy.yaml", metadata["core_hdf5"])
    gst = int(cfg.exploration.get("guidance_start_timestep", 50))
    dp, vib, bridge, obs_adapter = load_model_pair(cfg, metadata["dp_ckpt"], metadata["vib_ckpt"])

    measured = {}
    primed_eta = None
    priming_record = None

    def measure_pair(cap, eta):
        planner = KLCostPlanner(vib, bridge=bridge, obs_adapter=obs_adapter,
                                cap=cap, eta_dimless=False)
        dp.initialize_scout_planner(planner, gst, eta)
        result = measure_guidance(dp, planner, obs, eta, scale)
        record = dict(kappa=cap, eta=eta, R_mean=result["R_mean"], C_mean=result["C_mean"],
                      cap_fraction=float((result["kl"] > cap).mean()))
        print(f"[joint-calib] {record}", flush=True)
        return record

    all_measurements = []

    def measure(cap):
        nonlocal primed_eta, priming_record
        if reference is None:
            record = measure_pair(cap, args.potential_cap / cap)
        elif args.sequential_c and primed_eta is not None:
            record = measure_pair(cap, primed_eta)
            all_measurements.append(record)
        else:
            # Start from the closest already measured cap's calibrated eta.
            # Only core statistics enter either level of this calibration.
            eta = (min(measured.values(), key=lambda d: abs(math.log(d["kappa"] / cap)))["eta"]
                   if measured else args.eta_initial if args.eta_initial is not None else reference["eta"])
            inner = []
            for _ in range(args.eta_max_probes):
                record = measure_pair(cap, eta)
                inner.append(record)
                if abs(record["R_mean"] / args.target_r - 1) <= args.band:
                    break
                eta = max(1e-5, min(1e4, eta * args.target_r / max(record["R_mean"], 1e-12)))
            record = min(inner, key=lambda d: abs(d["R_mean"] / args.target_r - 1))
            all_measurements.extend(inner)
            if abs(record["R_mean"] / args.target_r - 1) > args.band:
                failure = {key: metadata[key] for key in keys}
                failure.update(record, R_target=args.target_r, R_converged=False,
                               C_target=reference["C_mean"], C_converged=False,
                               termination_reason="inner_R_not_converged", history=all_measurements)
                args.out.write_text(json.dumps(failure, indent=2) + "\n")
                raise SystemExit("Inner R calibration failed; no C-matched rollout is authorized by this result")
            if args.sequential_c:
                primed_eta = record["eta"]
                priming_record = dict(record)
        measured[cap] = record
        return record["R_mean"] if reference is None else record["C_mean"]

    target = args.target_r if reference is None else reference["C_mean"]
    result = solve_kappa(measure, target, args.initial_kappa,
                         band=args.band, max_probes=args.max_probes,
                         lower=args.kappa_min, upper=args.kappa_max, direction="unknown")
    selected = measured[result["kappa"]]
    out = {key: metadata[key] for key in keys}
    out.update(selected, R_target=args.target_r, R_converged=result["converged"],
               potential_cap=selected["eta"] * selected["kappa"], potential_target=args.potential_cap,
               band=args.band, B=len(next(iter(obs.values()))), guided_steps=gst, rng_seed=0,
               meanabs_a_raw=scale, history=list(measured.values()),
               termination_reason=result["termination_reason"])
    if reference is not None:
        out.update(C_target=target, C_converged=result["converged"],
                   R_converged=abs(selected["R_mean"] / args.target_r - 1) <= args.band,
                   joint_converged=result["converged"],
                   C_reference=str(args.c_reference.resolve()), reference=reference,
                   inner_measurements=all_measurements)
        if args.sequential_c:
            # R is a priming target in the existing two-stage scheme. Once
            # eta is chosen, C calibration can change the achieved final R.
            # Record that final R without presenting it as a matched-R arm.
            out.update(calibration_mode="sequential_C", R_target=None,
                       R_priming_target=args.target_r, R_priming_measurement=priming_record,
                       final_R_in_priming_band=out["R_converged"],
                       calibration_converged=result["converged"])
            out.pop("joint_converged", None)
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2), flush=True)
    if not result["converged"]:
        raise SystemExit("Calibration did not reach its target band; do not use this pair for rollout")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="research directory with checkpoint metadata")
    parser.add_argument("--task", choices=["can", "square", "coffee", "threading", "tool_hang"])
    parser.add_argument("--dp-ckpt", type=Path)
    parser.add_argument("--vib-ckpt", type=Path)
    parser.add_argument("--core-hdf5", type=Path)
    parser.add_argument("--eval-config", type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--potential-cap", type=float)
    mode.add_argument("--c-reference", type=Path, help="jointly preserve this base pair's measured C and the R target")
    parser.add_argument("--sequential-c", action="store_true",
                        help="with c-reference: prime eta at initial kappa, then hold eta fixed while matching C")
    parser.add_argument("--eta-initial", type=float, help="optional previous eta for C-mode dose priming")
    parser.add_argument("--eta-max-probes", type=int, default=12)
    parser.add_argument("--target-r", type=float, default=.01)
    parser.add_argument("--band", type=float, default=.1)
    parser.add_argument("--initial-kappa", type=float, default=2.5)
    parser.add_argument("--kappa-min", type=float, default=.001)
    parser.add_argument("--kappa-max", type=float, default=100)
    parser.add_argument("--max-probes", type=int, default=16)
    parser.add_argument("--no-reuse", action="store_true")
    parser.add_argument("--reuse-only", action="store_true", help="fail without allocating a GPU if no cached pair matches")
    parser.add_argument("--gpu", type=int, choices=range(8))
    parser.add_argument("--gpu-uuid")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    direct = [args.task, args.dp_ckpt, args.vib_ckpt, args.core_hdf5]
    if (args.source and any(x is not None for x in direct)) or (not args.source and not all(direct)):
        parser.error("use --source OR all of --task --dp-ckpt --vib-ckpt --core-hdf5")
    if args.eval_config and args.source:
        parser.error("--eval-config is for direct checkpoint mode; research cache uses the task config")
    values = (args.target_r, args.initial_kappa) + (() if args.potential_cap is None else (args.potential_cap,))
    if any(not math.isfinite(x) or x <= 0 for x in values):
        parser.error("P, R and initial kappa must be finite and positive")
    if args.eta_max_probes < 1:
        parser.error("eta-max-probes must be positive")
    if (args.sequential_c or args.eta_initial is not None) and args.c_reference is None:
        parser.error("sequential-c and eta-initial require c-reference")
    if args.eta_initial is not None and (not math.isfinite(args.eta_initial) or args.eta_initial <= 0):
        parser.error("eta-initial must be finite and positive")
    if not 0 <= args.band < 1:
        parser.error("band must be in [0,1)")
    calibrate(args)


if __name__ == "__main__":
    main()
