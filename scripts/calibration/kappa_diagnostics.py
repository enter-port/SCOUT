#!/usr/bin/env python3
"""Measure natural DP KL drift and guidance dose on the existing core batch.

Uses the same B=128 observation selection and RNG seed as cfk_rcalib.py.
Reads completed base eval metadata; writes only diagnostics in the probe tree.
No environment rollouts, checkpoint changes, or production planner changes.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scout.calib.core import prepare_core, load_model_pair
from scout.calib.dp_kl import posterior_diversity, calibrate_dose


def probe_task(root, task, kappas, target_r=None, natural_only=False, diversity_samples=0,
               eta_override=None):
    import numpy as np

    directory = root / task
    if (directory / "checkpoints.json").exists():
        source = json.loads((directory / "checkpoints.json").read_text())
        eta = source["eta_initial"]
        vib_paths = [Path(source["vib_ckpt"])]
    else:
        source = json.loads((directory / "eval/summary.json").read_text())
        eta = json.loads((directory / "eta.json").read_text())["eta"]
        vib_paths = list((Path(source["dp_ckpt"]).parents[3] / "dyn/dyn-base").glob("*/scout_vib.ckpt"))
    if eta_override is not None:
        eta = eta_override
    dp_path = Path(source["dp_ckpt"])
    if len(vib_paths) != 1:
        raise ValueError(f"Expected exactly one base dyn checkpoint: {vib_paths}")
    cfg, obs_dict, action_scale = prepare_core(
        f"configs/eval_{task}_entropy.yaml", source["core_hdf5"])
    gst = int(cfg.exploration.get("guidance_start_timestep", 50))
    dp, vib, bridge, obs_adapter = load_model_pair(cfg, dp_path, vib_paths[0])
    metadata = dict(source, task=task, dp_ckpt=str(dp_path), vib_ckpt=str(vib_paths[0]))

    if diversity_samples:
        posterior_diversity(directory, dict(source, task=task, vib_ckpt=str(vib_paths[0])),
                            dp, vib, bridge, obs_adapter, obs_dict, gst, diversity_samples)
        return

    output = directory / ("diagnostics" if target_r is None else "diagnostics_rmatched")
    output.mkdir(exist_ok=True)
    conditions = [("natural", 0.0, 2.5)]
    if not natural_only:
        conditions += [(f"k{k:g}", eta, k) for k in kappas]
    for label, dose, cap in conditions:
        dest = output / f"{label}.json"
        if dest.exists():
            saved = json.loads(dest.read_text())
            if (saved.get("R_target") != target_r or saved["kappa"] != cap
                    or Path(saved["dp_ckpt"]).resolve() != dp_path.resolve()
                    or Path(saved["vib_ckpt"]).resolve() != vib_paths[0].resolve()
                    or (target_r is None and saved["eta"] != dose)):
                raise ValueError(f"Existing diagnostic has different inputs: {dest}")
            print(f"[{task}] existing {dest}", flush=True)
            continue
        stats, arrays = calibrate_dose(metadata, dp, vib, bridge, obs_adapter,
                                       obs_dict, gst, action_scale, cap, dose,
                                       None if label == "natural" else target_r)
        # Natural probes remain uncalibrated, but carry the requested target
        # for compatibility with the existing diagnostic-cache identity.
        stats["R_target"] = target_r
        dose = stats["eta"]
        np.savez_compressed(output / f"{label}.npz", **arrays)
        dest.write_text(json.dumps(stats, indent=2) + "\n")
        print(f"[{task}] {label} eta={dose:.5g} R={stats['R_mean']:.5g} "
              f"C={stats['C_mean']:.5g} cap_fraction={stats['cap_fraction']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", choices=["can", "square", "coffee", "threading", "tool_hang"], required=True)
    parser.add_argument("--kappas", nargs="+", type=float, default=[.5, 1, 2.5, 5, 10])
    parser.add_argument("--target-r", type=float, help="recalibrate eta at each cap before measuring")
    parser.add_argument("--eta", type=float,
                        help="explicit fixed eta, or starting eta when target-r is set")
    parser.add_argument("--natural-only", action="store_true")
    parser.add_argument("--diversity-samples", type=int, default=0,
                        help="instead measure independent unguided DP posterior divergences")
    args = parser.parse_args()
    if any(not 0 < k < float("inf") for k in args.kappas):
        parser.error("kappas must be finite and positive")
    if args.target_r is not None and not 0 < args.target_r < float("inf"):
        parser.error("target-r must be finite and positive")
    if args.eta is not None and (not 0 < args.eta < float("inf")
                                  or args.natural_only or args.diversity_samples):
        parser.error("eta must be finite and positive and used with a guided probe")
    if args.diversity_samples and (args.diversity_samples < 2 or args.natural_only or args.target_r is not None):
        parser.error("diversity-samples requires at least two draws and cannot be combined with other probe modes")
    for task in args.tasks:
        probe_task(args.root, task, args.kappas, args.target_r, args.natural_only,
                   args.diversity_samples, args.eta)


if __name__ == "__main__":
    main()
