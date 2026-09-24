#!/usr/bin/env python3
"""Predict kappa from eight DP draws and calibrate R, with optional evaluation.

This is a candidate rule with task-dependent empirical validity, not a
guarantee of improvement on an arbitrary checkpoint.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--calibrate-only", action="store_true",
                        help="save the core-calibrated pair without environment rollouts")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    source = args.source.resolve()
    source.relative_to(root / "experiments")
    metadata_path = source / "checkpoints.json"
    if not metadata_path.exists():
        metadata_path = source / "diagnostics/natural.json"
    metadata = json.loads(metadata_path.read_text())
    sys.path.insert(0, str(root))
    from scout.calib.dp_kl import validate_calibration

    task = metadata["task"]
    if source.name != task or task not in ("can", "square", "coffee", "threading", "tool_hang"):
        raise ValueError("Source must be this task's experiment directory")
    if not args.calibrate_only and (not (source / "eval/summary.json").exists()
                                    or not (source / "failed.json").exists()):
        raise ValueError("Freeze the initial evaluation before running this arm")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), PYTHONUNBUFFERED="1")

    def run(command, log):
        with log.open("w") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    # This script owns only experiment orchestration. Sampling, the KL
    # statistic, and eta matching all live in scout.calib.dp_kl.
    calibration_path = source / "calib_dp_kl.json"
    if not calibration_path.exists():
        run([sys.executable, "-u", "-m", "scout.calib.dp_kl",
             "--source", str(source), "--gpu", str(args.gpu),
             "--gpu-uuid", args.gpu_uuid, "--out", str(calibration_path)],
            source / "kl_median_calibration.stdout")
    calibration = json.loads(calibration_path.read_text())
    diversity = json.loads((source / "dp_diversity/summary.json").read_text())
    for key in ("task", "dp_ckpt", "vib_ckpt", "core_hdf5"):
        if diversity[key] != metadata[key]:
            raise ValueError(f"DP diversity used a different {key}")
    if diversity["samples"] != 8 or diversity["B"] != 128:
        raise ValueError("Expected eight independent draws at 128 core observations")
    cap = diversity["independent_final_to_anchor"]["quantiles"]["p50"]
    validate_calibration(calibration, metadata, cap)
    if args.calibrate_only:
        print(json.dumps({key: calibration[key] for key in
                          ("task", "kappa", "eta", "R_mean", "R_target")}, indent=2))
        print(f"Core calibration saved at {calibration_path}")
        return
    output = source / "kl_median"
    if output.exists():
        raise FileExistsError(f"Inspect existing arm before reuse: {output}")
    command = [sys.executable, "-u", "scripts/calibration/kappa_eval_arm.py",
               "--source", str(source), "--calibration", str(calibration_path),
               "--output", str(output), "--gpu", str(args.gpu), "--gpu-uuid", args.gpu_uuid]
    # The arm runner establishes CUDA visibility itself from the physical index.
    with (source / "kl_median.driver.log").open("w") as stream:
        subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    result = json.loads((output / "summary.json").read_text())
    print(f"{task}: kappa={cap} rescued={result['exploration_rescued']} pass@5={result['pass_at_5']}")


if __name__ == "__main__":
    main()
