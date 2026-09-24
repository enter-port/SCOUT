#!/usr/bin/env python3
"""Evaluate a calibrated (eta, kappa) on an existing frozen failure set."""
import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="task directory containing eval/ and failed.json")
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--gpu-uuid", required=True, help="expected physical GPU UUID, checked before starting")
    parser.add_argument("--fixed-eta", action="store_true", help="evaluate a fixed-eta diagnostic without claiming matched R")
    parser.add_argument("--guide-off", action="store_true", help="run the paired unguided DP control")
    parser.add_argument("--wait-for-eval-seconds", type=int, default=0,
                        help="wait for another owned controller's initial eval before checking GPU and starting")
    args = parser.parse_args()
    root = Path.cwd().resolve()
    output = args.output.resolve()
    output.relative_to(root / "experiments")
    source = args.source.resolve()
    source.relative_to(root / "experiments")
    if args.wait_for_eval_seconds < 0:
        parser.error("wait-for-eval-seconds must be nonnegative")
    deadline = time.monotonic() + args.wait_for_eval_seconds
    while not ((source / "eval/summary.json").is_file() and (source / "failed.json").is_file()):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Initial eval is not complete: {source}")
        time.sleep(min(10, max(0, deadline - time.monotonic())))
    calibration = json.loads(args.calibration.read_text())
    baseline = json.loads((source / "eval/summary.json").read_text())
    if not args.guide_off and not args.fixed_eta and not calibration.get("R_converged", False):
        raise ValueError("Refusing an unconverged dose calibration")
    if not args.guide_off and calibration.get("C_target") is not None and not calibration.get("C_converged"):
        raise ValueError("Refusing an unconverged joint C calibration")
    if args.fixed_eta and calibration.get("R_target") is not None:
        raise ValueError("Use matched-dose validation for a diagnostic with an R target")
    if any(not math.isfinite(calibration[key]) or calibration[key] <= 0 for key in ("eta", "kappa")):
        raise ValueError("eta and kappa must be finite and positive")
    for key in ["dp_ckpt", "core_hdf5"]:
        if Path(calibration[key]).resolve() != Path(baseline[key]).resolve():
            raise ValueError(f"Calibration and frozen eval disagree on {key}")
    if baseline["n_init_states"] != 100 or baseline["eval_seed"] != 42:
        raise ValueError("Expected 100 frozen eval scenes starting at seed 42")
    task = baseline["task"]
    if task not in ["can", "square", "coffee", "threading", "tool_hang"]:
        raise ValueError(task)
    for key in ["dp_ckpt", "vib_ckpt", "core_hdf5"]:
        if not Path(calibration[key]).is_file():
            raise FileNotFoundError(calibration[key])
    failed_set = source / "failed.json"
    if not failed_set.is_file():
        raise FileNotFoundError(failed_set)
    reading = subprocess.check_output([
        "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid,memory.used", "--format=csv,noheader,nounits"
    ], text=True).strip().split(",")
    if (reading[0].strip() != args.gpu_uuid or int(reading[1].strip()) >= 128
            or args.gpu_uuid == "GPU-349c44c7-da75-dba2-5a49-8f9a19a84832"):
        raise RuntimeError(f"GPU identity/occupancy check failed: {reading}")
    output.mkdir(parents=True, exist_ok=False)
    summary, success, all_trajs = [output / n for n in ["summary.json", "success.hdf5", "all.hdf5"]]
    guidance = ["--guide", "off"] if args.guide_off else [
        "--guide", "atypical", "--vib-ckpt", calibration["vib_ckpt"],
        "--guidance-scale", str(calibration["eta"]), "--atypical-cap", str(calibration["kappa"])]
    command = [
        "bash", "scripts/infra/shard_rollout.sh", "4", str(summary), str(success), str(all_trajs),
        calibration["core_hdf5"], "--", "--config", f"configs/eval_{task}_entropy.yaml",
        "--task", task, "--exp-num", "0", "--base-dp-ckpt", calibration["dp_ckpt"],
        "--core-hdf5", calibration["core_hdf5"], *guidance,
        "--seed", "42", "--eval-seed", "42", "--n-init-states", "100", "--explore-mode", "rescue",
        "--failed-set-json", str(failed_set), "--explore-try-times", "5", "--n-envs", "25",
        "--no-wandb", "--output-dir", str(output), "--output-success", str(success), "--output-all", str(all_trajs),
    ]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), SCOUT_RENDER_GPU=str(args.gpu),
               MUJOCO_GL="egl", TMPDIR="/tmp", PYTHONUNBUFFERED="1", PYTHON=sys.executable)
    manifest = {"calibration": calibration, "command": command, "gpu": args.gpu,
                "gpu_uuid": args.gpu_uuid, "fixed_eta": args.fixed_eta, "guide_off": args.guide_off}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{task} eta={calibration['eta']} kappa={calibration['kappa']} -> {output}", flush=True)
    with (output / "rollout.stdout").open("w") as log:
        result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT)
    (output / "exit_code.txt").write_text(str(result.returncode) + "\n")
    result.check_returncode()
    result_summary = json.loads(summary.read_text())
    if result_summary["explore_try_times"] != 5:
        raise ValueError("Unexpected retry count in completed rollout")
    print(f"rescued={result_summary['exploration_rescued']} pass@5={result_summary['pass_at_5']}", flush=True)


if __name__ == "__main__":
    main()
