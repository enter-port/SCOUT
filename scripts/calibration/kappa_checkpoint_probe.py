#!/usr/bin/env python3
"""Calibrate eta, freeze 100 scenes, then compare DP and SCOUT at one kappa.

The source directory must contain checkpoints.json with task, dp_ckpt,
vib_ckpt, core_hdf5 and eta_initial. This evaluates existing checkpoints;
it does not train models. Completed phases can be reused only when their
saved inputs match. Incomplete rollout directories require inspection.
"""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=range(8), required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--kappa", type=float, default=2.5)
    parser.add_argument("--kappa-from", type=Path, help="read the predicted kappa from a core-only transfer JSON")
    parser.add_argument("--calibration", type=Path, help="use an already calibrated joint pair instead of recalibrating eta")
    parser.add_argument("--arm-name", help="name for this SCOUT result, e.g. p6")
    parser.add_argument("--controls-only", action="store_true", help="freeze initial scenes and run DP; no guidance calibration or SCOUT")
    args = parser.parse_args()
    if args.controls_only and (args.calibration or args.kappa_from):
        parser.error("controls-only does not use a guidance calibration")
    if args.kappa_from:
        if args.calibration:
            parser.error("kappa-from and calibration are mutually exclusive")
        args.kappa = json.loads(args.kappa_from.read_text())["kappa"]
    if not math.isfinite(args.kappa) or args.kappa <= 0:
        parser.error("kappa must be finite and positive")
    root = Path.cwd().resolve()
    source = args.source.resolve()
    source.relative_to(root / "experiments")
    inputs = json.loads((source / "checkpoints.json").read_text())
    calibration = None
    if args.calibration:
        calibration = json.loads(args.calibration.read_text())
        for key in ("task", "dp_ckpt", "vib_ckpt", "core_hdf5"):
            if calibration[key] != inputs[key]:
                raise ValueError(f"Joint calibration uses a different {key}")
        target_r = calibration.get("R_target")
        if (not calibration.get("R_converged") or not isinstance(target_r, (int, float))
                or not math.isfinite(target_r) or target_r <= 0
                or not math.isfinite(calibration["R_mean"])
                or abs(calibration["R_mean"] / target_r - 1) > .1):
            raise ValueError("Joint calibration must reach its stated R target within 10 percent")
        if calibration.get("C_target") is not None and not calibration.get("C_converged"):
            raise ValueError("Joint C calibration must also reach its C target")
        args.kappa = calibration["kappa"]
    arm_name = args.arm_name or ("controls" if args.controls_only else f"k{args.kappa:g}")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", arm_name) or arm_name in ("dp", "eval"):
        raise ValueError("Invalid SCOUT arm name")
    task = inputs["task"]
    if task not in ("can", "square", "coffee", "threading", "tool_hang"):
        raise ValueError(task)
    for key in ("dp_ckpt", "vib_ckpt", "core_hdf5"):
        if not Path(inputs[key]).is_file():
            raise FileNotFoundError(inputs[key])
    signature = source / f"probe_{arm_name}.inputs.json"
    requested = dict(inputs, kappa=args.kappa, R_target=.01, workers=4, n_envs=25,
                     eval_seed=42, n_init_states=100, attempts=5)
    if args.controls_only:
        requested.update(controls_only=True, R_target=None)
    if calibration is not None:
        requested["R_target"] = calibration["R_target"]
        requested["joint_calibration"] = calibration
    if args.kappa_from:
        requested["kappa_transfer"] = json.loads(args.kappa_from.read_text())
    if signature.exists() and json.loads(signature.read_text()) != requested:
        raise ValueError("Existing probe has different inputs")
    signature.write_text(json.dumps(requested, indent=2) + "\n")
    config = f"configs/eval_{task}_entropy.yaml"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu), SCOUT_RENDER_GPU=str(args.gpu),
               MUJOCO_GL="egl", TMPDIR="/tmp", PYTHONUNBUFFERED="1", PYTHON=sys.executable)

    def run(command, log):
        reading = subprocess.check_output([
            "nvidia-smi", "-i", str(args.gpu), "--query-gpu=uuid,memory.used",
            "--format=csv,noheader,nounits"], text=True).strip().split(",")
        if (reading[0].strip() != args.gpu_uuid or int(reading[1]) >= 128
                or args.gpu_uuid == "GPU-349c44c7-da75-dba2-5a49-8f9a19a84832"):
            raise RuntimeError(f"GPU identity/occupancy check failed: {reading}")
        with log.open("w") as stream:
            subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)

    if args.controls_only:
        calibration = {key: inputs[key] for key in ("task", "dp_ckpt", "vib_ckpt", "core_hdf5")}
        calibration.update(eta=inputs["eta_initial"], kappa=args.kappa,
                           R_target=None, R_converged=False, guidance_parameters_unused=True)
        cal_path = source / "control_metadata.json"
        if cal_path.exists() and json.loads(cal_path.read_text()) != calibration:
            raise ValueError("Existing control has different inputs")
        cal_path.write_text(json.dumps(calibration, indent=2) + "\n")
    elif calibration is None:
        eta_path = source / f"eta_k{args.kappa:g}.json"
        if not eta_path.exists():
            run([sys.executable, "scripts/coffee/cfk_rcalib.py", "--eval-config", config,
                 "--dp-ckpt", inputs["dp_ckpt"], "--vib-ckpt", inputs["vib_ckpt"],
                 "--core-hdf5", inputs["core_hdf5"], "--eta-prev", str(inputs["eta_initial"]),
                 "--kappa-prev", str(args.kappa), "--max-repeat", "8", "--out", str(eta_path)],
                source / f"eta_k{args.kappa:g}.stdout")
        eta = json.loads(eta_path.read_text())
        if not eta["converged"] or eta["kappa"] != args.kappa:
            raise ValueError("Dose calibration did not converge at requested kappa")
        calibration = {key: inputs[key] for key in ("task", "dp_ckpt", "vib_ckpt", "core_hdf5")}
        calibration.update(eta=eta["eta"], kappa=args.kappa, R_target=.01,
                           R_mean=eta["R_mean"], R_converged=True, eta_source=str(eta_path))
        cal_path = source / f"calibration_k{args.kappa:g}.json"
        if cal_path.exists() and json.loads(cal_path.read_text()) != calibration:
            raise ValueError("Existing calibration has different inputs")
        cal_path.write_text(json.dumps(calibration, indent=2) + "\n")
    else:
        cal_path = args.calibration.resolve()
    evaluation = source / "eval"
    if not (source / "failed.json").exists():
        evaluation.mkdir(exist_ok=False)
        run([sys.executable, "-m", "scout.eval.run_rollout", "--config", config,
             "--task", task, "--exp-num", "0", "--base-dp-ckpt", inputs["dp_ckpt"],
             "--core-hdf5", inputs["core_hdf5"], "--guide", "off", "--seed", "42",
             "--eval-seed", "42", "--n-init-states", "100", "--explore-mode", "rescue",
             "--eval-only", "--save-failed-set", str(source / "failed.json"),
             "--n-envs", "25", "--no-wandb", "--output-dir", str(evaluation),
             "--output-json", str(evaluation / "summary.json")], evaluation / "stdout")
    baseline = json.loads((evaluation / "summary.json").read_text())
    for key in ("dp_ckpt", "core_hdf5"):
        if Path(baseline[key]).resolve() != Path(inputs[key]).resolve():
            raise ValueError(f"Existing eval uses different {key}")
    for arm in (("dp",) if args.controls_only else ("dp", arm_name)):
        output = source / arm
        if output.exists():
            if ((output / "exit_code.txt").is_file()
                    and (output / "exit_code.txt").read_text().strip() == "0"
                    and (output / "summary.json").is_file()):
                previous = json.loads((output / "manifest.json").read_text())["calibration"]
                if any(previous[key] != calibration[key] for key in ("dp_ckpt", "core_hdf5")):
                    raise ValueError(f"Existing arm uses different inputs: {output}")
                if arm != "dp" and previous != calibration:
                    raise ValueError(f"Existing arm uses different calibration: {output}")
                continue
            raise FileExistsError(f"Inspect incomplete arm before resuming: {output}")
        command = [sys.executable, "scripts/calibration/kappa_eval_arm.py", "--source", str(source),
                   "--calibration", str(cal_path), "--output", str(output),
                   "--gpu", str(args.gpu), "--gpu-uuid", args.gpu_uuid]
        if arm == "dp":
            command.append("--guide-off")
        # The arm runner handles GPU visibility and checks occupancy itself.
        # Calling it outside the modified env prevents nested CUDA renumbering.
        with (source / f"{arm}.driver.log").open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
        result = json.loads((output / "summary.json").read_text())
        print(f"{task} {arm}: rescued={result['exploration_rescued']} pass@5={result['pass_at_5']}", flush=True)


if __name__ == "__main__":
    main()
