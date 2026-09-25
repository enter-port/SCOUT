"""Calibration adapters; algorithms remain in scout.calib."""
import math
import subprocess

from .common import Context, finish, parser, read_json


def _require_measurement(result, key, target, band):
    value = float(result[key])
    if not result.get("converged"):
        raise ValueError(f"calibration {key} did not converge")
    if (not math.isfinite(target) or target <= 0 or not math.isfinite(value)
            or abs(value / target - 1) > band + 1e-12):
        raise ValueError(f"calibration {key} is outside the target band")


def calibrate(ctx, out, dp, dyn, previous, base=None, mode=None, options=None):
    cfg = dict(ctx.c.get("calib", {}))
    if options:
        cfg.update(options)
    mode = mode or cfg.get("mode", "rc")
    if mode not in {"none", "r", "rc", "pr", "dp_kl"}:
        raise ValueError(f"unknown calibration mode: {mode}")
    if mode == "none":
        return dict(previous)
    if mode == "rc" and not base:
        raise ValueError("rc calibration requires a fixed base DP/dyn/eta/kappa reference")

    def action(work):
        common = ["--eval-config", ctx.eval_config, "--core-hdf5", ctx.core]
        pair = ["--dp-ckpt", dp, "--vib-ckpt", dyn]
        target, band = float(cfg.get("target_r", .01)), float(cfg.get("band", .1))
        result_path = work / "dose.json"
        if mode in {"r", "rc"}:
            eta_path = work / "eta.json"
            ctx.module("scout.calib.eta_r", common + pair + ["--eta-prev", previous["eta"],
                       "--kappa-prev", previous["kappa"], "--target", target,
                       "--band-lo", target * (1 - band), "--band-hi", target * (1 + band),
                       "--batch-size", cfg.get("batch_size", 128), "--out", eta_path], work / "eta.log")
            result_path = eta_path
            result = dict(previous) if ctx.dry else read_json(eta_path)
            if not ctx.dry:
                _require_measurement(result, "R_mean", target, band)
            if mode == "rc":
                result_path = work / "kappa.json"
                ctx.module("scout.calib.kappa_c", common + ["--base-dp-ckpt", base["dp"],
                           "--base-vib-ckpt", base["dyn"], "--base-eta", base["eta"],
                           "--base-kappa", base["kappa"], "--round-dp-ckpt", dp,
                           "--round-vib-ckpt", dyn, "--round-eta", result["eta"],
                           "--kappa0", previous["kappa"], "--band", band,
                           "--batch-size", cfg.get("batch_size", 128), "--out", result_path], work / "kappa.log")
        else:
            gpu_uuid = ("<gpu-uuid>" if ctx.dry else subprocess.check_output(
                ["nvidia-smi", "-i", ctx.gpu, "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip())
            args = common + pair + ["--task", ctx.task, "--gpu", ctx.gpu, "--gpu-uuid", gpu_uuid,
                                   "--target-r", target, "--out", result_path]
            if mode == "pr":
                args += ["--potential-cap", cfg.get("potential", 6), "--band", band,
                         "--initial-kappa", previous["kappa"]]
            else:
                args += ["--eta-initial", previous["eta"], "--samples", cfg.get("samples", 8)]
            ctx.module("scout.calib.joint_pr" if mode == "pr" else "scout.calib.dp_kl",
                       args, work / "calib.log")
        result = dict(previous) if ctx.dry else read_json(result_path)
        if mode == "rc" and not ctx.dry:
            _require_measurement(result, "C_mean", float(result["C_target"]), band)
        if not all(math.isfinite(float(result[k])) and float(result[k]) > 0 for k in ("eta", "kappa")):
            raise ValueError("calibration returned a nonpositive/nonfinite dose")
        if mode in {"pr", "dp_kl"} and not ctx.dry and not result.get("R_converged"):
            raise ValueError("joint/median calibration did not converge")
        if mode in {"pr", "dp_kl"} and not ctx.dry:
            # Verify the measured pair rather than trusting only the solver flag.
            actual_band = band if mode == "pr" else .1
            r = float(result["R_mean"])
            if not math.isfinite(r) or abs(r / target - 1) > actual_band + 1e-12:
                raise ValueError("calibration R is outside the target band")
            if mode == "pr" and abs(result["eta"] * result["kappa"] / float(cfg.get("potential", 6)) - 1) > band + 1e-12:
                raise ValueError("calibration eta*kappa is outside the potential band")
        return {"eta": result["eta"], "kappa": result["kappa"], "json": str(result_path),
                "artifacts": ([str(eta_path)] if mode == "rc" else []) + [str(result_path)]}

    return ctx.stage(out, {"dp": dp, "dyn": dyn, "previous": previous, "base": base,
                           "mode": mode, "options": options or {}}, action)


def main():
    p = parser(__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--dp-ckpt", required=True)
    p.add_argument("--dyn-ckpt", required=True)
    p.add_argument("--eta", type=float, required=True)
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--mode", choices=["none", "r", "rc", "pr", "dp_kl"])
    p.add_argument("--base-reference", help="JSON containing dp,dyn,eta,kappa for RC")
    args = p.parse_args()
    ctx = Context.from_args(args)
    finish(calibrate(ctx, ctx.path(args.out), str(ctx.path(args.dp_ckpt)), str(ctx.path(args.dyn_ckpt)),
                     {"eta": args.eta, "kappa": args.kappa},
                     read_json(args.base_reference) if args.base_reference else None, args.mode))


if __name__ == "__main__":
    main()
