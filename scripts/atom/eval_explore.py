"""Unguided evaluation, freeze failed scenes, then sharded rescue and merge."""
from pathlib import Path

from .common import Context, finish, parser, read_json


def run(ctx, out, dp, dyn=None, arm="ATY", round_num=1, dose=None, frozen=None,
        options=None):
    if arm not in {"ATY", "ORBIT", "DP"}:
        raise ValueError(f"unknown arm: {arm}")
    if arm != "DP" and not dyn:
        raise ValueError("guided exploration requires a dynamics checkpoint")
    dose = dose or {"eta": 3.0, "kappa": 2.5}
    opts = dict(ctx.c.get("rollout", {}))
    if options:
        opts.update(options)
    workers = int(opts.get("workers", 4))
    scenes = int(opts.get("scenes", 100))
    tries = int(opts.get("pass_k", 5))
    eval_envs = int(opts.get("eval_envs", 12))
    envs_per_worker = int(opts.get("envs_per_worker", 12))
    if min(workers, scenes, tries, eval_envs, envs_per_worker) < 1 or workers > scenes:
        raise ValueError("require scenes >= workers >= 1 and pass_k >= 1")
    common = ["--config", ctx.eval_config, "--task", ctx.task, "--exp-num", round_num,
              "--base-dp-ckpt", dp, "--core-hdf5", ctx.core,
              "--seed", opts.get("seed", 42), "--eval-seed", opts.get("seed", 42),
              "--n-init-states", scenes, "--explore-mode", "rescue"]

    def guidance_args():
        if arm == "DP":
            return ["--guide", "off"]
        guidance = ["--guide", {"ATY": "atypical", "ORBIT": "orbit"}[arm],
                    "--vib-ckpt", dyn, "--guidance-scale", dose["eta"],
                    "--atypical-cap", dose["kappa"]]
        if arm == "ORBIT":
            orbit = {**ctx.c.get("orbit", {}), **dose.get("orbit", {})}
            guidance += ["--orbit-lam", orbit.get("lam", .5),
                         "--orbit-delta", orbit.get("delta", .25),
                         "--orbit-sigma", orbit.get("sigma", .05),
                         "--orbit-round", round_num,
                         "--orbit-sigma-decay", orbit.get("sigma_decay", .5),
                         "--orbit-noise-anneal", orbit.get("anneal", 2),
                         "--orbit-fb-clamp", "soft"]
        return guidance

    def evaluate(work):
        failed, metrics = work / "failed.json", work / "eval.json"
        wb = (["--no-wandb"] if ctx.wandb == "disabled" else
              ["--wandb-minimal", "--wandb-project", ctx.project, "--wandb-name", f"{arm}-round{round_num}"])
        # The threading round driver evaluates with the same guide/dose as
        # rescue.  --eval-only still performs exactly one attempt per fixed
        # scene and only freezes the failures for phase B.
        ctx.module("scout.eval.run_rollout", common + guidance_args() + ["--eval-only",
                   "--save-failed-set", failed, "--output-json", metrics,
                   "--output-dir", work, "--n-envs", eval_envs, *wb], work / "eval.log")
        result = {"failed": str(failed), "eval": str(metrics),
                  "artifacts": [str(failed), str(metrics)]}
        if not ctx.dry and metrics.exists():
            measured = read_json(metrics)
            result["wandb_run_id"] = measured.get("wandb_run_id")
        return result

    evaluation = frozen or ctx.stage(Path(out) / "eval", {"dp": dp, "round": round_num}, evaluate)

    def explore(work):
        metrics, success, all_data = work / "explore.json", work / "success.hdf5", work / "all.hdf5"
        guidance = guidance_args()
        stop_first = bool(opts.get("stop_on_first_success", True))
        ctx.module("scripts.atom.shard_rollout", [workers, metrics, success, all_data, ctx.core,
                    "--", *common, *guidance, "--failed-set-json", evaluation["failed"],
                    "--explore-try-times", tries, *(["--stop-on-first-success"] if stop_first else []),
                    "--no-wandb", "--n-envs", envs_per_worker,
                    "--flush-every", opts.get("flush_every", 100), "--output-dir", work,
                    "--output-success", success, "--output-all", all_data], work / "explore.log",
                   extra_env={"CLEANUP_SHARDS": "1"})
        result = {**evaluation, "metrics": str(metrics),
                  "success": str(success) if ctx.dry or success.exists() else None,
                  "all": str(all_data) if ctx.dry or all_data.exists() else None}
        result["artifacts"] = evaluation["artifacts"] + [str(metrics)] + [result[k] for k in ("success", "all") if result[k]]
        # No failures / no rescued scenes legitimately produce no HDF5.
        if not ctx.dry:
            m = read_json(metrics)
            if m["n_init_states"] != scenes or m["explore_try_times"] != tries:
                raise ValueError("merged rollout protocol does not match configured scene/pass budget")
            for key, count in (("success", m.get("collected_trajs", 0)), ("all", m.get("n_all_trajs", 0))):
                if count and not result[key]:
                    raise ValueError(f"rollout reports {count} {key} trajectories but merged HDF5 is missing")
            result.update(success_rate=m["success_rate"], pass_at_k=m["pass_at_5"],
                          wandb_run_id=evaluation.get("wandb_run_id"))
        return result

    return ctx.stage(Path(out) / "explore", {"dp": dp, "dyn": dyn, "arm": arm,
                     "round": round_num, "dose": dose, "options": options or {},
                     "evaluation": evaluation}, explore)


def main():
    p = parser(__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--dp-ckpt", required=True)
    p.add_argument("--dyn-ckpt")
    p.add_argument("--arm", choices=["ATY", "ORBIT", "DP"], default="ATY")
    p.add_argument("--round", type=int, default=1)
    p.add_argument("--eta", type=float, default=3.)
    p.add_argument("--kappa", type=float, default=2.5)
    args = p.parse_args()
    ctx = Context.from_args(args)
    finish(run(ctx, ctx.path(args.out), str(ctx.path(args.dp_ckpt)),
               str(ctx.path(args.dyn_ckpt)) if args.dyn_ckpt else None,
               args.arm, args.round, {"eta": args.eta, "kappa": args.kappa}))


if __name__ == "__main__":
    main()
