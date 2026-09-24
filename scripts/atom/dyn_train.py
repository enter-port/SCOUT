"""Train dynamics on core plus accumulated successes AND failures using the new DP encoder."""
from .common import Context, checkpoint, dataset, finish, parser


def train(ctx, out, dp, trajectories=(), name="dyn-base", base=False, options=None,
          resume_run_id=None):
    spec = {"dp": dp, "trajectories": list(trajectories), "name": name,
            "base": base, "options": options or {}, "resume_run_id": resume_run_id}
    def action(work):
        data = str(ctx.core) if base else dataset(ctx, work / "all_accum.hdf5", trajectories)
        config_path = work / "config.yaml"
        opts = ctx.training_options("dyn", base=base, overrides=options)
        if not ctx.dry:
            import yaml
            cfg = yaml.safe_load(ctx.path(ctx.c.get("dyn_config", f"configs/{ctx.task}/dyn.yaml")).read_text())
            cfg["dataset"].update(zarr_path=data, feature_cache=True)
            cfg["model"]["E_s"]["base_dp_ckpt"] = dp
            cfg.update(seed=ctx.seed, cudnn_deterministic=True, save_dir=str(work),
                       num_epochs=int(opts.get("epochs", 300)),
                       batch_size=int(opts.get("batch_size", cfg["batch_size"])),
                       beta=float(opts.get("beta", cfg["beta"])),
                       steps_per_epoch=int(opts.get("steps_per_epoch", cfg["steps_per_epoch"])),
                       use_wandb=ctx.wandb != "disabled")
            cfg.setdefault("wandb", {}).update(name=name, project=ctx.project, minimal=True)
            config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        else:
            print(f"DRY_RUN: dyn config dataset={data} encoder={dp} options={opts} -> {config_path}")
            ctx.module("scout.train_vib", ["--config", config_path], work / "train.log",
                       extra_env=({"WANDB_RUN_ID": resume_run_id, "WANDB_RESUME": "must"}
                                  if resume_run_id and ctx.wandb != "disabled" else None))
        ckpt = checkpoint(work, "*/scout_vib.ckpt", ctx.dry)
        return {"dyn": ckpt, "dataset": data, "artifacts": [ckpt, data]}

    return ctx.stage(out, spec, action)


def main():
    p = parser(__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--dp-ckpt", required=True)
    p.add_argument("--all-hdf5", nargs="*", default=[])
    p.add_argument("--base", action="store_true")
    args = p.parse_args()
    ctx = Context.from_args(args)
    finish(train(ctx, ctx.path(args.out), str(ctx.path(args.dp_ckpt)), args.all_hdf5, base=args.base))


if __name__ == "__main__":
    main()
