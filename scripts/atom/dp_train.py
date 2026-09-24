"""Train a DP from scratch on core plus accumulated successful trajectories."""
from .common import Context, ROOT, checkpoint, dataset, finish, parser


def train(ctx, out, successes=(), name="DP-base", base=False, options=None,
          resume_run_id=None):
    spec = {"successes": list(successes), "name": name, "base": base,
            "options": options or {}, "resume_run_id": resume_run_id}

    def action(work):
        data = str(ctx.core) if base else dataset(ctx, work / "success_accum.hdf5", successes)
        cfg = ctx.training_options("dp", base=base, overrides=options)
        epochs = int(cfg.get("epochs", 600))
        workers = int(cfg.get("workers", 8))
        bs = int(cfg.get("batch_size", 64))
        args = [ctx.py, ROOT / "train.py", "--config-path", "configs", "--config-name",
                ctx.c.get("dp_config", f"base_dp_{ctx.task}_image"),
                f"task.dataset_path={data}", f"training.seed={ctx.seed}",
                f"task.dataset.seed={ctx.seed}", "training.resume=False", "training.rollout_every=0",
                "training.sample_every=100", "training.cudnn_benchmark=false",
                "+training.cudnn_deterministic=true", "training.device=cuda:0",
                f"training.num_epochs={epochs}",
                f"training.checkpoint_every={cfg.get('checkpoint_every', 100 if base else 300)}",
                f"dataloader.batch_size={bs}", f"val_dataloader.batch_size={bs}",
                "+logging.metric_prefix=DP/", "+logging.wandb_minimal=true",
                f"logging.name={name}", f"logging.project='{ctx.project}'",
                f"logging.mode={ctx.wandb}", f"hydra.run.dir={work}"]
        if not base:
            args += ["task.train_filter_key=scout_aug"]
        args += cfg.get("overrides", [])
        ctx.run(args + [f"dataloader.num_workers={workers}",
                        f"dataloader.persistent_workers={str(workers > 0).lower()}"], work / "train.log",
                extra_env=({"WANDB_RUN_ID": resume_run_id, "WANDB_RESUME": "must"}
                           if resume_run_id and ctx.wandb != "disabled" else None))
        ckpt = checkpoint(work, "checkpoints/*.ckpt", ctx.dry)
        return {"dp": ckpt, "dataset": data, "artifacts": [ckpt, data]}

    return ctx.stage(out, spec, action)


def main():
    p = parser(__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--success-hdf5", nargs="*", default=[])
    p.add_argument("--base", action="store_true")
    args = p.parse_args()
    ctx = Context.from_args(args)
    finish(train(ctx, ctx.path(args.out), args.success_hdf5, base=args.base))


if __name__ == "__main__":
    main()
