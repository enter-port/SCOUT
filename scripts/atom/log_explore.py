"""Append validated, merged exploration metrics to the existing round run."""
import argparse

from .common import read_json


def publish(metrics, *, project, run_id, name, mode, directory):
    import wandb

    if not run_id:
        raise ValueError("exploration upload requires the evaluation run ID")
    tries = int(metrics["explore_try_times"])
    payload = {
        "explore_init_done": int(metrics["n_failed"]),
        f"explore/pass@{tries}": metrics["pass_at_5"],
        "explore/success_count": metrics["exploration_rescued"],
        "explore/total": metrics["n_failed"],
        "final/explore_success_num": metrics["exploration_rescued"],
        "final/pass_at_k": metrics["pass_at_5"],
    }
    if metrics.get("avg_jerk") is not None:
        payload["explore/avg_jerk"] = metrics["avg_jerk"]
    run = wandb.init(project=project, id=run_id, resume="must", name=name,
                     mode=mode, dir=str(directory))
    try:
        run.define_metric("explore_init_done", hidden=True)
        run.define_metric("explore/*", step_metric="explore_init_done")
        # Automatic W&B steps stay monotonic across eval/explore/DP/dyn.
        run.log(payload)
    except BaseException:
        run.finish(exit_code=1)
        raise
    else:
        run.finish()


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--metrics", required=True)
    p.add_argument("--project", required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--mode", choices=["online", "offline"], required=True)
    p.add_argument("--dir", required=True)
    args = p.parse_args()
    publish(read_json(args.metrics), project=args.project, run_id=args.run_id,
            name=args.name, mode=args.mode, directory=args.dir)


if __name__ == "__main__":
    main()
