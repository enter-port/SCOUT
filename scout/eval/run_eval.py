"""SCOUT base-DP metric evaluation launcher (eval half of the eval/rollout split).

Evaluates a frozen base DP checkpoint with the SOE round-summary protocol:

  * N (default 100) random initial states, each run ``try_times`` (default 5)
    times via the single-process vectorized rollout engine.
  * **success_rate**    : fraction solved on the FIRST try.
  * **jerk_baseline**   : mean action jerk over successful first-try trajs.
  * **base_pass_at_5**  : fraction solved at least once across all tries.

No dynamics model / VIB / guided exploration / retrain -- pure baseline metric
measurement. Use ``run_rollout.py`` for data collection, ``run_round.py`` for
the full self-improvement loop (eval -> guided rollout -> retrain).

Server usage (venv active, wandb env sourced, cwd = repo root):
  .venv/bin/python -m scout.eval.run_eval \\
      --config configs/eval_square.yaml --task square \\
      --base-dp-ckpt <.../580.ckpt> \\
      --core-hdf5   <.../image_v141_abs_core20.hdf5> \\
      --cuda-visible-devices 0

Parallelism: ``cfg.eval.n_envs`` (default 50) drives the vectorized rollout
(N envs, batched policy inference); ``n_envs==1`` falls back to sequential. A
wandb run (project ``scout-eval``, name ``DP-{task}-eval``) streams live
``eval/baseline_*`` progress + a final ``round/*`` summary. ``--no-wandb``
disables.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from scout.eval.evaluator import EvalPipeline
from scout.eval.factories import (
    load_cfg,
    make_default_env_factory,
    make_lpb_dp_factory,
)


def _to_plain(d: Any) -> Any:
    """Recursively convert EasyDict -> plain dict/list (wandb-safe config)."""
    if isinstance(d, dict):
        return {k: _to_plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_to_plain(x) for x in d]
    return d


def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/eval_<task>.yaml")
    p.add_argument("--task", required=True, help="lift | can | square (wandb name)")
    p.add_argument("--base-dp-ckpt", required=True,
                   help="base DP ckpt to evaluate")
    p.add_argument("--core-hdf5", required=True,
                   help="core hdf5 (env_meta source for the env factory)")
    p.add_argument("--n-init-states", type=int, default=None,
                   help="override cfg.eval.n_init_states (default 100)")
    p.add_argument("--try-times", type=int, default=None,
                   help="override cfg.eval.try_times (default 5 = pass@5)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="override cfg.eval.n_envs (parallel env count; "
                        "default 50, 1 -> sequential fallback)")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="GPU id (e.g. 0)")
    p.add_argument("--wandb-name", default=None, help="default DP-{task}-eval")
    p.add_argument("--wandb-project", default=None,
                   help="override cfg.wandb.project (default scout-eval)")
    p.add_argument("--wandb-dir", default=None,
                   help="wandb run dir (default log_root)")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb live logging")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    cfg.base_dp.initial_ckpt_path = args.base_dp_ckpt
    cfg.dataset.path = args.core_hdf5
    if args.n_init_states is not None:
        cfg.eval.n_init_states = int(args.n_init_states)
    if args.try_times is not None:
        cfg.eval.try_times = int(args.try_times)
    if args.n_envs is not None:
        cfg.eval.n_envs = int(args.n_envs)
    cfg.eval.n_envs = int(getattr(cfg.eval, "n_envs", 1))

    # CUDA_VISIBLE_DEVICES MUST be set BEFORE torch.device() -- otherwise the
    # device would be pinned to physical GPU 0 regardless of the requested id.
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    wandb_name = args.wandb_name or f"DP-{args.task}-eval"

    # ---- wandb (live baseline progress) --------------------------------- #
    wcfg = cfg.get("wandb", {}) or {}
    use_wandb = bool(wcfg.get("use_wandb", True)) and not args.no_wandb
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(
                project=args.wandb_project or wcfg.get("project", "scout-eval"),
                name=wandb_name,
                config=_to_plain(cfg),
                dir=args.wandb_dir or f"data/outputs/eval_{args.task}",
                tags=list(wcfg.get("tags", ["step2", "eval"])) + [args.task],
            )
            print(f"[run_eval] wandb: project={wandb_run.project} name={wandb_name}")
        except Exception as e:  # wandb optional -- never block the run on it
            print(f"[run_eval] wandb disabled (init failed: {e})")
            wandb_run = None
    else:
        print("[run_eval] wandb disabled (--no-wandb / cfg.wandb.use_wandb=false)")

    dp_factory = make_lpb_dp_factory(device)
    env_factory = make_default_env_factory(cfg)

    print(f"[run_eval] task={args.task} wandb={wandb_name} "
          f"n_init={cfg.eval.n_init_states} try_times={cfg.eval.try_times} "
          f"n_envs={cfg.eval.n_envs} device={device}")
    print(f"[run_eval] base_dp = {args.base_dp_ckpt}")
    print(f"[run_eval] core    = {args.core_hdf5}")

    pipeline = EvalPipeline(cfg=cfg, dp_factory=dp_factory,
                            env_factory=env_factory, device=device,
                            wandb_run=wandb_run)
    try:
        metrics = pipeline.run(args.base_dp_ckpt)
        print("\n=== EVAL METRICS ===")
        print(metrics)
        print(f"\n[run_eval] DONE. success_rate={metrics['success_rate']:.4f} "
              f"base_pass_at_5={metrics.get('base_pass_at_5', 0.0):.4f} "
              f"jerk_baseline={metrics['jerk_baseline']:.4f}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
