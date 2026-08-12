"""SCOUT successful-trajectory collection launcher (rollout half of the
eval/rollout split).

Rolls out a base DP (and, optionally, a SCOUT dynamics model / VIB for guided
exploration) over N (default 100) random initial states x ``try_times``
(default 5) tries and writes EVERY successful trajectory to an hdf5 you name:

  * ``--guide off`` (default): raw base DP via ``predict_action`` -- no
    dynamics model needed. Equivalent to the "base-DP 100x5" collection.
  * ``--guide dyn``: SCOUT VIB-guided exploration via
    ``predict_action_dyn_guided`` over ALL init states (needs ``--vib-ckpt``).

Output: ``--output <path.hdf5>`` -- the core hdf5 (``--core-hdf5``) copied with
the successful rollouts appended as new ``data/demo_*`` groups + a
``mask/<aug_mask_key>`` selecting core + rollouts (see hdf5_writer). You pick
the name + path.

No metric measurement (use ``run_eval.py``) and no retrain (use
``run_round.py`` for the full self-improvement loop).

Server usage (venv active, wandb env sourced, cwd = repo root):
  .venv/bin/python -m scout.eval.run_rollout \\
      --config configs/eval_square.yaml --task square \\
      --base-dp-ckpt <.../580.ckpt> \\
      --core-hdf5   <.../image_v141_abs_core20.hdf5> \\
      --output      data/outputs/rollout_collect_square/rollout_successes.hdf5 \\
      --guide off --cuda-visible-devices 0

Parallelism: ``cfg.eval.n_envs`` (default 50) drives the vectorized rollout. A
wandb run (project ``scout-eval``, name ``DP-{task}-base-rollout`` /
``DP-{task}-SCOUT-rollout``) streams live ``explore/*`` progress +
``round/collected``. ``--no-wandb`` disables.
"""

from __future__ import annotations

import argparse
import os
from typing import Any

import torch

from scout.eval.collector import RolloutCollector
from scout.eval.factories import (
    load_cfg,
    make_default_env_factory,
    make_lpb_dp_factory,
    make_scout_vib_factory,
)
from scout.eval.hdf5_writer import write_rollouts_to_hdf5


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
    p.add_argument("--base-dp-ckpt", required=True, help="base DP ckpt to roll out")
    p.add_argument("--vib-ckpt", default=None,
                   help="scout_vib.ckpt (Step 1); required for --guide dyn only")
    p.add_argument("--core-hdf5", required=True,
                   help="core hdf5 (env_meta source + augmented-write base)")
    p.add_argument("--output", required=True,
                   help="output hdf5 path (core + successful rollouts appended)")
    p.add_argument("--guide", choices=["dyn", "off"], default="off",
                   help="'dyn' = SCOUT VIB-guided exploration over all init "
                        "states (needs --vib-ckpt); 'off' (default) = raw base "
                        "DP, no dyn-model guidance.")
    p.add_argument("--aug-mask-key", default=None,
                   help="mask key written to --output selecting core + rollouts "
                        "(default cfg.self_improvement.scout_aug_mask = scout_aug)")
    p.add_argument("--n-init-states", type=int, default=None,
                   help="override cfg.eval.n_init_states (default 100)")
    p.add_argument("--try-times", type=int, default=None,
                   help="override cfg.eval.try_times (default 5)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="override cfg.eval.n_envs (parallel env count; "
                        "default 50, 1 -> sequential fallback)")
    p.add_argument("--cuda-visible-devices", default=None, help="GPU id (e.g. 0)")
    p.add_argument("--wandb-name", default=None,
                   help="default DP-{task}-base-rollout / DP-{task}-SCOUT-rollout")
    p.add_argument("--wandb-project", default=None,
                   help="override cfg.wandb.project (default scout-eval)")
    p.add_argument("--wandb-dir", default=None,
                   help="wandb run dir (default alongside --output)")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb live logging")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    cfg.base_dp.initial_ckpt_path = args.base_dp_ckpt
    if args.vib_ckpt is not None:
        cfg.vib.ckpt_path = args.vib_ckpt
        cfg.vib.base_dp_ckpt = args.base_dp_ckpt
    cfg.dataset.path = args.core_hdf5
    if args.n_init_states is not None:
        cfg.eval.n_init_states = int(args.n_init_states)
    if args.try_times is not None:
        cfg.eval.try_times = int(args.try_times)
    if args.n_envs is not None:
        cfg.eval.n_envs = int(args.n_envs)
    cfg.eval.n_envs = int(getattr(cfg.eval, "n_envs", 1))

    guided = (args.guide == "dyn")
    if guided and (args.vib_ckpt is None
                   or str(getattr(cfg.vib, "ckpt_path", "")).startswith("<")):
        raise SystemExit("[run_rollout] --guide dyn needs --vib-ckpt (SCOUT VIB "
                         "ckpt for guided exploration). --guide off does not.")

    # CUDA_VISIBLE_DEVICES MUST be set BEFORE torch.device().
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    tag = "SCOUT" if guided else "base"
    wandb_name = args.wandb_name or f"DP-{args.task}-{tag}-rollout"
    out_dir = os.path.dirname(os.path.abspath(args.output)) or "."

    # ---- wandb (live collection progress) ------------------------------- #
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
                dir=args.wandb_dir or out_dir,
                tags=list(wcfg.get("tags", ["step2", "rollout"])) + [args.task],
            )
            print(f"[run_rollout] wandb: project={wandb_run.project} name={wandb_name}")
        except Exception as e:  # wandb optional -- never block the run on it
            print(f"[run_rollout] wandb disabled (init failed: {e})")
            wandb_run = None
    else:
        print("[run_rollout] wandb disabled (--no-wandb / cfg.wandb.use_wandb=false)")

    dp_factory = make_lpb_dp_factory(device)
    scout_vib_factory = make_scout_vib_factory(cfg, device) if guided else None
    env_factory = make_default_env_factory(cfg)

    print(f"[run_rollout] task={args.task} guide={args.guide} wandb={wandb_name} "
          f"n_init={cfg.eval.n_init_states} try_times={cfg.eval.try_times} "
          f"n_envs={cfg.eval.n_envs} device={device}")
    print(f"[run_rollout] base_dp = {args.base_dp_ckpt}")
    print(f"[run_rollout] VIB     = {args.vib_ckpt}")
    print(f"[run_rollout] core    = {args.core_hdf5}")
    print(f"[run_rollout] output  = {args.output}")

    collector = RolloutCollector(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
        env_factory=env_factory, device=device, guided=guided, wandb_run=wandb_run,
    )
    try:
        trajs = collector.run(args.base_dp_ckpt,
                              vib_ckpt=args.vib_ckpt if guided else None)
        if not trajs:
            raise RuntimeError(
                f"[run_rollout] 0 successful trajs in "
                f"{cfg.eval.n_init_states}x{cfg.eval.try_times} -- nothing "
                "written. Check base DP quality / horizon.")
        os.makedirs(out_dir, exist_ok=True)
        aug_mask_key = (args.aug_mask_key
                        or cfg.self_improvement.get("scout_aug_mask", "scout_aug"))
        write_rollouts_to_hdf5(
            cfg.dataset.path, args.output, trajs,
            core_filter_key=cfg.dataset.core_filter_key,
            aug_mask_key=aug_mask_key,
        )
        print(f"\n[run_rollout] DONE. {len(trajs)} successful trajs "
              f"(guide={args.guide}) -> {args.output}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
