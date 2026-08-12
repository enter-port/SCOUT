"""SCOUT multi-round self-improvement launcher (SOE protocol; real-env entry).

The full self-improvement loop: per round, :class:`SelfImprovementLoop` composes
:class:`EvalPipeline` (base-DP metrics) + :class:`RolloutCollector` (VIB-guided
collection over ALL init states), then retrains DP_{i+1} on the augmented data.

  1. **eval**      -- EvalPipeline: N init states x try_times -> success_rate,
                      jerk, base_pass_at_5.
  2. **collect**   -- RolloutCollector (guided): VIB-guided exploration over ALL
                      N init states x try_times -> every successful traj kept.
  3. **retrain**   -- merge (core hdf5 + successes) -> augmented hdf5; retrain
                      a DP with wandb name ``DP-{task}-SCOUT-1``.

For standalone eval (metrics only, no collection / retrain) use
``run_eval.py``. For standalone rollout collection (guided or unguided, no
metrics / no retrain) use ``run_rollout.py``. This launcher is the ONLY entry
that does the full eval + collect + retrain cycle.

Note on the round/retrain seam: ``SelfImprovementLoop.run()`` deliberately
SKIPS the retrain after the last round (it retrains *between* rounds). So this
launcher runs the loop for the eval + collection, then invokes the retrain_fn
itself -- yielding exactly the final DP_{n} retrain on all accumulated rollouts.

Server usage (venv active, wandb env sourced, cwd = repo root):
  .venv/bin/python -m scout.eval.run_round \\
      --config configs/eval_square.yaml --task square \\
      --base-dp-ckpt <.../580.ckpt> \\
      --vib-ckpt    <.../scout_vib.ckpt> \\
      --core-hdf5   <.../image_v141_abs_core20.hdf5> \\
      --num-rounds 1 --cuda-visible-devices 0

Parallelism: ``cfg.eval.n_envs`` (default 50) drives the vectorized rollout. A
wandb run (project ``scout-eval``, name ``DP-{task}-SCOUT-1``) streams live
``eval/baseline_*`` + ``explore/*`` + ``round/*``. ``--no-wandb`` disables.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List

import torch
from easydict import EasyDict

from scout.eval.factories import (
    load_cfg,
    make_default_env_factory,
    make_lpb_dp_factory,
    make_scout_vib_factory,
)
from scout.eval.self_improvement import SelfImprovementLoop


def _to_plain(d: Any) -> Any:
    """Recursively convert EasyDict -> plain dict/list (wandb-safe config)."""
    if isinstance(d, dict):
        return {k: _to_plain(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_to_plain(x) for x in d]
    return d


# --------------------------------------------------------------------------- #
# retrain callback (augmented hdf5 + LPB train.py with SCOUT wandb name)
# --------------------------------------------------------------------------- #
def make_round_retrain_fn(log_root: str, wandb_name: str,
                          num_epochs: int, cuda_visible_devices):
    """retrain_fn that writes the augmented hdf5 (core + rollouts) and shells
    out to LPB ``train.py`` with the SCOUT wandb run name (``logging.name``) and
    project ``scout-base-dp``.

    Mirrors ``self_improvement.default_retrain_fn_factory`` but injects the
    wandb-name override (``DP-{task}-SCOUT-1``) so the SCOUT-1 run is
    distinguishable from the base DP (``DP-{task}-base``), and pins ``num_epochs``
    (default 600 = the base DP budget, for a fair base-vs-explore comparison).
    """
    def retrain_fn(cfg: EasyDict, round_idx: int,
                   successful_rollouts: List[dict], prev_dp_ckpt: str) -> str:
        from scout.train_base_dp import train
        from scout.eval.hdf5_writer import write_rollouts_to_hdf5

        core_path = cfg.dataset.path
        round_dir = os.path.join(log_root, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)
        new_path = os.path.join(round_dir, "augmented.hdf5")
        write_rollouts_to_hdf5(
            core_path, new_path, successful_rollouts,
            core_filter_key=cfg.dataset.core_filter_key,
            aug_mask_key=cfg.self_improvement.scout_aug_mask,
        )
        print(f"[run_round] wrote augmented hdf5 "
              f"({len(successful_rollouts)} rollouts + core) -> {new_path}")
        new_ckpt = train(
            config_name=cfg.base_dp.config_name,
            config_dir=cfg.base_dp.config_dir,
            dataset_path=new_path,
            train_filter_key=cfg.self_improvement.scout_aug_mask,
            log_dir=round_dir,
            num_epochs=num_epochs,
            extra_overrides={
                "training.resume": False,        # from scratch on augmented data
                "logging.name": wandb_name,       # DP-<task>-SCOUT-1
                "logging.project": "scout-base-dp",
            },
            cuda_visible_devices=cuda_visible_devices,
        )
        return new_ckpt

    return retrain_fn


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/eval_<task>.yaml")
    p.add_argument("--task", required=True, help="lift | can | square (wandb name)")
    p.add_argument("--base-dp-ckpt", required=True, help="E0 base DP ckpt (= DP0)")
    p.add_argument("--vib-ckpt", required=True,
                   help="scout_vib.ckpt (Step 1); required for guided exploration")
    p.add_argument("--core-hdf5", required=True,
                   help="core hdf5 (env_meta source + augmented-write base)")
    p.add_argument("--num-rounds", type=int, default=1,
                   help="self-improvement rounds (default 1; loop retrains "
                        "between rounds, this launcher does the final retrain)")
    p.add_argument("--n-init-states", type=int, default=None,
                   help="override cfg.eval.n_init_states (default 100)")
    p.add_argument("--try-times", type=int, default=None,
                   help="override cfg.eval.try_times (default 5)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="override cfg.eval.n_envs (parallel env count; "
                        "default 50, 1 -> sequential fallback)")
    p.add_argument("--num-epochs", type=int, default=600,
                   help="retrain epochs (default 600 = base DP, fair compare)")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="GPU id for exploration + retrain (e.g. 0)")
    p.add_argument("--wandb-name", default=None, help="default DP-{task}-SCOUT-1")
    p.add_argument("--wandb-project", default=None,
                   help="override cfg.wandb.project (default scout-eval)")
    p.add_argument("--wandb-dir", default=None, help="wandb run dir (default log_root)")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb live logging")
    p.add_argument("--log-root", default=None)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    cfg = load_cfg(args.config)
    cfg.base_dp.initial_ckpt_path = args.base_dp_ckpt
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

    # CUDA_VISIBLE_DEVICES MUST be set BEFORE torch.device() -- otherwise the
    # device would be pinned to physical GPU 0 regardless of the requested id.
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    wandb_name = args.wandb_name or f"DP-{args.task}-SCOUT-1"
    log_root = args.log_root or f"data/outputs/scout_round_{args.task}"

    # ---- wandb (live eval + exploration progress) ----------------------- #
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
                dir=args.wandb_dir or log_root,
                tags=list(wcfg.get("tags", ["step2", "self-improvement"])) + [args.task],
            )
            print(f"[run_round] wandb: project={wandb_run.project} name={wandb_name}")
        except Exception as e:  # wandb optional -- never block the run on it
            print(f"[run_round] wandb disabled (init failed: {e})")
            wandb_run = None
    else:
        print("[run_round] wandb disabled (--no-wandb / cfg.wandb.use_wandb=false)")

    dp_factory = make_lpb_dp_factory(device)
    scout_vib_factory = make_scout_vib_factory(cfg, device)
    env_factory = make_default_env_factory(cfg)
    retrain_fn = make_round_retrain_fn(log_root, wandb_name, args.num_epochs,
                                       args.cuda_visible_devices)

    print(f"[run_round] task={args.task} wandb={wandb_name} "
          f"rounds={args.num_rounds} n_init={cfg.eval.n_init_states} "
          f"try_times={cfg.eval.try_times} n_envs={cfg.eval.n_envs} "
          f"epochs={args.num_epochs} device={device}")
    print(f"[run_round] DP0 = {args.base_dp_ckpt}")
    print(f"[run_round] VIB = {args.vib_ckpt}")
    print(f"[run_round] core = {args.core_hdf5}")

    loop = SelfImprovementLoop(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
        env_factory=env_factory, retrain_fn=retrain_fn, device=device,
        wandb_run=wandb_run,
    )
    try:
        # The loop does eval + guided collect per round, retrains BETWEEN rounds
        # (skips the retrain after the last round). We run the loop, then do the
        # final retrain ourselves on ALL accumulated rollouts -> DP-<task>-SCOUT-1.
        history = loop.run(num_rounds=args.num_rounds)

        print("\n=== ROUND METRICS ===")
        for h in history:
            print(h)

        successful = loop.accumulated_rollouts
        if not successful:
            raise RuntimeError(
                "[run_round] 0 successful rollouts -- nothing to retrain on. "
                "Guidance likely produced no usable data (check ‖∂μ/∂a‖ liveness "
                "+ β). Aborting before retrain.")

        # the final retrain: merge core + ALL successes, train DP-<task>-SCOUT-1.
        new_ckpt = retrain_fn(cfg, args.num_rounds - 1, successful,
                              cfg.base_dp.initial_ckpt_path)
        print(f"\n[run_round] DONE. retrained DP -> {new_ckpt}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
