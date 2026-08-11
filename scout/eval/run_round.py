"""SCOUT single-round self-improvement launcher (SOE protocol; real-env entry).

One round of (exactly the SOE single-round protocol, with SCOUT guidance):

  1. frozen base DP0 baseline       -- N=100 init states x 1 try  -> success_rate.
  2. SCOUT classifier-guided        -- up to `try_times`=5 retries per FAILED init
     exploration                       state; one fresh z~N(0,I) locked PER ROLLOUT
                                       and held across all its chunks (scout_design
                                       §1; distinct from SOE's per-chunk resample).
                                       -> Pass@5.
  3. collect ALL successful exploration rollouts (SOE: no cap).
  4. merge (core hdf5 + successes) -> augmented hdf5; retrain a DP with wandb
     name ``SCOUT-baseDP-{task}-exp1``.

This wires ``SelfImprovementLoop`` with the REAL factories (make_lpb_dp_factory
/ make_scout_vib_factory / make_default_env_factory). The loop module's own
``__main__`` only exercises the mock dry-run.

Note on the round/retrain seam: ``SelfImprovementLoop.run(num_rounds=1)``
deliberately SKIPS the retrain after the last round (it retrains *between*
rounds). So this launcher runs the loop for the exploration + metrics only,
then invokes the retrain_fn itself -- yielding exactly one exploration + one
retrain (what the SOE single-round protocol wants).

Server usage (venv active, wandb env sourced, cwd = repo root):
  .venv/bin/python -m scout.eval.run_round \\
      --config configs/eval_lift.yaml --task lift \\
      --base-dp-ckpt <.../580.ckpt> \\
      --vib-ckpt    <.../scout_vib.ckpt> \\
      --core-hdf5   <.../image_v141_abs_core10.hdf5> \\
      --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import os
from typing import List

import torch
from easydict import EasyDict

from scout.eval.self_improvement import (
    SelfImprovementLoop,
    _write_augmented_hdf5,
    load_cfg,
    make_default_env_factory,
    make_lpb_dp_factory,
    make_scout_vib_factory,
)


# --------------------------------------------------------------------------- #
# retrain callback (augmented hdf5 + LPB train.py with SCOUT wandb name)
# --------------------------------------------------------------------------- #
def make_round_retrain_fn(log_root: str, wandb_name: str,
                          num_epochs: int, cuda_visible_devices):
    """retrain_fn that writes the augmented hdf5 (core + rollouts) and shells
    out to LPB ``train.py`` with the SCOUT wandb run name (``logging.name``) and
    project ``scout-base-dp``.

    Mirrors ``self_improvement.default_retrain_fn_factory`` but injects the
    wandb-name override (``SCOUT-baseDP-{task}-exp1``) so the exp1 run is
    distinguishable from the E0 base DP, and pins ``num_epochs`` (default 600 =
    the base DP budget, for a fair base-vs-explore comparison).
    """
    def retrain_fn(cfg: EasyDict, round_idx: int,
                   successful_rollouts: List[dict], prev_dp_ckpt: str) -> str:
        from scout.train_base_dp import train

        core_path = cfg.dataset.path
        round_dir = os.path.join(log_root, f"round_{round_idx}")
        os.makedirs(round_dir, exist_ok=True)
        new_path = os.path.join(round_dir, "augmented.hdf5")
        _write_augmented_hdf5(
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
                "logging.name": wandb_name,       # SCOUT-baseDP-<task>-exp1
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
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="configs/eval_<task>.yaml")
    p.add_argument("--task", required=True, help="lift | can | square (wandb name)")
    p.add_argument("--base-dp-ckpt", required=True, help="E0 base DP ckpt (= DP0)")
    p.add_argument("--vib-ckpt", required=True, help="scout_vib.ckpt (Step 1)")
    p.add_argument("--core-hdf5", required=True,
                   help="core hdf5 (env_meta source + augmented-write base)")
    p.add_argument("--n-init-states", type=int, default=None,
                   help="override cfg.eval.n_init_states (smoke: 2)")
    p.add_argument("--try-times", type=int, default=None,
                   help="override cfg.eval.try_times (smoke: 1)")
    p.add_argument("--num-epochs", type=int, default=600,
                   help="retrain epochs (default 600 = base DP, fair compare)")
    p.add_argument("--cuda-visible-devices", default=None,
                   help="GPU id for exploration + retrain (e.g. 0)")
    p.add_argument("--wandb-name", default=None,
                   help="default SCOUT-baseDP-{task}-exp1")
    p.add_argument("--log-root", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--force-explore-all", action="store_true",
                   help="smoke-only: guided exploration on ALL init states "
                        "(ignores only_failed_of) so a tiny smoke exercises the "
                        "guided + write-back + retrain path even if baseline "
                        "solves everything. Real run leaves this off.")
    args = p.parse_args()

    cfg = load_cfg(args.config)
    # resolve run-specific paths (config may still carry <TBD> placeholders).
    cfg.base_dp.initial_ckpt_path = args.base_dp_ckpt
    cfg.vib.ckpt_path = args.vib_ckpt
    cfg.vib.base_dp_ckpt = args.base_dp_ckpt
    cfg.dataset.path = args.core_hdf5
    if args.n_init_states is not None:
        cfg.eval.n_init_states = int(args.n_init_states)
    if args.try_times is not None:
        cfg.eval.try_times = int(args.try_times)

    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    wandb_name = args.wandb_name or f"SCOUT-baseDP-{args.task}-exp1"
    log_root = args.log_root or f"data/outputs/scout_round_{args.task}"

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    dp_factory = make_lpb_dp_factory(device)
    scout_vib_factory = make_scout_vib_factory(cfg, device)
    env_factory = make_default_env_factory(cfg)
    retrain_fn = make_round_retrain_fn(log_root, wandb_name, args.num_epochs,
                                       args.cuda_visible_devices)

    print(f"[run_round] task={args.task} wandb={wandb_name} "
          f"n_init={cfg.eval.n_init_states} try_times={cfg.eval.try_times} "
          f"epochs={args.num_epochs} device={device}")
    print(f"[run_round] DP0 = {args.base_dp_ckpt}")
    print(f"[run_round] VIB = {args.vib_ckpt}")
    print(f"[run_round] core = {args.core_hdf5}")

    loop = SelfImprovementLoop(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
        env_factory=env_factory, retrain_fn=retrain_fn, device=device,
        force_explore_all=args.force_explore_all,
    )
    # ONE exploration round (loop.run(1) does baseline + guided exploration +
    # metrics; it skips retrain after the last round -- we run it ourselves).
    history = loop.run(num_rounds=1)

    print("\n=== ROUND METRICS ===")
    for h in history:
        print(h)

    successful = loop.accumulated_rollouts
    if not successful:
        raise RuntimeError(
            "[run_round] 0 successful exploration rollouts -- nothing to retrain "
            "on. Guidance likely produced no usable data (check ‖∂μ/∂a‖ liveness "
            "+ β). Aborting before retrain.")

    # the single retrain: merge core + successes, train SCOUT-baseDP-<task>-exp1.
    new_ckpt = retrain_fn(cfg, 0, successful, cfg.base_dp.initial_ckpt_path)
    print(f"\n[run_round] DONE. retrained DP -> {new_ckpt}")


if __name__ == "__main__":
    main()
