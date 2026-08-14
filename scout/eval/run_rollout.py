"""SCOUT SOE rollout launcher (steps 2 -> 3 -> 4 of the self-improvement round).

Given a base-DP ckpt (and, for guided exploration, a SCOUT VIB/dyn ckpt) plus a
core hdf5, this runs the SOE rollout flow:

  step 2  roll the base DP once over N (default 100) random init states (pure
          base path) -> ``success_rate`` (first-try) + the FAILED init set.
  step 3  re-roll the FAILED inits ``try_times`` (default 5) times:
            ``--guide dyn`` : VIB-guided exploration (``predict_action_dyn_guided``)
            ``--guide off`` : plain base-DP retry (``predict_action``; baseline)
          -> successful trajectories (for retrain data) + ``pass@5`` +
          ``avg_jerk`` (over EVERY exploration trajectory, success + failure).
  step 4  merge the successful trajectories with the core hdf5.

Outputs (per the data convention, under ``--output-dir`` = data/{task}/rollout/):
  * ``{task}_exp{N}.hdf5``        -- core + successful rollouts (retrain input).
  * ``{task}_success_exp{N}.hdf5``-- successful rollouts only (archive).
  * ``{task}_rollout_exp{N}.json``-- success_rate / pass@5 / avg_jerk / counts.

wandb logs three metrics whose x-axis is the completed-init-count of their
phase (``eval/success_rate`` vs step-2 init count; ``rollout/pass@5`` and
``rollout/avg_jerk`` vs failed-init explore-completed count), via
``wandb.define_metric(step_metric=...)``.

Server usage (venv active, wandb env sourced, cwd = repo root):
  .venv/bin/python -m scout.eval.run_rollout \\
      --config configs/eval_can.yaml --task can --exp-num 1 \\
      --base-dp-ckpt data/can/train/DP-can-base/checkpoints/580.ckpt \\
      --core-hdf5   data/can/rollout/can_core.hdf5 \\
      --guide dyn --vib-ckpt data/can/train/can-SCOUT/scout_vib.ckpt \\
      --cuda-visible-devices 0
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch

from scout.eval.rollout_pipeline import RolloutPipeline
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
    # ---- inputs ----
    p.add_argument("--config", required=True, help="configs/eval_<task>.yaml")
    p.add_argument("--task", required=True, help="lift | can | square | transport")
    p.add_argument("--base-dp-ckpt", required=True, help="base DP ckpt to roll out")
    p.add_argument("--vib-ckpt", default=None,
                   help="scout_vib.ckpt; required for --guide dyn only")
    p.add_argument("--core-hdf5", default=None,
                   help="core hdf5 (env_meta source + augmented-write base). "
                        "Required for full rollout; optional for --success-only "
                        "(falls back to base-DP config task.dataset_path for env_meta).")
    # ---- exploration mode ----
    p.add_argument("--guide", choices=["dyn", "off"], default="off",
                   help="'dyn' = VIB-guided exploration on failed inits (needs "
                        "--vib-ckpt); 'off' (default) = plain base-DP retry "
                        "(baseline).")
    p.add_argument("--success-only", action="store_true",
                   help="only run step2 (base-path success_rate on N seed-fixed "
                        "inits); skip explore (step3) + merge (step4). No VIB / "
                        "no hdf5 needed -- pure DP success-rate eval of any ckpt.")
    # ---- outputs (naming convention: {task}_{exp,success_exp,rollout_exp}{N}) ----
    p.add_argument("--exp-num", type=int, default=1,
                   help="exploration round number N for output naming (default 1)")
    p.add_argument("--output-dir", default=None,
                   help="output dir (default data/{task}/rollout/)")
    p.add_argument("--output-merged", default=None,
                   help="explicit merged-hdf5 path (overrides {task}_exp{N}.hdf5)")
    p.add_argument("--output-success", default=None,
                   help="explicit success-only hdf5 path (overrides "
                        "{task}_success_exp{N}.hdf5)")
    p.add_argument("--output-json", default=None,
                   help="explicit json path (default log/{task}_{tag}_rollout_exp{N}.json; "
                        "for --success-only: log/{wandb_name}.json)")
    p.add_argument("--aug-mask-key", default=None,
                   help="mask key written to the merged hdf5 selecting core + "
                        "rollouts (default cfg.self_improvement.scout_aug_mask)")
    # ---- overrides ----
    p.add_argument("--n-init-states", type=int, default=None,
                   help="override cfg.eval.n_init_states (default 100)")
    p.add_argument("--try-times", type=int, default=None,
                   help="override cfg.eval.try_times (default 5)")
    p.add_argument("--n-envs", type=int, default=None,
                   help="override cfg.eval.n_envs (parallel env count; default 50)")
    p.add_argument("--seed", type=int, default=42,
                   help="base seed for the N init scenes (init i uses seed+i; "
                        "default 42 -> seeds 42..141). Same seed across runs -> "
                        "same 100 scenes -> controlled DP-vs-SCOUT comparison.")
    p.add_argument("--cuda-visible-devices", default=None, help="GPU id (e.g. 0)")
    p.add_argument("--wandb-name", default=None,
                   help="default DP-{task}-{SCOUT|base}-rollout-exp{N}")
    p.add_argument("--wandb-project", default=None,
                   help="override cfg.wandb.project (default scout-eval)")
    p.add_argument("--wandb-dir", default=None,
                   help="wandb run dir (default --output-dir)")
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
    cfg.eval.seed = args.seed          # base seed for init scenes (pipeline reads it)

    guided = (args.guide == "dyn") and not args.success_only
    if guided and (args.vib_ckpt is None
                   or str(getattr(cfg.vib, "ckpt_path", "")).startswith("<")):
        raise SystemExit("[run_rollout] --guide dyn needs --vib-ckpt (SCOUT VIB "
                         "ckpt for guided exploration). --guide off does not.")
    if not args.success_only and args.core_hdf5 is None:
        raise SystemExit("[run_rollout] --core-hdf5 required for full rollout "
                         "(only --success-only may omit it).")

    # CUDA_VISIBLE_DEVICES MUST be set BEFORE torch.device().
    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = (torch.device(args.device) if args.device
              else torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    tag = "SCOUT" if guided else "DP"          # a in {task}_{a}_... : SCOUT | DP
    if args.success_only:
        wandb_name = args.wandb_name or f"{args.task}_eval_exp{args.exp_num}"
    else:
        wandb_name = args.wandb_name or f"DP-{args.task}-{tag}-rollout-exp{args.exp_num}"

    # ---- output paths (convention: {task}_{a}_{...}) ----
    out_dir = args.output_dir or os.path.join("data", args.task, "rollout")
    os.makedirs(out_dir, exist_ok=True)
    log_dir = os.path.join(out_dir, "log")
    os.makedirs(log_dir, exist_ok=True)
    merged_path = args.output_merged or os.path.join(
        out_dir, f"{args.task}_{tag}_exp{args.exp_num}.hdf5")
    success_path = args.output_success or os.path.join(
        out_dir, f"{args.task}_{tag}_success_exp{args.exp_num}.hdf5")
    if args.success_only:
        json_path = args.output_json or os.path.join(log_dir, f"{wandb_name}.json")
    else:
        json_path = os.path.join(log_dir, f"{args.task}_{tag}_rollout_exp{args.exp_num}.json")

    # ---- wandb (live progress; x-axis = completed-init-count) ------------ #
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
            # x-axis = completed-init-count of each phase.
            wandb.define_metric("eval_init_done")
            wandb.define_metric("explore_init_done")
            wandb.define_metric("eval/success_rate", step_metric="eval_init_done")
            wandb.define_metric("rollout/pass@5", step_metric="explore_init_done")
            wandb.define_metric("rollout/avg_jerk", step_metric="explore_init_done")
            print(f"[run_rollout] wandb: project={wandb_run.project} name={wandb_name}")
        except Exception as e:  # wandb optional -- never block the run on it
            print(f"[run_rollout] wandb disabled (init failed: {e})")
            wandb_run = None
    else:
        print("[run_rollout] wandb disabled (--no-wandb / cfg.wandb.use_wandb=false)")

    def on_progress(phase: str, payload: dict,
                    baseline_solved: int = 0, n_total: int = 0):
        """Engine -> wandb: log the three metrics vs their phase's init count."""
        if wandb_run is None:
            return
        if phase == "eval":
            completed = int(payload.get("completed", 0))
            succ = int(payload.get("successes", 0))
            wandb_run.log({
                "eval/success_rate": succ / max(completed, 1),
                "eval_init_done": completed,
            })
        elif phase == "explore":
            eid = int(payload.get("explore_init_done", 0))
            solved_failed = int(payload.get("solved_failed", 0))
            pass5 = (baseline_solved + solved_failed) / max(n_total, 1)
            jn = int(payload.get("jerk_n", 0))
            js = float(payload.get("jerk_sum", 0.0))
            avg_jerk = js / jn if jn > 0 else 0.0
            wandb_run.log({
                "rollout/pass@5": pass5,
                "rollout/avg_jerk": avg_jerk,
                "explore_init_done": eid,
            })

    dp_factory = make_lpb_dp_factory(device)
    scout_vib_factory = make_scout_vib_factory(cfg, device) if guided else None
    env_factory = make_default_env_factory(cfg)

    print(f"[run_rollout] task={args.task} guide={args.guide} wandb={wandb_name} "
          f"n_init={cfg.eval.n_init_states} try_times={cfg.eval.try_times} "
          f"n_envs={cfg.eval.n_envs} device={device}")
    print(f"[run_rollout] base_dp = {args.base_dp_ckpt}")
    print(f"[run_rollout] VIB     = {args.vib_ckpt}")
    print(f"[run_rollout] core    = {args.core_hdf5}")
    print(f"[run_rollout] merged  = {merged_path}")
    print(f"[run_rollout] success = {success_path}")
    print(f"[run_rollout] json    = {json_path}")

    pipeline = RolloutPipeline(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
        env_factory=env_factory, device=device, guided=guided,
    )
    try:
        result = pipeline.run(
            args.base_dp_ckpt,
            vib_ckpt=args.vib_ckpt if guided else None,
            on_progress=on_progress if wandb_run is not None else None,
            success_only=args.success_only,
        )
        metrics = result["metrics"]

        if args.success_only:
            # step2-only metrics; no hdf5; simplified json (no pass@5 / jerk_explore)
            summary = {
                "task": args.task, "mode": "success_only",
                "dp_ckpt": args.base_dp_ckpt,
                "n_init_states": int(cfg.eval.n_init_states),
                "seed": args.seed,
                "success_rate": metrics["success_rate"],
                "jerk_baseline": metrics["jerk_baseline"],
                "baseline_solved": metrics["baseline_solved"],
                "n_failed": metrics["n_failed"],
                "failed_init_indices": metrics["failed_init_indices"],
            }
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)
            if wandb_run is not None:
                N = int(cfg.eval.n_init_states)
                wandb_run.log({"eval/success_rate": metrics["success_rate"],
                               "eval_init_done": N})
            print(f"\n[run_rollout] SUCCESS-ONLY DONE. "
                  f"success_rate={metrics['success_rate']:.3f} "
                  f"({metrics['baseline_solved']}/{int(cfg.eval.n_init_states)}) "
                  f"jerk_baseline={metrics['jerk_baseline']:.4f}")
            print(f"[run_rollout] json -> {json_path}")
        else:
            trajs = result["trajs"]

            # ---- step 4: merge (core + rollouts) + success-only archive ------- #
            aug_mask_key = (args.aug_mask_key
                            or cfg.get("self_improvement", {}).get("scout_aug_mask",
                                                                   "scout_aug"))
            write_rollouts_to_hdf5(
                cfg.dataset.path, merged_path, trajs,
                core_filter_key=cfg.dataset.core_filter_key,
                aug_mask_key=aug_mask_key, include_core=True,
            )
            if trajs:
                write_rollouts_to_hdf5(
                    cfg.dataset.path, success_path, trajs,
                    core_filter_key=cfg.dataset.core_filter_key,
                    aug_mask_key=aug_mask_key, include_core=False,
                )
            else:
                print("[run_rollout] 0 successful trajs -- skipping success-only hdf5")

            # ---- final wandb points (converged values at full init counts) ---- #
            if wandb_run is not None:
                N = int(cfg.eval.n_init_states)
                wandb_run.log({"eval/success_rate": metrics["success_rate"],
                               "eval_init_done": N})
                wandb_run.log({"rollout/pass@5": metrics["pass_at_5"],
                               "rollout/avg_jerk": metrics["avg_jerk"],
                               "explore_init_done": metrics["n_failed"]})

            # ---- JSON summary ------------------------------------------------- #
            summary = {
                "task": args.task, "mode": args.guide,
                "exp_num": args.exp_num,
                "dp_ckpt": args.base_dp_ckpt, "vib_ckpt": args.vib_ckpt,
                "core_hdf5": args.core_hdf5,
                "n_init_states": int(cfg.eval.n_init_states),
                "try_times": int(cfg.eval.try_times),
                "seed": args.seed,
                "guided": int(guided),
                "success_rate": metrics["success_rate"],
                "pass_at_5": metrics["pass_at_5"],
                "avg_jerk": metrics["avg_jerk"],
                "jerk_baseline": metrics["jerk_baseline"],
                "baseline_solved": metrics["baseline_solved"],
                "n_failed": metrics["n_failed"],
                "exploration_rescued": metrics["exploration_rescued"],
                "collected_trajs": metrics["collected_trajs"],
                "failed_init_indices": metrics["failed_init_indices"],
                "outputs": {"merged": merged_path, "success_only": success_path},
            }
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)

            print(f"\n[run_rollout] DONE. success_rate={metrics['success_rate']:.3f} "
                  f"pass@5={metrics['pass_at_5']:.3f} avg_jerk={metrics['avg_jerk']:.4f} "
                  f"collected={metrics['collected_trajs']}")
            print(f"[run_rollout] merged  -> {merged_path}")
            print(f"[run_rollout] success -> {success_path}")
            print(f"[run_rollout] json    -> {json_path}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
