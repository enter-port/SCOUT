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

Outputs (per the data convention, under ``--output-dir`` = data/{task}/rollout/;
``{tag}`` = SCOUT for ``--guide dyn``, DP for ``--guide off``):
  * ``{task}_{tag}_success_exp{N}.hdf5`` -- core + successful EXPLORATION
    rollouts (DP-retrain input; baseline first-try successes NOT included).
  * ``{task}_{tag}_all_exp{N}.hdf5``     -- core + ALL trajectories of the
    round: every one of the N baseline (step-2) rollouts plus all ``try_times``
    exploration trajectories per failed init, success AND failure
    (dyn/VIB-retrain input -- diversified transitions vs z-exploration drift).
  * ``log/{task}_{tag}_rollout_exp{N}.json`` -- success_rate / pass@5 / counts.

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
    p.add_argument("--guide", default="off",
                   help="'dyn' = VIB-guided exploration (z ~ prior per rollout; "
                        "needs --vib-ckpt); 'expert' = expert z-bank guidance "
                        "(z* = nearest core-data bank entry per action chunk; "
                        "needs --vib-ckpt + --core-hdf5); 'novelty' = entropy "
                        "cost, minimize KDE density of the candidate's encoder "
                        "code in the scene's visited-code set (方案二; needs "
                        "--vib-ckpt); 'atypical' = entropy cost, maximize "
                        "KL to the policy's own unguided-intent encoder "
                        "(方案三; needs --vib-ckpt); 'combo' = novelty + "
                        "atypical summed (方案二+三; needs --vib-ckpt); "
                        "'shell' = 方案A: per-retry random target posterior on "
                        "the kappa-shell of the intent posterior (SOE spray in "
                        "cost form; needs --vib-ckpt); "
                        "'rand_<idea>' = entropy-random-dev registry plugin "
                        "(scout/guidance/rand_costs/; needs --vib-ckpt); "
                        "'particle' = entropy cost + parallel inter-particle "
                        "repulsion (a scene's retries launch as ONE slot "
                        "group and repel each other's behaviour codes while "
                        "generating; needs --vib-ckpt); "
                        "'off' (default) = plain base-DP rollout (baseline).")
    p.add_argument("--pg-lambda", type=float, default=1.0,
                   help="particle: repulsion weight (lambda). Calibrate via "
                        "smoke so the repulsion inject magnitude lands at "
                        "0.5-1x the entropy-cost magnitude (2026-08-30 plan).")
    p.add_argument("--pg-h-scale", type=float, default=1.0,
                   help="particle: bandwidth multiplier c -- h = c * median "
                        "of the group's pairwise mu distances (median "
                        "heuristic; per-group, per-replan).")
    p.add_argument("--pg-start", type=int, default=0,
                   help="particle: 0-based denoise-step index at which the "
                        "repulsion switches ON (timing ablation G1/G2/G3 = "
                        "0/50/90; before it the loss is bit-identical to "
                        "--guide atypical).")
    p.add_argument("--failed-set-json", default=None,
                   help="rescue mode: load the FROZEN failure set from this "
                        "json (explore-only -- the eval phase is skipped and "
                        "pass@k is measured on exactly the recorded failed "
                        "inits).")
    p.add_argument("--save-failed-set", default=None,
                   help="rescue mode: save the baseline run's failed inits to "
                        "this json (run once with the base DP, then reuse for "
                        "every experiment via --failed-set-json).")
    p.add_argument("--novelty-h", type=float, default=5.0,
                   help="novelty KDE kernel width floor, in units of the "
                        "encoder's running per-dim sigma; width also adapts "
                        "to the buffer spread (default 5.0 -- quadratic-"
                        "repulsion regime, constant force, no saturation)")
    p.add_argument("--novelty-sample-z", type=int, default=0,
                   help="novelty: evaluate the code at mu+sigma*eps with a "
                        "per-chunk fixed eps (1, default) or at mu (0)")
    p.add_argument("--atypical-cap", type=float, default=10.0,
                   help="atypical: cap on the KL bonus in nats (default 10)")
    p.add_argument("--shell-kappa", type=float, default=2.5,
                   help="shell (方案A): target-shell radius in nats -- the "
                        "random target posterior sits exactly this many nats "
                        "from the intent posterior (default 2.5)")
    p.add_argument("--rand-kwargs", default="",
                   help="rand_<idea> plugins: comma-separated k=v pairs "
                        "merged into entropy_kwargs (floats auto-parsed); "
                        "idea-specific knobs WITHOUT touching shared code")
    p.add_argument("--combo-nov-weight", type=float, default=1.0,
                   help="combo: weight of the novelty cost term (default 1.0; "
                        "0.5 at scale 2.0 keeps its force at the h0.5/s1.0 "
                        "solo calibration)")
    p.add_argument("--combo-att-weight", type=float, default=1.0,
                   help="combo: weight of the atypical cost term (default 1.0)")
    p.add_argument("--success-only", action="store_true",
                   help="only run step2 (base-path success_rate on N seed-fixed "
                        "inits); skip explore (step3) + merge (step4). No VIB / "
                        "no hdf5 needed -- pure DP success-rate eval of any ckpt.")
    p.add_argument("--bank-hdf5", default=None,
                   help="expert z-bank source hdf5 (default: --core-hdf5). "
                        "e.g. a round's success_accum.hdf5 -- the exact data "
                        "that trained the rollout DP.")
    # ---- outputs (naming: {task}_{tag}_{success_exp,all_exp,rollout_exp}{N}) ----
    p.add_argument("--exp-num", type=int, default=1,
                   help="exploration round number N for output naming (default 1)")
    p.add_argument("--output-dir", default=None,
                   help="output dir (default data/{task}/rollout/)")
    p.add_argument("--output-success", default=None,
                   help="explicit success hdf5 path (core + successful "
                        "exploration rollouts; overrides "
                        "{task}_{tag}_success_exp{N}.hdf5)")
    p.add_argument("--output-all", default=None,
                   help="explicit all hdf5 path (core + every trajectory: N "
                        "baseline + try_times-per-failed-init exploration; "
                        "overrides {task}_{tag}_all_exp{N}.hdf5)")
    p.add_argument("--output-json", default=None,
                   help="explicit json path (default log/{task}_{tag}_rollout_exp{N}.json; "
                        "for --success-only: log/{wandb_name}.json)")
    p.add_argument("--aug-mask-key", default=None,
                   help="mask key written to both output hdf5s selecting core + "
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
    # ---- experiment2 split protocol (user 2026-08-17) --------------------- #
    p.add_argument("--eval-seed", type=int, default=None,
                   help="split mode: seed of the FIXED eval scene set "
                        "(default --seed, i.e. 42 -> 42..141, 100 scenes)")
    p.add_argument("--explore-seed", type=int, default=None,
                   help="split mode: base seed of the explore scene set; round i "
                        "uses i*1000+42 (seeds ..+499 -> 500 scenes). Passing "
                        "this switches the pipeline to the split protocol; "
                        "omitting it keeps the legacy retry-failed-inits mode.")
    p.add_argument("--explore-mode", choices=["fresh", "rescue"], default="fresh",
                   help="explore protocol (user 2026-08-23): 'fresh' (default) = "
                        "split protocol, explore rolls fresh scenes from "
                        "--explore-seed; 'rescue' = SOE protocol, explore "
                        "retries ONLY the failed eval inits (same scenes) "
                        "--explore-try-times each. DP data = successful "
                        "retries; dyn data = per failed init {successes, else "
                        "first retry}. --explore-seed/--n-explore ignored.")
    p.add_argument("--n-explore", type=int, default=500,
                   help="split mode: number of explore scenes (default 500)")
    p.add_argument("--explore-try-times", type=int, default=1,
                   help="rollouts per explore scene in fresh mode / retries per "
                        "failed eval init in rescue mode (default 1; rescue "
                        "drivers pass 5)")
    p.add_argument("--eval-only", action="store_true",
                   help="split protocol but SKIP the explore phase: run the "
                        "seed-fixed eval set once, report success_rate/jerk, "
                        "write no hdf5 (final round of a chain)")
    p.add_argument("--cuda-visible-devices", default=None, help="GPU id (e.g. 0)")
    p.add_argument("--wandb-name", default=None,
                   help="default DP-{task}-{SCOUT|base}-rollout-exp{N}")
    p.add_argument("--wandb-project", default=None,
                   help="override cfg.wandb.project (default scout-eval)")
    p.add_argument("--wandb-dir", default=None,
                   help="wandb run dir (default --output-dir)")
    p.add_argument("--no-wandb", action="store_true",
                   help="disable wandb live logging")
    p.add_argument("--wandb-minimal", action="store_true",
                   help="formal entropy experiment (2026-08-24): log ONLY "
                        "eval/success_rate + explore/pass@10; the creator also "
                        "pre-registers the DP/loss and dyn/KL-loss, dyn/mse-loss "
                        "axes for the retrain stages that resume this run")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    # --guide validation: static set + auto-discovered rand registry
    # (entropy-random-dev, 2026-08-27; free-form instead of argparse choices)
    from scout.guidance.rand_costs import REGISTRY as _RAND
    _static = ("dyn", "off", "expert", "novelty", "atypical", "combo",
               "shell", "particle")
    if not (args.guide in _static or (args.guide.startswith("rand_")
                                      and args.guide[5:] in _RAND)):
        p.error(f"--guide must be one of {_static} or rand_<idea>, "
                f"ideas available: {sorted(_RAND)} (got {args.guide!r})")
    rand_ek = {}
    for kv in filter(None, args.rand_kwargs.split(",")):
        k, _, v = kv.partition("=")
        try:
            rand_ek[k.strip()] = float(v)
        except ValueError:
            rand_ek[k.strip()] = v.strip()

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
    # split-protocol defaults may live in the config (cfg.explore.*); explicit
    # CLI --explore-seed wins. Absent section -> legacy retry-failed mode.
    # Rescue mode overrides all of this -- the config's explore section must
    # NOT force split mode when the driver asked for the SOE rescue protocol.
    _exp = getattr(cfg, "explore", None)
    if args.explore_mode == "rescue":
        args.explore_seed = None
    else:
        if args.explore_seed is None and _exp is not None \
                and getattr(_exp, "base_seed_round1", None) is not None:
            args.explore_seed = int(_exp.base_seed_round1)
        if _exp is not None:
            if getattr(_exp, "n_scenes", None) is not None and args.n_explore == 500:
                args.n_explore = int(_exp.n_scenes)
            if getattr(_exp, "try_times", None) is not None \
                    and args.explore_try_times == 1:
                args.explore_try_times = int(_exp.try_times)
    rescue_mode = args.explore_mode == "rescue"
    split_mode = (not rescue_mode) and (args.explore_seed is not None
                                        or args.eval_only)
    eval_seed = args.eval_seed if args.eval_seed is not None else args.seed

    guided = (args.guide in ("dyn", "expert", "novelty", "atypical", "combo",
                             "shell", "particle") or args.guide.startswith("rand_")
              ) and not args.success_only
    if guided and (args.vib_ckpt is None
                   or str(getattr(cfg.vib, "ckpt_path", "")).startswith("<")):
        raise SystemExit(f"[run_rollout] --guide {args.guide} needs --vib-ckpt "
                         "(SCOUT VIB ckpt). --guide off does not.")
    if guided and args.guide == "expert" and args.core_hdf5 is None:
        raise SystemExit("[run_rollout] --guide expert needs --core-hdf5 "
                         "(the expert z-bank is built from it).")
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
    success_path = args.output_success or os.path.join(
        out_dir, f"{args.task}_{tag}_success_exp{args.exp_num}.hdf5")
    all_path = args.output_all or os.path.join(
        out_dir, f"{args.task}_{tag}_all_exp{args.exp_num}.hdf5")
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
            if args.wandb_minimal:
                # formal entropy experiment (user 2026-08-24): ONLY these keys
                # ever reach this run. The creator must pre-register EVERY
                # section's axes here (2026-08-18 lesson: define_metric calls
                # from the resuming retrain processes cannot override the
                # panel config of an already-created run; explicit per-name
                # defs dispatch immediately, globs do not).
                wandb.define_metric("eval_init_done", hidden=True)
                wandb.define_metric("explore_init_done", hidden=True)
                wandb.define_metric("eval/success_rate", step_metric="eval_init_done")
                wandb.define_metric("explore/pass@10", step_metric="explore_init_done")
                wandb.define_metric("DP/epoch", hidden=True)
                wandb.define_metric("DP/loss", step_metric="DP/epoch")
                wandb.define_metric("dyn/epoch", hidden=True)
                for _n in ("KL-loss", "mse-loss"):
                    wandb.define_metric(f"dyn/{_n}", step_metric="dyn/epoch")
            elif split_mode:
                # experiment2+ layout: one wandb run per ROUND. THIS process
                # CREATES the run, so it must pre-register the metric axes of
                # EVERY section here -- define_metric calls from the later
                # DP/dyn processes (which resume this run) do NOT override the
                # panel config of an already-created run (observed 2026-08-18:
                # dyn/* panels plotted against the global _step ~2e4 instead
                # of dyn/epoch). eval/*,explore/* below; DP/*,dyn/* for the
                # retrain stages that resume this run via WANDB_RUN_ID.
                # NOTE: the retrain defs must be EXPLICIT per-name, not globs.
                # In wandb 0.28 a glob def (f"{sec}/*") is kept client-side
                # (handler._metric_globs) and only uploaded when a matching
                # key is LOGGED in the same process -- the creator never logs
                # DP/dyn rows, so a glob pre-registration never reaches the
                # backend (why the earlier f"{sec}/*" attempt didn't work).
                # Explicit defs dispatch immediately (handler.
                # _handle_defined_metric -> _dispatch_record). Names = exactly
                # the keys the retrains log (verified via historyKeys).
                _retrain_axes = {
                    "DP": ("train_loss", "lr", "global_step",
                           "train_action_mse_error"),
                    "dyn": ("latent_mse", "kl", "lr"),
                }
                for sec, names in _retrain_axes.items():
                    wandb.define_metric(f"{sec}/epoch", hidden=True)
                    for n in names:
                        wandb.define_metric(f"{sec}/{n}",
                                            step_metric=f"{sec}/epoch")
                wandb.define_metric("eval/env_done")
                wandb.define_metric("explore/env_done")
                wandb.define_metric("eval/success_rate", step_metric="eval/env_done")
                wandb.define_metric("explore/success_count", step_metric="explore/env_done")
                wandb.define_metric("explore/avg_jerk", step_metric="explore/env_done")
            else:
                # x-axis = completed-init-count of each phase.
                wandb.define_metric("eval_init_done")
                wandb.define_metric("explore_init_done")
                wandb.define_metric("eval/success_rate", step_metric="eval_init_done")
                wandb.define_metric("rollout/pass@5", step_metric="explore_init_done")
                wandb.define_metric("rollout/avg_jerk", step_metric="explore_init_done")
            print(f"[run_rollout] wandb: project={wandb_run.project} name={wandb_name} "
                  f"run_id={wandb_run.id}")
        except Exception as e:  # wandb optional -- never block the run on it
            print(f"[run_rollout] wandb disabled (init failed: {e})")
            wandb_run = None
    else:
        print("[run_rollout] wandb disabled (--no-wandb / cfg.wandb.use_wandb=false)")

    def on_progress(phase: str, payload: dict,
                    baseline_solved: int = 0, n_total: int = 0):
        """Engine -> wandb: log the metrics vs their phase's scene count."""
        if wandb_run is None:
            return
        if phase == "eval":
            completed = int(payload.get("completed", 0))
            succ = int(payload.get("successes", 0))
            if args.wandb_minimal:
                wandb_run.log({
                    "eval/success_rate": succ / max(completed, 1),
                    "eval_init_done": completed,
                })
            elif split_mode:
                wandb_run.log({
                    "eval/env_done": completed,
                    "eval/success_rate": succ / max(completed, 1),
                })
            else:
                wandb_run.log({
                    "eval/success_rate": succ / max(completed, 1),
                    "eval_init_done": completed,
                })
        elif phase == "explore":
            eid = int(payload.get("explore_init_done", 0))
            solved_failed = int(payload.get("solved_failed", 0))
            jn = int(payload.get("jerk_n", 0))
            js = float(payload.get("jerk_sum", 0.0))
            avg_jerk = js / jn if jn > 0 else 0.0
            if args.wandb_minimal:
                p10 = (baseline_solved + solved_failed) / max(n_total, 1)
                wandb_run.log({
                    "explore/pass@10": p10,
                    "explore_init_done": eid,
                })
            elif split_mode:
                wandb_run.log({
                    "explore/env_done": eid,
                    "explore/success_count": solved_failed,
                    "explore/avg_jerk": avg_jerk,
                })
            else:
                pass5 = (baseline_solved + solved_failed) / max(n_total, 1)
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
    print(f"[run_rollout] success = {success_path}")
    print(f"[run_rollout] all     = {all_path}")
    print(f"[run_rollout] json    = {json_path}")

    pipeline = RolloutPipeline(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
        env_factory=env_factory, device=device, guided=guided,
        guide_mode=args.guide if guided else "dyn",
        bank_hdf5=args.bank_hdf5,
        entropy_kwargs={"novelty_h": args.novelty_h,
                        "novelty_sample_z": bool(args.novelty_sample_z),
                        "atypical_cap": args.atypical_cap,
                        "combo_nov_weight": args.combo_nov_weight,
                        "combo_att_weight": args.combo_att_weight,
                        "shell_kappa": args.shell_kappa,
                        "shell_seed": args.seed,
                        "pg_lambda": args.pg_lambda,
                        "pg_h_scale": args.pg_h_scale,
                        "pg_start": args.pg_start,
                        **rand_ek},
        failed_set_json=args.failed_set_json,
        save_failed_set=args.save_failed_set,
    )
    try:
        result = pipeline.run(
            args.base_dp_ckpt,
            vib_ckpt=args.vib_ckpt if guided else None,
            on_progress=on_progress if wandb_run is not None else None,
            success_only=args.success_only,
            explore_seed=args.explore_seed,
            n_explore=args.n_explore,
            explore_try_times=args.explore_try_times,
            eval_only=args.eval_only,
            explore_mode=args.explore_mode,
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
            all_trajs = result.get("all_trajs", [])

            # ---- step 4: two hdf5 outputs ------------------------------------- #
            # success = core + successful EXPLORATION trajs  -> DP retrain
            # all     = core + EVERY traj of the round       -> dyn/VIB retrain
            aug_mask_key = (args.aug_mask_key
                            or cfg.get("self_improvement", {}).get("scout_aug_mask",
                                                                   "scout_aug"))
            if trajs:
                write_rollouts_to_hdf5(
                    cfg.dataset.path, success_path, trajs,
                    core_filter_key=cfg.dataset.core_filter_key,
                    aug_mask_key=aug_mask_key, include_core=True,
                )
            else:
                print("[run_rollout] 0 successful exploration trajs "
                      "-- skipping success hdf5")
            if all_trajs:
                write_rollouts_to_hdf5(
                    cfg.dataset.path, all_path, all_trajs,
                    core_filter_key=cfg.dataset.core_filter_key,
                    aug_mask_key=aug_mask_key, include_core=True,
                )
            else:
                print("[run_rollout] 0 all-trajs -- skipping all hdf5")

            # ---- final wandb points (converged values at full scene counts) -- #
            if wandb_run is not None:
                if split_mode:
                    wandb_run.log({
                        "eval/env_done": int(cfg.eval.n_init_states),
                        "eval/success_rate": metrics["success_rate"],
                    })
                    if not args.eval_only:
                        wandb_run.log({
                            "explore/env_done": metrics["explore_total"],
                            "explore/success_count": metrics["explore_solved"],
                            "explore/total": metrics["explore_total"],   # final only
                            "explore/avg_jerk": metrics["avg_jerk"],
                        })
                    # /final = cross-stage summary of the shared round-run;
                    # DP retrain adds dp_train_loss, dyn retrain adds the rest
                    final_pts = {"final/eval_success_rate": metrics["success_rate"]}
                    if not args.eval_only:
                        final_pts["final/explore_success_num"] = metrics["explore_solved"]
                    wandb_run.log(final_pts)
                else:
                    N = int(cfg.eval.n_init_states)
                    wandb_run.log({"eval/success_rate": metrics["success_rate"],
                                   "eval_init_done": N})
                    if args.wandb_minimal:
                        if "pass_at_5" in metrics:   # value = pass@(try_times)
                            wandb_run.log({
                                "explore/pass@10": metrics["pass_at_5"],
                                "explore_init_done": metrics["n_failed"]})
                    else:
                        wandb_run.log({"rollout/pass@5": metrics["pass_at_5"],
                                       "rollout/avg_jerk": metrics["avg_jerk"],
                                       "explore_init_done": metrics["n_failed"]})

            # ---- JSON summary ------------------------------------------------- #
            summary = {
                "task": args.task,
                "mode": args.guide + (":eval-only" if args.eval_only else ""),
                "exp_num": args.exp_num,
                "dp_ckpt": args.base_dp_ckpt, "vib_ckpt": args.vib_ckpt,
                "core_hdf5": args.core_hdf5,
                "n_init_states": int(cfg.eval.n_init_states),
                "try_times": int(cfg.eval.try_times),
                "seed": args.seed,
                "guided": int(guided),
                "wandb_run_id": (wandb_run.id if wandb_run is not None else None),
                "success_rate": metrics["success_rate"],
                "avg_jerk": metrics.get("avg_jerk"),
                "jerk_baseline": metrics["jerk_baseline"],
                "baseline_solved": metrics["baseline_solved"],
                "n_failed": metrics["n_failed"],
                "collected_trajs": metrics.get("collected_trajs", 0),
                "n_success_trajs": len(trajs),
                "n_all_trajs": len(all_trajs),
                "failed_init_indices": metrics.get("failed_init_indices"),
                "outputs": {"success": success_path, "all": all_path},
            }
            if args.eval_only:
                summary.update({"protocol": "eval_only", "eval_seed": eval_seed})
            elif split_mode:
                summary.update({
                    "protocol": "split",
                    "eval_seed": eval_seed,
                    "explore_seed": metrics["explore_seed"],
                    "n_explore": metrics["n_explore"],
                    "explore_try_times": metrics["explore_try_times"],
                    "explore_solved": metrics["explore_solved"],
                    "explore_total": metrics["explore_total"],
                })
            else:
                # rescue (SOE) and legacy share the retry-failed-inits metric
                # schema; only the data-selection rules differ (see pipeline).
                summary.update({
                    "protocol": "rescue" if rescue_mode else "legacy",
                    "pass_at_5": metrics["pass_at_5"],
                    "exploration_rescued": metrics["exploration_rescued"],
                })
                if rescue_mode:
                    summary["explore_try_times"] = metrics["explore_try_times"]
                else:
                    summary["n_baseline_trajs"] = int(cfg.eval.n_init_states)
            with open(json_path, "w") as f:
                json.dump(summary, f, indent=2)

            if split_mode:
                print(f"\n[run_rollout] DONE. success_rate={metrics['success_rate']:.3f} "
                      f"explore {metrics['explore_solved']}/{metrics['explore_total']} "
                      f"avg_jerk={metrics['avg_jerk']:.4f} "
                      f"collected={metrics['collected_trajs']}")
            else:
                print(f"\n[run_rollout] DONE. success_rate={metrics['success_rate']:.3f} "
                      f"pass@5={metrics['pass_at_5']:.3f} avg_jerk={metrics['avg_jerk']:.4f} "
                      f"collected={metrics['collected_trajs']}")
            print(f"[run_rollout] success -> {success_path} ({len(trajs)} trajs)")
            print(f"[run_rollout] all     -> {all_path} ({len(all_trajs)} trajs)")
            print(f"[run_rollout] json    -> {json_path}")
    finally:
        if wandb_run is not None:
            wandb_run.finish()


if __name__ == "__main__":
    main()
