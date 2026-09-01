"""SOE rollout pipeline: eval (step 2) -> explore-failed-only (step 3).

A single orchestrator implementing the SOE self-improvement rollout flow for
SCOUT (scout_design.md / the SOE `run.py` round structure, steps 2-4):

  step 2  :func:`evaluate_baseline_vec` (``n_tries=1``) over N init states
           -> ``success_rate`` (first-try) + the FAILED init states.
  step 3  :func:`evaluate_exploration_vec` (``only_failed_of=first_results``,
           ``try_times=5``) on the FAILED inits only:
             guided=True  -> VIB-guided denoising (``predict_action_dyn_guided``)
             guided=False -> plain base-DP retry (``predict_action``; baseline)
           -> successful trajs (with obs, for hdf5 write-back) + pass@5 +
           avg_jerk over EVERY exploration trajectory (success + failure, SOE
           3rd-difference norm).
  step 4  done by the CLI (:func:`hdf5_writer.write_rollouts_to_hdf5`); this
          class returns the metrics dict + BOTH trajectory groups:
            * ``trajs``     -- successful EXPLORATION trajs only (baseline
              first-try successes NOT included) -> the DP-retrain "success"
              hdf5 (core + successes);
            * ``all_trajs`` -- EVERY trajectory of the round: all N baseline
              (step-2) rollouts + all ``try_times`` exploration trajectories
              per failed init (success + failure) -> the dyn/VIB-retrain "all"
              hdf5 (diversified (s,a,s') transitions against z-exploration
              drift).

The wandb running view (``eval/success_rate``, ``rollout/pass@5``,
``rollout/avg_jerk`` vs completed-init-count) is driven by the ``on_progress``
callback the CLI supplies; the engine's internal wandb logging is disabled
(``wandb_run=None``) so the custom x-axis (``wandb.define_metric`` step_metric)
is the sole source.
"""

from __future__ import annotations

import json
import os
from typing import Callable, List, Optional

import numpy as np
import torch

from scout.eval.rollout_vec import (
    evaluate_baseline_vec,
    evaluate_exploration_vec,
)
from scout.eval.rollout import (
    collect_initial_states,
    make_action_bridge,
    make_obs_adapter,
)
from scout.eval.metrics import (
    success_rate_per_round,
    pass_at_k,
    jerk,
    jerk_of_results,
)
from scout.eval.factories import DPFactory, VIBFactory, EnvFactory


class RolloutPipeline:
    """SOE step-2 -> step-3 rollout orchestrator (vec engine).

    Args:
        cfg        : loaded eval config (configs/eval_<task>.yaml). Reads
                     ``cfg.eval.{horizon, try_times, n_init_states, n_envs,
                     log_every, view_names, proprio_keys}`` and
                     ``cfg.exploration.{guidance_scale,
                     guidance_start_timestep}``.
        dp_factory / scout_vib_factory / env_factory : from
                     :mod:`scout.eval.factories`.
        device     : torch device.
        guided     : True = VIB-guided exploration (mode=dyn); False = plain
                     base-DP retry on failed inits (mode=off / baseline).
        log_every  : tick period for the engine progress callback.
    """

    def __init__(self, cfg, dp_factory: DPFactory,
                 scout_vib_factory: Optional[VIBFactory],
                 env_factory: EnvFactory,
                 device: Optional[torch.device] = None,
                 guided: bool = False, guide_mode: str = "dyn",
                 bank_hdf5: Optional[str] = None,
                 log_every: Optional[int] = None,
                 entropy_kwargs: Optional[dict] = None,
                 failed_set_json: Optional[str] = None,
                 save_failed_set: Optional[str] = None):
        self.cfg = cfg
        self.dp_factory = dp_factory
        self.scout_vib_factory = scout_vib_factory
        self.env_factory = env_factory
        self.entropy_kwargs = entropy_kwargs or {}
        # rescue protocol fixed-failure-set support (user 2026-08-30): the
        # baseline eval runs ONCE, its failed inits are saved to a json, and
        # every subsequent experiment EXPLORE-ONLY re-runs on exactly that
        # scene set (pass@10 measured on a frozen failure set).
        self.failed_set_json = failed_set_json
        self.save_failed_set = save_failed_set
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.guided = bool(guided)
        # "dyn" = SCOUT exploration guidance (z ~ prior, per trajectory);
        # "expert" = expert z-bank guidance (z* = nearest bank entry, per
        # action chunk; scout/guidance/expert_bank.py). Only read when guided.
        self.guide_mode = str(guide_mode)
        # expert-mode bank source; None -> cfg.dataset.path (the core hdf5).
        self.bank_hdf5 = bank_hdf5
        self.n_envs = int(getattr(cfg.eval, "n_envs", 1))
        self.log_every = int(log_every if log_every is not None
                             else getattr(cfg.eval, "log_every", 10))
        # guidance / obs-adapter config (only used when guided)
        self.guidance_scale = float(getattr(cfg, "exploration", {})
                                    .get("guidance_scale", 5.0))
        self.guidance_start_timestep = int(
            getattr(cfg, "exploration", {}).get("guidance_start_timestep", 50))
        self.view_names = list(getattr(cfg.eval, "view_names", []))
        self.proprio_keys = list(getattr(cfg.eval, "proprio_keys", []))
        # base seed for reproducible init scenes (42 -> init i uses seed 42+i,
        # i.e. 42..141) + torch/np seeded for step2 reproducibility. None ->
        # unseeded (legacy). Same seed across runs -> same 100 scenes -> a
        # controlled DP-vs-SCOUT comparison (success_rate / failed set match).
        self.base_seed = getattr(cfg.eval, "seed", None)

    # ------------------------------------------------------------------ #
    def _attach_planner(self, dp, scout_vib):
        """Attach the SCOUT planner to ``dp`` (seam ① + ②). Idempotent; skipped
        for mocks without ``initialize_scout_planner``.

        Per design §4: planner carries the frozen ScoutVIB (``E_s`` +
        ``vib_enc``) + the unnormalize-only action bridge (from this DP's
        normalizer) + the obs-adapter (LPB keyed obs -> E_s format). z is
        sampled fresh per rollout inside the vec runner -- EXCEPT in expert
        mode (``guide_mode="expert"``): the expert z-bank planner selects z*
        per action chunk inside the denoise loop (user 2026-08-21).
        """
        from scout.guidance.planner import ScoutPlanner
        bridge = make_action_bridge(dp)                                   # seam ②
        obs_adapter = make_obs_adapter(self.view_names, self.proprio_keys)  # seam ①
        if self.guide_mode == "expert":
            from scout.guidance.expert_bank import (
                ScoutExpertPlanner,
                build_expert_z_bank,
            )
            bank_src = self.bank_hdf5 or self.cfg.dataset.path
            bank = build_expert_z_bank(
                bank_src, scout_vib,
                view_names=self.view_names, proprio_keys=self.proprio_keys,
                device=self.device,
                stride=int(getattr(getattr(self.cfg, "exploration", {}),
                                   "bank_stride", 1)),
            )
            planner = ScoutExpertPlanner(scout_vib, z_bank=bank,
                                         bridge=bridge, obs_adapter=obs_adapter)
            print(f"[rollout] expert z-bank guidance: {bank.shape[0]} "
                  f"entries from {bank_src}")
        elif self.guide_mode == "exploit":
            # LPB-parity exploit guidance (user 2026-08-29): NN distance from
            # the D_s-predicted next state latent to an expert STATE-latent
            # bank (exploit_costs.py) -- attract to the expert manifold,
            # opposite direction of the entropy-cost family.
            from scout.guidance.exploit_costs import (
                ExploitCostPlanner,
                build_expert_state_bank,
            )
            bank_src = self.bank_hdf5 or self.cfg.dataset.path
            bank = build_expert_state_bank(
                bank_src, scout_vib,
                view_names=self.view_names, proprio_keys=self.proprio_keys,
                device=self.device,
                stride=int(getattr(getattr(self.cfg, "exploration", {}),
                                   "bank_stride", 1)),
            )
            latent = str(self.entropy_kwargs.get("exploit_latent", "eye"))
            ood = self.entropy_kwargs.get("exploit_ood_threshold", None)
            knn = int(self.entropy_kwargs.get("exploit_knn", 1) or 1)
            slope = self.entropy_kwargs.get("exploit_gate_slope", None)
            cap = float(self.entropy_kwargs.get("exploit_gate_cap", 2.0) or 2.0)
            planner = ExploitCostPlanner(
                scout_vib, state_bank=bank, bridge=bridge,
                obs_adapter=obs_adapter, latent=latent,
                ood_threshold=(None if ood is None else float(ood)),
                knn=knn,
                gate_slope=(None if slope is None else float(slope)),
                gate_cap=cap)
            print(f"[rollout] exploit state-bank guidance: bank={bank.shape} "
                  f"latent={latent} ood_threshold={ood} knn={knn} "
                  f"gate_slope={slope} gate_cap={cap} src={bank_src}")
        elif self.guide_mode in ("novelty", "atypical", "combo", "shell"):
            # entropy-dev (user 2026-08-24 方案二/三; 2026-08-27 方案A shell):
            # only the cost changes; same injection path, same frozen dyn/VIB
            # encoder.
            from scout.guidance.entropy_costs import (
                AtypicalCostPlanner,
                ComboCostPlanner,
                NoveltyCostPlanner,
                ShellTargetCostPlanner,
            )
            ek = dict(self.entropy_kwargs or {})
            if self.guide_mode == "novelty":
                planner = NoveltyCostPlanner(
                    scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                    h_scale=float(ek.get("novelty_h", 5.0)),
                    sample_z=bool(ek.get("novelty_sample_z", False)),
                )
            elif self.guide_mode == "atypical":
                planner = AtypicalCostPlanner(
                    scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                    cap=float(ek.get("atypical_cap", 10.0)),
                )
            elif self.guide_mode == "shell":
                planner = ShellTargetCostPlanner(
                    scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                    shell_kappa=float(ek.get("shell_kappa", 2.5)),
                    shell_seed=int(ek.get("shell_seed", 42)),
                )
            else:
                planner = ComboCostPlanner(
                    scout_vib, bridge=bridge, obs_adapter=obs_adapter,
                    h_scale=float(ek.get("novelty_h", 5.0)),
                    sample_z=bool(ek.get("novelty_sample_z", False)),
                    cap=float(ek.get("atypical_cap", 10.0)),
                    nov_weight=float(ek.get("combo_nov_weight", 1.0)),
                    att_weight=float(ek.get("combo_att_weight", 1.0)),
                )
            print(f"[rollout] entropy cost guidance: mode={self.guide_mode} {ek}")
        else:
            planner = ScoutPlanner(scout_vib, bridge=bridge,
                                   obs_adapter=obs_adapter, z=None)
        init = getattr(dp, "initialize_scout_planner", None)
        if callable(init):
            init(planner, self.guidance_start_timestep, self.guidance_scale)

    # ------------------------------------------------------------------ #
    def run(self, dp_ckpt: str, vib_ckpt: Optional[str] = None,
            on_progress: Optional[Callable[..., None]] = None,
            success_only: bool = False,
            explore_seed: Optional[int] = None,
            n_explore: Optional[int] = None,
            explore_try_times: int = 1,
            eval_only: bool = False,
            explore_mode: str = "fresh",
            skip_eval: bool = False,
            scene_slice: Optional[tuple] = None) -> dict:
        """Run SOE step 2 (eval) + step 3 (explore failed only).

        ``on_progress`` is invoked as
            ``on_progress("eval",    payload)``                      (step 2)
            ``on_progress("explore", payload, baseline_solved, N)``  (step 3)
        where ``payload`` carries the engine's running counters (see
        :mod:`scout.eval.rollout_vec`). The CLI uses these to plot the three
        wandb metrics vs completed-init-count.

        Split mode (experiment2, user 2026-08-17): pass ``explore_seed`` (and
        optionally ``n_explore`` / ``explore_try_times``) to DECOUPLE the two
        phases: eval measures the seed-fixed scene set (same every round),
        explore rolls FRESH scenes from ``explore_seed`` (driver passes
        ``round*1000+42``; scenes differ every round). Explore is no longer
        gated on eval failures -- every explore scene gets ``try_times``
        rollouts; successes -> DP retrain data, ALL trajs -> dyn retrain data
        (eval-phase trajs are measurement only and are NOT collected).

        Rescue mode (user 2026-08-23, ``explore_mode="rescue"``): SOE protocol
        -- explore retries ONLY the failed eval inits (same initial states)
        ``explore_try_times`` each; DP data = successful retries, dyn data =
        per failed init {successful retries if any, else FIRST retry}. See
        :meth:`_run_rescue`.

        Returns ``{"metrics": {...},
                   "trajs":     [successful EXPLORATION trajs, with obs],  # DP
                   "all_trajs": [every traj of the round, with obs]}``.     # dyn
        """
        if explore_mode == "rescue":
            return self._run_rescue(
                dp_ckpt, vib_ckpt=vib_ckpt, on_progress=on_progress,
                try_times=int(explore_try_times), eval_only=eval_only,
                scene_slice=scene_slice)
        if scene_slice is not None:
            raise ValueError(
                "scene_slice is only implemented for explore_mode='rescue' "
                f"(got {explore_mode!r})")
        if explore_seed is not None or eval_only or skip_eval:
            if skip_eval and eval_only:
                raise ValueError("--skip-eval and --eval-only are mutually "
                                 "exclusive (skip-eval drops the eval phase, "
                                 "eval-only drops the explore phase)")
            return self._run_split(
                dp_ckpt, vib_ckpt=vib_ckpt, on_progress=on_progress,
                explore_seed=int(explore_seed) if explore_seed is not None else 0,
                n_explore=int(n_explore) if n_explore is not None else 500,
                explore_try_times=int(explore_try_times),
                eval_only=eval_only, skip_eval=skip_eval)
        horizon = int(self.cfg.eval.horizon)
        try_times = int(getattr(self.cfg.eval, "try_times", 5))
        n_init = int(getattr(self.cfg.eval, "n_init_states", 100))
        print(f"[rollout] dp_ckpt={dp_ckpt} guided={self.guided} "
              f"n_init={n_init} try_times={try_times} n_envs={self.n_envs}"
              + (f" vib_ckpt={vib_ckpt}" if self.guided else ""))
        if self.base_seed is not None:
            torch.manual_seed(int(self.base_seed))
            np.random.seed(int(self.base_seed))
            print(f"[rollout] base_seed={self.base_seed}: init i seeded "
                  f"{self.base_seed}+i ({self.base_seed}..{self.base_seed + n_init - 1}); "
                  "torch/np seeded for step2 reproducibility")

        dp = self.dp_factory(dp_ckpt)
        n_action_steps = int(getattr(dp, "n_action_steps", 1))
        init_states = collect_initial_states(self.env_factory, n_init_states=n_init,
                                             base_seed=self.base_seed)
        N = len(init_states)
        print(f"[rollout] collected {N} init states")

        # ---- step 2: baseline eval, 1 try each (pure base path) ----
        eval_cb = (lambda p: on_progress("eval", p)) if on_progress else None
        first_results, _, _ = evaluate_baseline_vec(
            dp, self.env_factory, init_states, horizon=horizon,
            n_envs=self.n_envs, n_action_steps=n_action_steps, device=self.device,
            n_tries=1, metric_prefix="eval",
            # record frames unless --success-only (the `all` hdf5 needs every
            # baseline traj with obs; pure-eval mode skips the memory cost)
            record_obs=not success_only,
            on_progress=eval_cb, wandb_run=None, log_every=self.log_every,
        )
        baseline_solved = int(sum(1 for s, _ in first_results if s))
        print(f"[rollout] step2 eval: success_rate={baseline_solved}/{N} "
              f"({baseline_solved / max(N, 1):.3f}); "
              f"{N - baseline_solved} failed inits")
        if success_only:
            # --success-only: skip step3 (explore) + step4 (merge); return step2
            # metrics only (no VIB / no hdf5 needed). Used to eval an arbitrary
            # DP ckpt's pure success_rate on the seed-fixed 100-init scene set.
            print("[rollout] --success-only: skipping explore + merge")
            return {"metrics": {
                "success_rate": success_rate_per_round(first_results),
                "jerk_baseline": jerk_of_results(first_results, only_successful=True),
                "baseline_solved": baseline_solved,
                "n_failed": N - baseline_solved,
                "failed_init_indices": [i for i, (s, _) in enumerate(first_results)
                                        if not s],
            }, "trajs": [], "all_trajs": []}

        # ---- step 3: explore FAILED inits only (guided or plain DP retry) ----
        if self.guided:
            if self.scout_vib_factory is None:
                raise ValueError("guided=True requires a scout_vib_factory")
            scout_vib = (self.scout_vib_factory(vib_ckpt)
                         if vib_ckpt is not None else self.scout_vib_factory())
            self._attach_planner(dp, scout_vib)
        expl_cb = (lambda p: on_progress("explore", p, baseline_solved, N)
                   ) if on_progress else None
        expl = evaluate_exploration_vec(
            dp, self.env_factory, init_states, horizon=horizon,
            try_times=try_times, n_envs=self.n_envs, n_action_steps=n_action_steps,
            device=self.device, only_failed_of=first_results, guided=self.guided,
            on_progress=expl_cb, wandb_run=None, log_every=self.log_every,
        )
        trajs: List[dict] = [t for e in expl for t in e["successful_trajs"]]
        n_failed = N - baseline_solved
        exploration_rescued = int(sum(1 for e in expl
                                      if e["solved"] and not e["baseline_solved"]))
        print(f"[rollout] step3 explore: rescued {exploration_rescued}/{n_failed} "
              f"failed inits; collected {len(trajs)} successful trajs")

        all_trajs = _assemble_all_trajs(first_results, expl)
        print(f"[rollout] all-traj group: {len(all_trajs)} trajs "
              f"(baseline {N} + explore "
              f"{len(all_trajs) - N}) -> dyn-retrain hdf5")
        metrics = {
            "success_rate": success_rate_per_round(first_results),
            "pass_at_5": pass_at_k(expl, first_results, k=try_times),
            "avg_jerk": _jerk_all_explore_trajs(expl),
            "jerk_baseline": jerk_of_results(first_results, only_successful=True),
            "baseline_solved": baseline_solved,
            "n_failed": n_failed,
            "exploration_rescued": exploration_rescued,
            "collected_trajs": len(trajs),
            "n_all_trajs": len(all_trajs),
            "n_baseline_trajs": N,
            "failed_init_indices": [i for i, (s, _) in enumerate(first_results)
                                    if not s],
        }
        return {"metrics": metrics, "trajs": trajs, "all_trajs": all_trajs}

    # ------------------------------------------------------------------ #
    def _run_split(self, dp_ckpt: str, vib_ckpt: Optional[str] = None,
                   on_progress: Optional[Callable[..., None]] = None,
                   explore_seed: int = 1042, n_explore: int = 500,
                   explore_try_times: int = 1,
                   eval_only: bool = False,
                   skip_eval: bool = False) -> dict:
        """experiment2 split protocol (user 2026-08-17).

        eval   : the seed-fixed measurement set (``cfg.eval.seed`` ->
                 seeds seed..seed+n_eval-1, default 42..141, 100 scenes),
                 1 try each, NO data collection (record_obs=False).
        explore: FRESH scenes every round -- ``explore_seed`` ->
                 seeds explore_seed..explore_seed+n_explore-1 (driver passes
                 round*1000+42; default here round1 = 1042..1541, 500 scenes),
                 ``try_times`` rollouts per scene (default 1). Guided or plain
                 per ``self.guided``.
                 successes -> ``trajs`` (DP retrain data);
                 ALL explore trajs -> ``all_trajs`` (dyn retrain data).

        ``skip_eval`` (user 2026-08-29): drop the pure-DP eval phase and run
        ONLY the guided explore phase -- for one-try guidance tests where the
        baseline number comes from a separate ``--eval-only`` run (same ckpt
        + same seed reproduces the eval segment bit-for-bit, so the repeated
        segment is pure waste).
        """
        horizon = int(self.cfg.eval.horizon)
        n_eval = int(self.cfg.eval.n_init_states)
        print(f"[rollout:split] dp_ckpt={dp_ckpt} guided={self.guided} "
              f"eval: n={n_eval} seed={self.base_seed} | "
              f"explore: n={n_explore} seed={explore_seed} "
              f"try_times={explore_try_times} n_envs={self.n_envs}"
              + (f" vib_ckpt={vib_ckpt}" if self.guided else ""))
        if self.base_seed is not None:
            torch.manual_seed(int(self.base_seed))
            np.random.seed(int(self.base_seed))

        dp = self.dp_factory(dp_ckpt)
        n_action_steps = int(getattr(dp, "n_action_steps", 1))

        # ---- phase 1: eval on the seed-fixed scene set (measurement only) -- #
        first_results = None
        baseline_solved = 0
        if not skip_eval:
            eval_states = collect_initial_states(
                self.env_factory, n_init_states=n_eval, base_seed=self.base_seed)
            eval_cb = (lambda p: on_progress("eval", p)) if on_progress else None
            first_results, _, _ = evaluate_baseline_vec(
                dp, self.env_factory, eval_states, horizon=horizon,
                n_envs=self.n_envs, n_action_steps=n_action_steps, device=self.device,
                n_tries=1, metric_prefix="eval",
                record_obs=False,                      # eval trajs are NOT data
                on_progress=eval_cb, wandb_run=None, log_every=self.log_every,
            )
            baseline_solved = int(sum(1 for s, _ in first_results if s))
            print(f"[rollout:split] eval: success_rate={baseline_solved}/{n_eval} "
                  f"({baseline_solved / max(n_eval, 1):.3f})")
        else:
            print("[rollout:split] skip-eval: baseline comes from a separate "
                  "--eval-only run; ONLY the guided explore phase runs here")

        # ---- phase 2: explore on FRESH scenes (data collection) ------------ #
        if eval_only:
            # user 2026-08-18: eval-only round (final round of a chain) --
            # measurement only; no explore, no hdf5, no retrains downstream.
            metrics = {
                "success_rate": success_rate_per_round(first_results),
                "jerk_baseline": jerk_of_results(first_results, only_successful=True),
                "baseline_solved": baseline_solved,
                "n_failed": n_eval - baseline_solved,
                "failed_init_indices": [i for i, (s, _) in
                                        enumerate(first_results) if not s],
                "eval_only": True,
            }
            print("[rollout:split] eval-only round -- skipping explore phase")
            return {"metrics": metrics, "trajs": [], "all_trajs": []}
        if self.guided:
            if self.scout_vib_factory is None:
                raise ValueError("guided=True requires a scout_vib_factory")
            scout_vib = (self.scout_vib_factory(vib_ckpt)
                         if vib_ckpt is not None else self.scout_vib_factory())
            self._attach_planner(dp, scout_vib)
        explore_states = collect_initial_states(
            self.env_factory, n_init_states=n_explore, base_seed=explore_seed)
        expl_cb = (lambda p: on_progress("explore", p, 0, len(explore_states))
                   ) if on_progress else None
        expl = evaluate_exploration_vec(
            dp, self.env_factory, explore_states, horizon=horizon,
            try_times=explore_try_times, n_envs=self.n_envs,
            n_action_steps=n_action_steps, device=self.device,
            only_failed_of=None,                    # EVERY scene explored
            guided=self.guided,
            on_progress=expl_cb, wandb_run=None, log_every=self.log_every,
        )
        trajs: List[dict] = [t for e in expl for t in e["successful_trajs"]]
        all_trajs: List[dict] = [t for e in expl for t in e.get("all_trajs", [])]
        explore_solved = int(sum(1 for e in expl if e["solved"]))
        print(f"[rollout:split] explore: solved {explore_solved}/{n_explore} "
              f"scenes; collected {len(trajs)} successful trajs "
              f"({len(all_trajs)} total explore trajs -> dyn)")
        metrics = {
            "explore_seed": explore_seed,
            "n_explore": n_explore,
            "explore_try_times": explore_try_times,
            "explore_solved": explore_solved,
            "explore_total": n_explore,
            "avg_jerk": _jerk_all_explore_trajs(expl),
            "collected_trajs": len(trajs),
            "n_all_trajs": len(all_trajs),
        }
        if skip_eval:
            metrics["skip_eval"] = True   # baseline fields live in the eval-only run
        else:
            metrics.update({
                "success_rate": success_rate_per_round(first_results),
                "jerk_baseline": jerk_of_results(first_results, only_successful=True),
                "baseline_solved": baseline_solved,
                "n_failed": n_eval - baseline_solved,
            })
        return {"metrics": metrics, "trajs": trajs, "all_trajs": all_trajs}

    # ------------------------------------------------------------------ #
    def _run_rescue(self, dp_ckpt: str, vib_ckpt: Optional[str] = None,
                    on_progress: Optional[Callable[..., None]] = None,
                    try_times: int = 5, eval_only: bool = False,
                    scene_slice: Optional[tuple] = None) -> dict:
        """SOE rescue protocol (user 2026-08-23) -- explore == eval scenes.

        eval   : the seed-fixed measurement set (``cfg.eval.seed`` -> seeds
                 seed..seed+n_eval-1, default 42..141, 100 scenes), 1 try
                 each, NO data collection (record_obs=False).
        explore: retry ONLY the failed eval inits, ``try_times`` each
                 (default 5), from the SAME initial states, guided or plain
                 per ``self.guided`` (SOE run.py:121-149 semantics).
                 DP data  = every SUCCESSFUL retry (``trajs``).
                 dyn data = per failed init: its successful retries if any,
                 else its FIRST retry (``all_trajs``) -- user rule: scenes
                 solved by exploration contribute their successes, all-failed
                 scenes contribute one representative (first) trajectory.
        """
        horizon = int(self.cfg.eval.horizon)
        n_eval = int(self.cfg.eval.n_init_states)
        print(f"[rollout:rescue] dp_ckpt={dp_ckpt} guided={self.guided} "
              f"eval: n={n_eval} seed={self.base_seed} | "
              f"explore: failed-of-eval x{try_times} n_envs={self.n_envs}"
              + (f" vib_ckpt={vib_ckpt}" if self.guided else ""))
        if self.base_seed is not None:
            torch.manual_seed(int(self.base_seed))
            np.random.seed(int(self.base_seed))

        dp = self.dp_factory(dp_ckpt)
        n_action_steps = int(getattr(dp, "n_action_steps", 1))

        # ---- phase 1: eval on the seed-fixed scene set (measurement only) -- #
        eval_states = collect_initial_states(
            self.env_factory, n_init_states=n_eval, base_seed=self.base_seed)
        _loaded_failed_set = False
        if (self.failed_set_json is not None
                and os.path.exists(self.failed_set_json)):
            # explore-only mode (user 2026-08-30): reuse the FROZEN failure
            # set of the baseline run instead of re-rolling the eval phase.
            # Init states are still collected (same seed -> bit-identical
            # scenes); only the baseline rollout is skipped.
            with open(self.failed_set_json) as f:
                spec = json.load(f)
            if int(spec.get("n_eval", -1)) != n_eval:
                raise RuntimeError(
                    f"failed-set json {self.failed_set_json} was built with "
                    f"n_eval={spec.get('n_eval')} != {n_eval} -- refusing")
            _bs = spec.get("base_seed")
            if (_bs is not None and self.base_seed is not None
                    and int(_bs) != int(self.base_seed)):
                raise RuntimeError(
                    f"failed-set json {self.failed_set_json} was built with "
                    f"base_seed={_bs} != {self.base_seed} -- different scene "
                    f"set; refusing")
            if spec.get("dp_ckpt") != dp_ckpt:
                print(f"[rollout:rescue] WARNING: failed set was recorded for "
                      f"dp_ckpt={spec.get('dp_ckpt')}, this run uses "
                      f"{dp_ckpt} (proceeding on the frozen scene set)")
            _failed = set(int(i) for i in spec["failed_init_indices"])
            first_results = [(i not in _failed, eval_states[i])
                             for i in range(n_eval)]
            baseline_solved = n_eval - len(_failed)
            _loaded_failed_set = True
            print(f"[rollout:rescue] explore-only: failed set loaded from "
                  f"{self.failed_set_json} ({len(_failed)} failed inits, "
                  f"recorded SR {baseline_solved}/{n_eval})")
        else:
            eval_cb = (lambda p: on_progress("eval", p)) if on_progress else None
            first_results, _, _ = evaluate_baseline_vec(
                dp, self.env_factory, eval_states, horizon=horizon,
                n_envs=self.n_envs, n_action_steps=n_action_steps, device=self.device,
                n_tries=1, metric_prefix="eval",
                record_obs=False,                      # eval trajs are NOT data
                on_progress=eval_cb, wandb_run=None, log_every=self.log_every,
            )
            baseline_solved = int(sum(1 for s, _ in first_results if s))
            if self.save_failed_set:
                os.makedirs(os.path.dirname(self.save_failed_set) or ".",
                            exist_ok=True)
                _spec = {
                    "failed_init_indices": [i for i, (s, _) in
                                            enumerate(first_results) if not s],
                    "n_eval": n_eval,
                    "base_seed": self.base_seed,
                    "dp_ckpt": dp_ckpt,
                    "baseline_solved": baseline_solved,
                }
                with open(self.save_failed_set, "w") as f:
                    json.dump(_spec, f, indent=1)
                print(f"[rollout:rescue] failed set saved to "
                      f"{self.save_failed_set} "
                      f"({n_eval - baseline_solved} failed inits)")
        n_failed = n_eval - baseline_solved
        # ---- scene slicing (multicore sharding, 2026-09-01) ---------------- #
        # Worker SLOT of SHARDS keeps scenes with original index %% SHARDS ==
        # SLOT (the frozen seed-fixed set is cut AFTER it is fully determined,
        # so every worker sees bit-identical scenes; original indices are
        # preserved through sel for failed_init_indices / explore_detail).
        sel = list(range(n_eval))
        if scene_slice is not None:
            slot, shards = scene_slice
            sel = list(range(slot, n_eval, shards))
            eval_states = [eval_states[i] for i in sel]
            first_results = [first_results[i] for i in sel]
            baseline_solved = int(sum(1 for s, _ in first_results))
            n_failed = len(first_results) - baseline_solved
            print(f"[rollout:rescue] scene slice {slot}/{shards}: "
                  f"{len(sel)} scenes (original indices {sel[0]}..{sel[-1]}); "
                  f"slice SR {baseline_solved}/{len(sel)}")
        print(f"[rollout:rescue] eval: success_rate={baseline_solved}/"
              f"{len(first_results)} "
              f"({baseline_solved / max(len(first_results), 1):.3f}); "
              f"{n_failed} failed inits -> explore")

        if eval_only:
            metrics = {
                "success_rate": success_rate_per_round(first_results),
                "jerk_baseline": jerk_of_results(first_results, only_successful=True),
                "baseline_solved": baseline_solved,
                "n_failed": n_failed,
                "failed_init_indices": [sel[i] for i, (s, _) in
                                        enumerate(first_results) if not s],
                "eval_only": True,
                "scene_slice": (list(scene_slice)
                                if scene_slice is not None else None),
                "n_eval_global": n_eval,
                "n_slice": len(sel),
            }
            print("[rollout:rescue] eval-only round -- skipping explore phase")
            return {"metrics": metrics, "trajs": [], "all_trajs": []}

        # ---- phase 2: explore = retry FAILED eval inits -------------------- #
        if self.guided:
            if self.scout_vib_factory is None:
                raise ValueError("guided=True requires a scout_vib_factory")
            scout_vib = (self.scout_vib_factory(vib_ckpt)
                         if vib_ckpt is not None else self.scout_vib_factory())
            self._attach_planner(dp, scout_vib)
        expl_cb = (lambda p: on_progress(
            "explore", p, baseline_solved, len(first_results))
                   ) if on_progress else None
        expl = evaluate_exploration_vec(
            dp, self.env_factory, eval_states, horizon=horizon,
            try_times=try_times, n_envs=self.n_envs,
            n_action_steps=n_action_steps, device=self.device,
            only_failed_of=first_results, guided=self.guided,
            on_progress=expl_cb, wandb_run=None, log_every=self.log_every,
        )
        trajs: List[dict] = [t for e in expl for t in e["successful_trajs"]]
        all_trajs: List[dict] = []
        for e in expl:
            if e["baseline_solved"]:
                continue                       # eval-solved scenes add no data
            if e["solved"]:
                all_trajs.extend(e["successful_trajs"])
            else:
                first = e.get("first_traj")
                if first is None:
                    raise RuntimeError(
                        "rescue: all-failed init has no first_traj "
                        "(engine must tag try_idx==0)")
                all_trajs.append(first)
        rescued = int(sum(1 for e in expl
                          if e["solved"] and not e["baseline_solved"]))
        print(f"[rollout:rescue] explore: rescued {rescued}/{n_failed} "
              f"failed inits; {len(trajs)} successful trajs -> DP retrain; "
              f"{len(all_trajs)} selected trajs -> dyn retrain")
        _avg_jerk, _jerk_n = _jerk_all_explore_stats(expl)
        metrics = {
            "success_rate": success_rate_per_round(first_results),
            "pass_at_5": pass_at_k(expl, first_results, k=try_times),
            "avg_jerk": _avg_jerk,
            "explore_jerk_n": _jerk_n,
            # explore-only: no baseline trajs were rolled -> no baseline jerk
            "jerk_baseline": (None if _loaded_failed_set else
                              jerk_of_results(first_results,
                                              only_successful=True)),
            "baseline_solved": baseline_solved,
            "n_failed": n_failed,
            "exploration_rescued": rescued,
            "explore_solved": rescued,
            "explore_total": n_failed,
            "explore_try_times": try_times,
            "collected_trajs": len(trajs),
            "n_all_trajs": len(all_trajs),
            "failed_init_indices": [sel[i] for i, (s, _) in enumerate(first_results)
                                    if not s],
            # per-failed-init rescue record (2026-08-31 budget-split design):
            # first_success_try is the 1-based try index of the first success
            # (= try_times when the init was never solved) -> pass@k curves and
            # per-scene rescued sets straight from the json, no fingerprinting.
            "explore_detail": [
                {"init": sel[i], "solved": bool(e["solved"]),
                 "first_success_try": int(e["n_tries"])}
                for i, e in enumerate(expl) if not e["baseline_solved"]
            ],
            "scene_slice": (list(scene_slice)
                            if scene_slice is not None else None),
            "n_eval_global": n_eval,
            "n_slice": len(sel),
        }
        if _loaded_failed_set:
            metrics["explore_only"] = True
            metrics["failed_set_source"] = self.failed_set_json
        return {"metrics": metrics, "trajs": trajs, "all_trajs": all_trajs}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _jerk_all_explore_stats(exploration_results):
    """(mean, count) companion of :func:`_jerk_all_explore_trajs` -- the
    count feeds the scene-shard merge's EXACT weighted mean (T<4 trajs are
    skipped from the mean in both; weighting by n_failed*try_times would be
    approximate whenever skips differ across shards)."""
    jerks: List[float] = []
    for e in exploration_results:
        for traj in e.get("all_trajs", []):
            j = jerk(traj["actions"])
            if j > 0.0:
                jerks.append(j)
    return (float(np.mean(jerks)) if jerks else 0.0), len(jerks)


def _jerk_all_explore_trajs(exploration_results) -> float:
    """Mean SOE jerk over EVERY exploration trajectory (``all_trajs``: success
    + failure), T<4 skipped. Same caliber as the engine's per-tick running
    ``avg_jerk`` (:func:`scout.eval.rollout_vec` accumulates jerk over every
    finalized traj). Used for the final JSON value.
    """
    return _jerk_all_explore_stats(exploration_results)[0]


def _assemble_all_trajs(first_results, exploration_results) -> List[dict]:
    """EVERY trajectory of the round, for the dyn-retrain ``all`` hdf5.

    Order: the N baseline (step-2) trajectories first (success AND failure --
    requires the pipeline's ``record_obs=True`` baseline), then every failed
    init's exploration trajectories (all ``try_times`` of them, success +
    failure). Baseline first-try successes are NOT written to the success
    (DP-retrain) file, but they DO appear here: the dyn/VIB retrain wants the
    diversified transition pool, not just the curated successes.
    """
    all_trajs: List[dict] = [traj for _, traj in first_results]
    for e in exploration_results:
        all_trajs.extend(e.get("all_trajs", []))
    return all_trajs


# --------------------------------------------------------------------------- #
# synthetic wiring check (metrics assembly on dummy first_results + expl)
# --------------------------------------------------------------------------- #
def _dry_run():
    """Verify the metrics assembly + helper wiring on synthetic results (no
    engine / no env). Run via ``python -m scout.eval.rollout_pipeline``.
    """
    rng = np.random.default_rng(0)

    def mk(T=15, A=10, succ=True):
        return {"actions": rng.standard_normal((T, A)).astype(np.float32),
                "rewards": np.zeros(T, dtype=np.float32),
                "dones": np.zeros(T, dtype=bool), "states": [{}] * T,
                "obs": [], "next_obs": [], "horizon": T, "success": succ,
                "initial_state_dict": None}

    N = 6
    # 3 baseline-successful, 3 failed; of the failed, 2 rescued by exploration.
    first_results = [(True, mk())] * 3 + [(False, mk())] * 3
    expl = [
        {"solved": True, "n_tries": 0, "successful_trajs": [], "all_trajs": [],
         "baseline_solved": True}] * 3 + [
        {"solved": True, "n_tries": 2,
         "successful_trajs": [mk()], "all_trajs": [mk(succ=False), mk()],
         "baseline_solved": False},
        {"solved": True, "n_tries": 4,
         "successful_trajs": [mk()], "all_trajs": [mk(succ=False)] * 4 + [mk()],
         "baseline_solved": False},
        {"solved": False, "n_tries": 5, "successful_trajs": [],
         "all_trajs": [mk(succ=False)] * 5, "baseline_solved": False},
    ]
    sr = success_rate_per_round(first_results)
    p5 = pass_at_k(expl, first_results, k=5)
    aj = _jerk_all_explore_trajs(expl)
    jb = jerk_of_results(first_results, only_successful=True)
    print(f"success_rate = {sr:.4f}  (expect 0.5000)")
    print(f"pass_at_5    = {p5:.4f}  (expect 0.8333 = 5/6)")
    print(f"avg_jerk     = {aj:.4f}  (expect > 0, over all all_trajs incl fails)")
    print(f"jerk_baseline= {jb:.4f}  (expect > 0)")
    assert abs(sr - 0.5) < 1e-6
    assert abs(p5 - 5.0 / 6.0) < 1e-6
    assert aj > 0.0
    # _jerk_all_explore_trajs covers all all_trajs (success + fail) -> nonzero.
    assert _jerk_all_explore_trajs(expl) > 0.0
    # all-traj assembly: 6 baseline + (0+0+0 baseline-solved) + (2+5+5 explore).
    all_t = _assemble_all_trajs(first_results, expl)
    assert len(all_t) == 18, f"all = 6 baseline + 12 explore; got {len(all_t)}"
    print(f"all_trajs     = {len(all_t)}  (expect 18 = 6 baseline + 12 explore)")
    print("[dry-run] rollout_pipeline metrics assembly OK")


if __name__ == "__main__":
    _dry_run()
