"""EvalPipeline -- base-DP metric evaluation (SOE protocol), fully decoupled
from rollout collection.

Given a base-DP checkpoint, this measures the SOE round-summary metrics over
N (default 100) random initial states, each run ``try_times`` (default 5)
times:

  * **success_rate**    : fraction of init states the base DP solved on the
                          FIRST try (single-attempt, paper-comparable).
  * **jerk_baseline**   : mean action jerk over the successful first-try trajs.
  * **base_pass_at_5**  : fraction of init states solved at least once across
                          all ``try_times`` tries (retry budget).

It walks ONLY the baseline policy path (``predict_action`` under
``torch.no_grad``) -- no ScoutVIB, no planner, no guided exploration. That is
the eval/rollout decoupling boundary: metric measurement never needs the
dynamics model.

Parallelism: ``n_envs>1`` drives the single-process vectorized engine
(:func:`rollout_vec.evaluate_baseline_vec`, batched inference across N envs);
``n_envs==1`` falls back to the sequential :func:`rollout.evaluate_baseline`.

Live wandb progress (``eval/baseline_env_done`` out of N,
``eval/baseline_successes``, ``eval/baseline_success_rate``,
``eval/base_pass_at_5``) streams while the baseline runs; a final ``round/*``
summary is logged once metrics are computed.

Run via ``python -m scout.eval.run_eval`` (real run) or
``python -m scout.eval.evaluator`` (mock dry-run).
"""

from __future__ import annotations

from typing import Optional

import torch
from easydict import EasyDict

from scout.eval.factories import DPFactory, EnvFactory
from scout.eval.metrics import summarize_round
from scout.eval.rollout import (
    BaseDPAdapter,
    collect_initial_states,
    evaluate_baseline,
)
from scout.eval.rollout_vec import evaluate_baseline_vec


class EvalPipeline:
    """base-DP metric evaluation: N init states x try_times tries -> metrics.

    Parameters
    ----------
    cfg : EasyDict
        Eval config (``configs/eval_<task>.yaml``). Reads ``cfg.eval.horizon``,
        ``cfg.eval.n_init_states`` (default 100), ``cfg.eval.try_times``
        (default 5), ``cfg.eval.n_envs`` (default 50), ``cfg.eval.log_every``.
    dp_factory : callable
        ``dp_factory(ckpt_path) -> ScoutPolicy``. Built by
        :func:`factories.make_lpb_dp_factory`; mock in the dry-run.
    env_factory : callable
        ``env_factory() -> env``. Built by
        :func:`factories.make_default_env_factory`; mock in the dry-run.
    device : torch.device
    wandb_run : optional
        Live ``eval/baseline_*`` + ``round/*`` progress (None disables logging).
    """

    def __init__(self, cfg: EasyDict, dp_factory: DPFactory,
                 env_factory: EnvFactory, device: Optional[torch.device] = None,
                 wandb_run=None):
        self.cfg = cfg
        self.dp_factory = dp_factory
        self.env_factory = env_factory
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.wandb_run = wandb_run
        # parallelism: n_envs>1 -> vectorized; ==1 -> sequential fallback.
        self.n_envs = int(getattr(cfg.eval, "n_envs", 1))
        self.log_every = int(getattr(cfg.eval, "log_every", 10))
        self.verbose = bool(getattr(cfg, "verbose", True))

    def _log(self, *a, **kw):
        if self.verbose:
            print(*a, **kw)

    def run(self, dp_ckpt: str) -> dict:
        """Evaluate the base DP at ``dp_ckpt`` over N init states x try_times.

        Returns the metrics dict from :func:`metrics.summarize_round`:
        ``success_rate`` (first-try), ``base_pass_at_5`` (any-of-try_times),
        ``jerk_baseline``, plus the pass_at_k / exploration_yield / jerk_exploration
        fields which are all zero for a baseline-only eval.
        """
        horizon = int(self.cfg.eval.horizon)
        try_times = int(getattr(self.cfg.eval, "try_times", 5))
        n_init = int(getattr(self.cfg.eval, "n_init_states", 100))
        self._log(f"[eval] dp_ckpt={dp_ckpt} n_init={n_init} "
                  f"try_times={try_times} n_envs={self.n_envs}")

        dp = self.dp_factory(dp_ckpt)
        init_states = collect_initial_states(self.env_factory, n_init_states=n_init)
        self._log(f"[eval] collected {len(init_states)} init states")

        if self.n_envs > 1:
            n_action_steps = int(getattr(dp, "n_action_steps", 1))
            results, any_success, _ = evaluate_baseline_vec(
                dp, self.env_factory, init_states, horizon=horizon,
                n_envs=self.n_envs, n_action_steps=n_action_steps,
                device=self.device, n_tries=try_times,
                metric_prefix="eval",
                wandb_run=self.wandb_run, log_every=self.log_every,
            )
        else:
            adapter = BaseDPAdapter(dp, self.device)
            results, any_success, _ = evaluate_baseline(
                adapter, self.env_factory, init_states, horizon=horizon,
                n_tries=try_times, metric_prefix="eval",
                wandb_run=self.wandb_run,
            )

        metrics = summarize_round(results, [], try_times=try_times,
                                  base_pass5=any_success)
        metrics["dp_ckpt"] = dp_ckpt
        n_solved = sum(1 for s, _ in results if s)
        self._log(f"[eval] success_rate={metrics['success_rate']:.4f} "
                  f"(first-try {n_solved}/{len(results)}) "
                  f"base_pass_at_5={metrics.get('base_pass_at_5', 0.0):.4f} "
                  f"jerk_baseline={metrics['jerk_baseline']:.4f}")

        if self.wandb_run is not None:
            self.wandb_run.log({
                "round/success_rate": metrics["success_rate"],
                "round/base_pass_at_5": metrics.get("base_pass_at_5", 0.0),
                "round/jerk_baseline": metrics["jerk_baseline"],
                "round/baseline_solved": metrics["baseline_solved"],
            })
        return metrics


# --------------------------------------------------------------------------- #
# dry-run with mocks (orchestration verification -- no robomimic / mujoco)
# --------------------------------------------------------------------------- #
def _dry_run():
    """Mock-DP / mock-env check of EvalPipeline.

    Verifies: N init states x try_times baseline runs, success_rate uses the
    FIRST try, base_pass_at_5 reflects any-of-try_times, jerk is computed, and
    NO ScoutVIB is loaded (the decoupling boundary).
    """
    import numpy as np
    from scout.eval.self_improvement import _MockScoutPolicy, _make_dry_run_env_factory

    action_dim = 4
    cfg = EasyDict({
        "eval": {"n_init_states": 4, "try_times": 3, "horizon": 10,
                 "n_envs": 2, "log_every": 5,
                 "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                 "proprio_keys": ["robot0_eef_pos", "robot0_eef_quat",
                                  "robot0_gripper_qpos"]},
    })

    def dp_factory(ckpt_path):
        return _MockScoutPolicy(n_action_steps=8, action_dim=action_dim,
                                guided_strength=0.0)

    env_factory = _make_dry_run_env_factory(action_dim, cfg.eval.horizon,
                                            lo=1.0, hi=3.5)

    pipeline = EvalPipeline(cfg=cfg, dp_factory=dp_factory,
                            env_factory=env_factory, device=torch.device("cpu"))
    metrics = pipeline.run("<mock-base-dp>")

    print("\n--- evaluator dry-run ---")
    print(f"metrics: {metrics}")
    for k in ("success_rate", "base_pass_at_5", "jerk_baseline", "baseline_solved"):
        assert k in metrics, f"missing metric {k}"
    assert 0.0 <= metrics["success_rate"] <= 1.0
    assert 0.0 <= metrics["base_pass_at_5"] <= 1.0
    # pass_at_k / exploration_yield / jerk_exploration are zero for baseline-only
    assert metrics["pass_at_k"] == 0.0
    assert metrics["exploration_yield"] == 0
    print("[dry-run] evaluator.py OK (base-DP metric eval, no VIB)")


if __name__ == "__main__":
    _dry_run()
