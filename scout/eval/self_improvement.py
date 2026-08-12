"""E4: multi-round self-improvement loop (Phase 5.3 / scout_design.md §5).

Thin orchestration layer that composes the two decoupled components:

  * :class:`scout.eval.evaluator.EvalPipeline`  -- base-DP metric evaluation
    (success_rate / jerk / base_pass_at_5 over N init states x try_times).
  * :class:`scout.eval.collector.RolloutCollector` -- VIB-guided successful-
    trajectory collection over ALL init states (data for the next round).

Per round (``cfg.self_improvement.num_rounds``, default 6):

  1. **eval**      -- EvalPipeline measures DP_i baseline metrics.
  2. **collect**   -- RolloutCollector runs VIB-guided exploration over ALL init
                      states; every successful traj (with obs) is accumulated.
  3. **retrain**   -- write augmented hdf5 (core demos + accumulated rollouts)
                      -> retrain via LPB train.py -> DP_{i+1} ckpt path.
                      (SKIPPED after the last round.)

The ScoutVIB is loaded fresh inside RolloutCollector each round (design §5:
VIB dynamics don't change; reloading is cheap vs. the rollout itself).

This is the ONLY place eval + rollout + retrain are coupled. For standalone
metric measurement use ``run_eval.py``; for standalone data collection use
``run_rollout.py`` -- neither touches this loop.

The dry-run (``python -m scout.eval.self_improvement``) wires mock policy /
mock VIB / mock env / mock retrain_fn and verifies ONE + TWO round(s)
end-to-end.
"""

from __future__ import annotations

import os
from typing import List, Optional

import numpy as np
import torch
from easydict import EasyDict

from scout.eval.collector import RolloutCollector
from scout.eval.evaluator import EvalPipeline
from scout.eval.factories import (
    DPFactory,
    EnvFactory,
    RetrainFn,
    VIBFactory,
    load_cfg,
    make_default_env_factory,
    make_lpb_dp_factory,
    make_scout_vib_factory,
)
from scout.eval.hdf5_writer import write_rollouts_to_hdf5

# backward-compat alias: run_round.py and older callers import
# ``_write_augmented_hdf5`` from here. The canonical name in the new
# hdf5_writer module is ``write_rollouts_to_hdf5``.
_write_augmented_hdf5 = write_rollouts_to_hdf5


# --------------------------------------------------------------------------- #
# loop (thin orchestrator over EvalPipeline + RolloutCollector)
# --------------------------------------------------------------------------- #
class SelfImprovementLoop:
    """Multi-round DP self-improvement: eval -> guided collect -> retrain.

    Args mirror the 5-step design. ``dp_factory`` / ``scout_vib_factory`` /
    ``env_factory`` / ``retrain_fn`` are all injected so the loop has no hard
    dependency on robomimic / mujoco / a particular ckpt path (the dry-run
    injects mocks).

    The loop ALWAYS runs guided collection (the SOE self-improvement protocol).
    For eval-only use :class:`EvalPipeline` directly (or ``run_eval.py``); for
    unguided / guided-only collection use :class:`RolloutCollector` (or
    ``run_rollout.py``).
    """

    def __init__(self, cfg: EasyDict, dp_factory: DPFactory,
                 scout_vib_factory: VIBFactory, env_factory: EnvFactory,
                 retrain_fn: RetrainFn,
                 device: Optional[torch.device] = None,
                 verbose: bool = True, wandb_run=None):
        self.cfg = cfg
        self.dp_factory = dp_factory
        self.scout_vib_factory = scout_vib_factory
        self.env_factory = env_factory
        self.retrain_fn = retrain_fn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        self.wandb_run = wandb_run

        # the two decoupled components (eval + rollout), sharing cfg / factories
        self.evaluator = EvalPipeline(
            cfg=cfg, dp_factory=dp_factory, env_factory=env_factory,
            device=self.device, wandb_run=wandb_run)
        self.collector = RolloutCollector(
            cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
            env_factory=env_factory, device=self.device, guided=True,
            wandb_run=wandb_run)

        # bookkeeping
        self.dp_path: str = cfg.base_dp.initial_ckpt_path
        self.history: List[dict] = []
        self.accumulated_rollouts: List[dict] = []   # grows each round

    def _log(self, *a, **kw):
        if self.verbose:
            print(*a, **kw)

    def run(self, num_rounds: Optional[int] = None) -> List[dict]:
        """Run ``num_rounds`` rounds (defaults to cfg.self_improvement.num_rounds).

        Per round: EvalPipeline metrics -> RolloutCollector guided collection ->
        write-back (augmented hdf5 + retrain) -> DP_{i+1}. The retrain is
        SKIPPED after the last round (nothing to feed the next round).

        Returns the per-round summaries (also stored in ``self.history``).
        """
        n_rounds = int(num_rounds or self.cfg.self_improvement.num_rounds)
        try_times = int(getattr(self.cfg.eval, "try_times", 5))
        vib_ckpt = getattr(self.cfg.vib, "ckpt_path", None)
        self._log(f"[loop] rounds={n_rounds} try_times={try_times} "
                  f"n_envs={self.evaluator.n_envs} guided=True")

        for r in range(n_rounds):
            self._log(f"\n=== round {r} ===  (dp_ckpt={self.dp_path})")

            # 1. eval: base-DP metrics (success_rate, jerk, base_pass_at_5)
            metrics = self.evaluator.run(self.dp_path)
            self._log(f"  eval: success_rate={metrics['success_rate']:.4f} "
                      f"base_pass_at_5={metrics.get('base_pass_at_5', 0.0):.4f}")

            # 2. collect: VIB-guided exploration over ALL init states
            trajs = self.collector.run(self.dp_path, vib_ckpt=vib_ckpt)
            self.accumulated_rollouts.extend(trajs)
            self._log(f"  collected {len(trajs)} guided trajs "
                      f"(accumulated {len(self.accumulated_rollouts)})")

            summary = dict(metrics)
            summary["round"] = r
            summary["dp_ckpt"] = self.dp_path
            summary["collected"] = len(trajs)
            summary["accumulated"] = len(self.accumulated_rollouts)
            self.history.append(summary)

            if self.wandb_run is not None:
                self.wandb_run.log({
                    "round/round": r,
                    "round/success_rate": metrics["success_rate"],
                    "round/base_pass_at_5": metrics.get("base_pass_at_5", 0.0),
                    "round/yield": len(trajs),
                    "round/accumulated": len(self.accumulated_rollouts),
                })

            # 3. retrain (between rounds only -- skip after the last round)
            if r + 1 < n_rounds:
                if not self.accumulated_rollouts:
                    raise RuntimeError(
                        f"[loop] 0 accumulated rollouts after round {r} -- "
                        "nothing to retrain on. Check guidance liveness + β.")
                new_dp_path = self.retrain_fn(
                    self.cfg, r, self.accumulated_rollouts, self.dp_path)
                self.dp_path = new_dp_path
                self._log(f"  retrain -> new dp_ckpt={new_dp_path}")

        return self.history


# --------------------------------------------------------------------------- #
# default retrain_fn (writes augmented HDF5 + calls scout.train_base_dp.train)
# --------------------------------------------------------------------------- #
def default_retrain_fn_factory(log_root: str) -> RetrainFn:
    """Build the default retrain callback.

    Returns a closure that, on each round, writes an augmented HDF5
    (core demos + accumulated successful rollouts as new demo_*) and invokes
    :func:`scout.train_base_dp.train` -- which shells out to the LPB ``train.py``
    (repo root) with ``cfg.base_dp.config_name`` overridden to point at the new
    hdf5 + per-round log dir. Real run only -- the dry-run swaps in a stub.
    """

    def retrain_fn(cfg: EasyDict, round_idx: int,
                   successful_rollouts: List[dict],
                   prev_dp_ckpt: str) -> str:
        from scout.train_base_dp import train

        # 1. write augmented hdf5 (core + successful rollouts, scout_aug mask)
        core_path = cfg.dataset.path
        round_dir = os.path.join(log_root, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        new_path = os.path.join(round_dir, "augmented.hdf5")
        aug_mask_key = cfg.self_improvement.scout_aug_mask
        write_rollouts_to_hdf5(core_path, new_path, successful_rollouts,
                               core_filter_key=cfg.dataset.core_filter_key,
                               aug_mask_key=aug_mask_key)

        # 2. retrain via LPB train.py -- per-round log_dir, train on scout_aug mask
        new_ckpt = train(
            config_name=cfg.base_dp.config_name,
            config_dir=cfg.base_dp.config_dir,
            dataset_path=new_path,
            train_filter_key=aug_mask_key,        # core + new rollouts
            log_dir=round_dir,
            num_epochs=int(cfg.self_improvement.num_epochs),
            # SCOUT threads the prior DP as the resume point (LPB workspace
            # resumes from <log_dir>/checkpoints/latest.ckpt when
            # training.resume=True; for true round-to-round init, copy
            # prev_dp_ckpt into round_dir/checkpoints/latest.ckpt before train).
            extra_overrides={"training.resume": False},
        )
        return new_ckpt

    return retrain_fn


# --------------------------------------------------------------------------- #
# dry-run mocks (shared by evaluator / collector / loop dry-runs)
# --------------------------------------------------------------------------- #
class _MockScoutPolicy(torch.nn.Module):
    """Mock of ScoutPolicy for the dry-run. Exposes the LPB interface the loop
    + adapters actually call: ``n_action_steps``, ``predict_action``,
    ``predict_action_dyn_guided``, ``normalizer``, ``reset``, ``eval``,
    ``initialize_scout_planner``. Unguided + guided actions differ so the
    scripted env sees different success rates (exercising the success filter)."""

    def __init__(self, n_action_steps=8, action_dim=4, guided_strength=0.0):
        super().__init__()
        self.n_action_steps = n_action_steps
        self.action_dim = action_dim
        self.guided_strength = float(guided_strength)
        self.normalizer = {}        # make_action_bridge falls back to Identity
        self._planner_attached = False

    def reset(self):
        pass

    def eval(self):
        return self

    def initialize_scout_planner(self, planner, gst, gscale):
        self._planner_attached = True

    def _chunk(self, B, guided):
        # baseline: small action[0]; guided: amplified -- so the scripted env
        # (threshold over |action|.sum()) solves more in exploration.
        mag = (0.2 + self.guided_strength) if guided else 0.2
        c = torch.zeros((B, self.n_action_steps, self.action_dim))
        c[..., 0] = mag
        return c

    def predict_action(self, obs_dict):
        B = next(iter(obs_dict.values())).shape[0]
        c = self._chunk(B, guided=False)
        return {"action": c, "action_pred": c}

    def predict_action_dyn_guided(self, obs_dict):
        B = next(iter(obs_dict.values())).shape[0]
        c = self._chunk(B, guided=True)
        return {"action": c, "action_pred": c}


class _MockScoutVIB(torch.nn.Module):
    """Mock ScoutVIB: just the attrs ScoutPlanner touches (style_dim + eval)."""

    def __init__(self, style_dim=8):
        super().__init__()
        self.style_dim = style_dim

    def eval(self):
        return self

    def encode(self, obs):
        # dim-agnostic placeholder s̄ (planner caches it but the mock policy
        # never calls into the real cost path).
        return torch.zeros(1, self.style_dim)


def _make_dry_run_env_factory(action_dim, horizon, lo, hi):
    class MockEnv:
        def __init__(self, seed=0):
            self.action_dim = action_dim
            self.horizon = horizon
            self._step = 0
            self._cum = 0.0
            self._state_dict = {"s": 0.0}
            self._rng = np.random.default_rng(seed)
            self.rollout_exceptions = ()

        def reset(self):
            self._step = 0
            self._cum = 0.0
            self._state_dict = {"s": float(self._rng.uniform(lo, hi))}
            return self._get_obs()

        def reset_to(self, state_dict):
            self._step = 0
            self._cum = 0.0
            self._state_dict = dict(state_dict)
            return self._get_obs()

        def _get_obs(self):
            return {"agentview_image": np.zeros((2, 3, 4, 4), dtype=np.float32),
                    "robot0_eye_in_hand_image": np.ones((2, 3, 4, 4), dtype=np.float32),
                    "robot0_eef_pos": np.zeros((2, action_dim // 2), dtype=np.float32),
                    "robot0_eef_quat": np.zeros((2, 4), dtype=np.float32),
                    "robot0_gripper_qpos": np.zeros((2, 2), dtype=np.float32)}

        def step(self, action):
            self._step += 1
            self._cum += float(np.abs(action).sum())
            done = self.is_success()["task"] or (self._step >= self.horizon)
            return self._get_obs(), 0.0, done, {}

        def is_success(self):
            return {"task": bool(self._cum >= self._state_dict["s"])}

        def get_state(self):
            return dict(self._state_dict)

        def close(self):
            pass

    return lambda: MockEnv()


# --------------------------------------------------------------------------- #
# dry-run: orchestration verification (eval -> guided collect -> [retrain])
# --------------------------------------------------------------------------- #
def _dry_run():
    """Mock ONE-round orchestration check of the self-improvement loop.

    Verifies EvalPipeline + RolloutCollector compose end-to-end: eval metrics
    computed -> guided trajs collected -> retrain_fn fires (when not last round).
    """
    action_dim = 4
    cfg = EasyDict({
        "base_dp": {"initial_ckpt_path": "<mock-dp-0>",
                    "config_name": "base_dp_lift_image", "config_dir": "configs"},
        "vib": {"ckpt_path": "<mock-vib>"},
        "dataset": {"path": "<mock-core>", "core_filter_key": "train"},
        "action_dim": action_dim,
        "eval": {"n_init_states": 4, "try_times": 3, "horizon": 10,
                 "n_envs": 2, "log_every": 5,
                 "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                 "proprio_keys": ["robot0_eef_pos", "robot0_eef_quat",
                                  "robot0_gripper_qpos"]},
        "exploration": {"guidance_scale": 5.0, "guidance_start_timestep": 50},
        "self_improvement": {"num_rounds": 1, "scout_aug_mask": "scout_aug",
                             "num_epochs": 1},
    })

    def dp_factory(ckpt_path):
        return _MockScoutPolicy(n_action_steps=8, action_dim=action_dim,
                                guided_strength=0.6)

    def scout_vib_factory(vib_ckpt=None):
        return _MockScoutVIB(style_dim=8)

    env_factory = _make_dry_run_env_factory(action_dim, cfg.eval.horizon,
                                            lo=2.5, hi=5.0)
    retrain_calls: List[tuple] = []

    def mock_retrain_fn(c, round_idx, successful_rollouts, prev_dp_ckpt):
        retrain_calls.append((round_idx, len(successful_rollouts), prev_dp_ckpt))
        return f"<mock-dp-{round_idx + 1}>"

    loop = SelfImprovementLoop(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
        env_factory=env_factory, retrain_fn=mock_retrain_fn,
        device=torch.device("cpu"),
    )
    history = loop.run()

    print("\n--- dry-run results ---")
    print(f"rounds run         : {len(history)}")
    print(f"history[0]         : {history[0]}")
    print(f"retrain calls      : {retrain_calls}")
    print(f"accumulated rollouts: {len(loop.accumulated_rollouts)}")

    assert len(history) == 1, "expected 1 round"
    h = history[0]
    assert "success_rate" in h and "collected" in h, h
    assert h["collected"] == len(loop.accumulated_rollouts)
    # num_rounds=1 -> last round skips retrain
    assert retrain_calls == [], "(num_rounds=1 -> no retrain expected)"
    print("[dry-run] self_improvement.py OK (eval + collect orchestration)")


def _dry_run_two_rounds():
    """Two-round variant: retrain fires after round 0; its ckpt feeds round 1."""
    action_dim = 4
    cfg = EasyDict({
        "base_dp": {"initial_ckpt_path": "<mock-dp-0>",
                    "config_name": "base_dp_lift_image", "config_dir": "configs"},
        "vib": {"ckpt_path": "<mock-vib>"},
        "dataset": {"path": "<mock-core>", "core_filter_key": "train"},
        "action_dim": action_dim,
        "eval": {"n_init_states": 3, "try_times": 2, "horizon": 8,
                 "n_envs": 2, "log_every": 5,
                 "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                 "proprio_keys": ["robot0_eef_pos", "robot0_eef_quat",
                                  "robot0_gripper_qpos"]},
        "exploration": {"guidance_scale": 1.0, "guidance_start_timestep": 50},
        "self_improvement": {"num_rounds": 2, "scout_aug_mask": "scout_aug",
                             "num_epochs": 1},
    })

    def dp_factory(ckpt_path):
        return _MockScoutPolicy(n_action_steps=8, action_dim=action_dim,
                                guided_strength=0.4)

    vib_factory = lambda vib_ckpt=None: _MockScoutVIB(style_dim=8)
    env_factory = _make_dry_run_env_factory(action_dim, cfg.eval.horizon,
                                            lo=1.5, hi=3.0)
    retrain_calls = []

    def retrain_fn(c, r_idx, rollouts, prev):
        retrain_calls.append((r_idx, len(rollouts), prev))
        return f"<mock-dp-{r_idx + 1}>"

    loop = SelfImprovementLoop(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=vib_factory,
        env_factory=env_factory, retrain_fn=retrain_fn,
        device=torch.device("cpu"), verbose=False,
    )
    history = loop.run()

    print("\n--- two-round dry-run ---")
    print(f"rounds run: {len(history)}")
    print(f"retrain calls: {retrain_calls}")
    print(f"final dp_path: {loop.dp_path}")
    print(f"history: {[(h['round'], round(h['success_rate'], 3), h['collected']) for h in history]}")
    assert len(history) == 2, "expected 2 rounds"
    assert len(retrain_calls) == 1, "expected 1 retrain (after round 0)"
    assert retrain_calls[0][1] > 0, "retrain should receive non-empty rollouts"
    assert retrain_calls[0][2] == "<mock-dp-0>", "round 0 should start from initial"
    assert loop.dp_path == "<mock-dp-1>", "round 1 should use retrain's output"
    print("[dry-run-2] self_improvement.py OK (retrain wiring)")


if __name__ == "__main__":
    _dry_run()
    _dry_run_two_rounds()
