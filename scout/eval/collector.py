"""RolloutCollector -- successful-trajectory collection, fully decoupled from
metric evaluation.

Given a base-DP checkpoint (and, optionally, a SCOUT dynamics-model / VIB
checkpoint), this rolls out N (default 100) random initial states x
``try_times`` (default 5) tries and keeps EVERY successful trajectory (with
per-frame obs, ready for hdf5 write-back). Two collection drivers:

  * **guided=False** (default; ``--guide off``): raw base DP via
    ``predict_action`` under ``torch.no_grad``. No dynamics model needed --
    equivalent to the "base-DP 100x5" collection.
  * **guided=True** (``--guide dyn``): SCOUT VIB-guided exploration via
    ``predict_action_dyn_guided`` (classifier guidance on the dynamics model).
    Runs over ALL init states (``only_failed_of=None``), each rollout locking a
    fresh ``z ~ N(0,I)``. Needs a VIB ckpt + planner attachment.

This is the rollout half of the eval/rollout decoupling: it produces data, not
metrics. :class:`EvalPipeline` never touches this; the self-improvement loop
composes the two.

Parallelism: ``n_envs>1`` drives the single-process vectorized engine; ``==1``
falls back to the sequential path. Live wandb progress streams as ``explore/*``
(``init_done``, ``tries_done``, ``collected``, ``yield``) -- the same dashboard
keys for guided and unguided collection.

Run via ``python -m scout.eval.run_rollout`` (real run) or
``python -m scout.eval.collector`` (mock dry-run, both guided + unguarded).
"""

from __future__ import annotations

from typing import List, Optional

import torch
from easydict import EasyDict

from scout.eval.factories import DPFactory, EnvFactory, VIBFactory
from scout.eval.rollout import (
    BaseDPAdapter,
    GuidedAdapter,
    collect_initial_states,
    evaluate_baseline,
    evaluate_exploration,
    make_action_bridge,
    make_obs_adapter,
)
from scout.eval.rollout_vec import (
    evaluate_baseline_vec,
    evaluate_exploration_vec,
)


class RolloutCollector:
    """Successful-trajectory collector: N init states x try_times -> List[traj].

    Parameters
    ----------
    cfg : EasyDict
        Eval config. Reads ``cfg.eval.{horizon, n_init_states, try_times,
        n_envs, log_every}`` and (guided only) ``cfg.exploration.{guidance_scale,
        guidance_start_timestep}`` + ``cfg.eval.{view_names, proprio_keys}``.
    dp_factory : callable
        ``dp_factory(ckpt_path) -> ScoutPolicy``.
    scout_vib_factory : callable or None
        ``scout_vib_factory() -> ScoutVIB``. REQUIRED when ``guided=True``;
        ignored (may be None) when ``guided=False``.
    env_factory : callable
        ``env_factory() -> env``.
    device : torch.device
    guided : bool
        True = VIB-guided exploration (needs scout_vib_factory + vib_ckpt);
        False = raw base DP (no dynamics model).
    wandb_run : optional
        Live ``explore/*`` + ``round/*`` progress (None disables logging).
    """

    def __init__(self, cfg: EasyDict, dp_factory: DPFactory,
                 scout_vib_factory: Optional[VIBFactory],
                 env_factory: EnvFactory,
                 device: Optional[torch.device] = None,
                 guided: bool = False, wandb_run=None):
        self.cfg = cfg
        self.dp_factory = dp_factory
        self.scout_vib_factory = scout_vib_factory
        self.env_factory = env_factory
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.guided = bool(guided)
        self.wandb_run = wandb_run
        self.n_envs = int(getattr(cfg.eval, "n_envs", 1))
        self.log_every = int(getattr(cfg.eval, "log_every", 10))
        self.verbose = bool(getattr(cfg, "verbose", True))
        # guidance / obs-adapter config (read once; only used when guided)
        self.guidance_scale = float(getattr(cfg, "exploration", {})
                                    .get("guidance_scale", 5.0))
        self.guidance_start_timestep = int(
            getattr(cfg, "exploration", {}).get("guidance_start_timestep", 50))
        self.view_names = list(getattr(cfg.eval, "view_names", []))
        self.proprio_keys = list(getattr(cfg.eval, "proprio_keys", []))
        # collected successful trajs (obs included), grown across runs
        self.collected: List[dict] = []

    def _log(self, *a, **kw):
        if self.verbose:
            print(*a, **kw)

    def _attach_planner(self, dp, scout_vib):
        """Attach the SCOUT planner to ``dp`` (seam ① + ②). Idempotent.

        Per design §4: planner carries the frozen ScoutVIB (``E_s`` +
        ``vib_enc``) + the unnormalize-only action bridge (from this DP's
        normalizer) + the obs-adapter (LPB keyed obs -> E_s format). z is
        sampled fresh per rollout inside the vec runner / GuidedAdapter.

        If ``dp`` has no ``initialize_scout_planner`` hook (e.g. a mock), skip
        -- the mock must already mimic the guided interface.
        """
        from scout.guidance.planner import ScoutPlanner
        bridge = make_action_bridge(dp)                      # seam ②
        obs_adapter = make_obs_adapter(self.view_names,      # seam ①
                                       self.proprio_keys)
        planner = ScoutPlanner(scout_vib, bridge=bridge,
                               obs_adapter=obs_adapter, z=None)
        init = getattr(dp, "initialize_scout_planner", None)
        if callable(init):
            init(planner, self.guidance_start_timestep, self.guidance_scale)

    def run(self, dp_ckpt: str,
            vib_ckpt: Optional[str] = None) -> List[dict]:
        """Roll out N init states x try_times, return EVERY successful traj.

        ``vib_ckpt`` is forwarded to ``scout_vib_factory`` only when
        ``self.guided``; ignored otherwise. Returned trajs carry per-frame
        ``obs``/``next_obs`` (collected with ``record_obs=True``) so they feed
        directly into :func:`hdf5_writer.write_rollouts_to_hdf5`.
        """
        horizon = int(self.cfg.eval.horizon)
        try_times = int(getattr(self.cfg.eval, "try_times", 5))
        n_init = int(getattr(self.cfg.eval, "n_init_states", 100))
        self._log(f"[rollout] dp_ckpt={dp_ckpt} guided={self.guided} "
                  f"n_init={n_init} try_times={try_times} n_envs={self.n_envs}"
                  + (f" vib_ckpt={vib_ckpt}" if self.guided else ""))

        dp = self.dp_factory(dp_ckpt)
        init_states = collect_initial_states(self.env_factory, n_init_states=n_init)
        self._log(f"[rollout] collected {len(init_states)} init states")

        if self.guided:
            if self.scout_vib_factory is None:
                raise ValueError("guided=True requires a scout_vib_factory")
            scout_vib = (self.scout_vib_factory(vib_ckpt)
                         if vib_ckpt is not None else self.scout_vib_factory())
            self._attach_planner(dp, scout_vib)
            if self.n_envs > 1:
                n_action_steps = int(getattr(dp, "n_action_steps", 1))
                expl = evaluate_exploration_vec(
                    dp, self.env_factory, init_states, horizon=horizon,
                    try_times=try_times, n_envs=self.n_envs,
                    n_action_steps=n_action_steps, device=self.device,
                    only_failed_of=None,            # ALL init states
                    wandb_run=self.wandb_run, log_every=self.log_every,
                )
            else:
                adapter = GuidedAdapter(dp, self.device)
                expl = evaluate_exploration(
                    adapter, self.env_factory, init_states, horizon=horizon,
                    try_times=try_times, only_failed_of=None,
                    wandb_run=self.wandb_run,
                )
            trajs = [t for entry in expl for t in entry["successful_trajs"]]
        else:
            # unguided: raw base DP over ALL init states, keep EVERY success.
            if self.n_envs > 1:
                n_action_steps = int(getattr(dp, "n_action_steps", 1))
                _, _, trajs = evaluate_baseline_vec(
                    dp, self.env_factory, init_states, horizon=horizon,
                    n_envs=self.n_envs, n_action_steps=n_action_steps,
                    device=self.device, n_tries=try_times, record_obs=True,
                    metric_prefix="explore",
                    wandb_run=self.wandb_run, log_every=self.log_every,
                )
            else:
                adapter = BaseDPAdapter(dp, self.device)
                _, _, trajs = evaluate_baseline(
                    adapter, self.env_factory, init_states, horizon=horizon,
                    n_tries=try_times, record_obs=True, metric_prefix="explore",
                    wandb_run=self.wandb_run,
                )

        self.collected.extend(trajs)
        self._log(f"[rollout] collected {len(trajs)} successful trajs "
                  f"(total {len(self.collected)})")
        if self.wandb_run is not None:
            self.wandb_run.log({
                "round/collected": len(trajs),
                "round/yield": len(trajs),
                "round/guided": int(self.guided),
            })
        return trajs


# --------------------------------------------------------------------------- #
# dry-run with mocks (guided + unguided paths)
# --------------------------------------------------------------------------- #
def _dry_run():
    """Mock check of RolloutCollector for both guided and unguided drivers.

    Verifies: N init states x try_times runs, EVERY successful traj is kept
    (with obs for hdf5 write-back), guided=True loads the VIB + planner
    (only_failed_of=None over ALL init states), guided=False skips the VIB
    entirely (the decoupling boundary).
    """
    import copy
    from scout.eval.self_improvement import _MockScoutPolicy, _MockScoutVIB, _make_dry_run_env_factory

    action_dim = 4
    base_cfg = EasyDict({
        "eval": {"n_init_states": 3, "try_times": 2, "horizon": 8,
                 "n_envs": 2, "log_every": 5,
                 "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                 "proprio_keys": ["robot0_eef_pos", "robot0_eef_quat",
                                  "robot0_gripper_qpos"]},
        "exploration": {"guidance_scale": 1.0, "guidance_start_timestep": 50},
    })

    for guided, label in ((True, "guided"), (False, "unguided")):
        cfg = copy.deepcopy(base_cfg)
        vib_calls: List[int] = []

        def dp_factory(ckpt_path):
            return _MockScoutPolicy(n_action_steps=8, action_dim=action_dim,
                                    guided_strength=0.6)

        def scout_vib_factory(vib_ckpt=None):
            vib_calls.append(1)
            return _MockScoutVIB(style_dim=8)

        env_factory = _make_dry_run_env_factory(action_dim, cfg.eval.horizon,
                                                lo=1.0, hi=3.5)
        collector = RolloutCollector(
            cfg=cfg, dp_factory=dp_factory, scout_vib_factory=scout_vib_factory,
            env_factory=env_factory, device=torch.device("cpu"), guided=guided,
        )
        trajs = collector.run("<mock-base-dp>",
                              vib_ckpt="<mock-vib>" if guided else None)

        assert len(trajs) == len(collector.collected)
        if trajs:
            assert "obs" in trajs[0], "rollout must record obs for hdf5 write-back"
        assert len(vib_calls) == (1 if guided else 0), \
            f"vib_calls={len(vib_calls)} for guided={guided}"
        print(f"[dry-run-rollout/{label}] OK: collected={len(trajs)} "
              f"vib_calls={len(vib_calls)}")
    print("[dry-run] collector.py OK (guided + unguided collection)")


if __name__ == "__main__":
    _dry_run()
