"""E4: 5-step multi-round self-improvement loop (Phase 5.3,
scout_design.md §5, scout_impl_plan.md Task 5.3).

Per round (6 rounds by default; config: num_rounds):

  1. DP_i            -- loaded from ckpt (round 0 = the Phase-2 output DP_0).
  2. ScoutVIB        -- loaded ONCE from the Phase-3 chosen-β ckpt; reused
                        across all rounds.
  3. Rollouts        -- baseline DP_i on N init states (1 try each); guided
                        exploration (GuidedAdapter with z~N(0,I), guidance_scale)
                        on the failed init states (up to try_times tries).
  4. Write-back      -- successful exploration rollouts -> source.add
                        (Phase-1 TransitionSource) -> merge-with-core ->
                        ``retrain_fn`` -> DP_{i+1} ckpt path.
  5. Metrics         -- success_rate / pass@k / yield / jerk (DP_{i+1} vs DP_i).

Real E4 (full robomimic env + mujoco + trained ckpts + GPU) is DEFERRED. To
keep the orchestration unit-testable in this dev env, every external dependency
is a factory passed into :class:`SelfImprovementLoop`:

  dp_factory(ckpt_path)        -> a fresh :class:`scout.policy.dp.DP`
                                  (state_dict loaded).
  scout_vib_factory()          -> a fresh :class:`scout.model.scout_vib.ScoutVIB`
                                  (state_dict loaded).
  env_factory()                -> an env (robomimic EnvBase or a mock).
  retrain_fn(cfg, round_idx,   -> path to the DP_{i+1} ckpt. The default impl
    successful_rollouts,         writes an augmented HDF5 (core + successful
    prev_dp_ckpt_path)           rollouts) and calls train_base_dp.train, but
                                  it's an injected dependency so the dry-run
                                  substitutes a no-op stub.
  source_factory()             -> a fresh :class:`TransitionSource` used to
                                  write back the round's successful transitions
                                  (default RobomimicLowdimSource on the core
                                  dataset; mock in the dry-run).

The dry-run (``python -m scout.eval.self_improvement``) wires a mock DP /
mock ScoutVIB / mock env / mock retrain_fn and verifies ONE round end-to-end:
loop iterates -> baseline runs -> guided exploration runs -> successful
rollouts are filtered and source.add is called -> retrain_fn fires -> metric
compare is recorded.
"""

from __future__ import annotations

import argparse
import os
import time
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import yaml
from easydict import EasyDict

from scout.data.transition_source import TransitionSource
from scout.eval.metrics import summarize_round
from scout.eval.rollout import (
    BaseDPAdapter,
    GuidedAdapter,
    collect_initial_states,
    evaluate_baseline,
    evaluate_exploration,
    rollout_to_transitions,
)


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
DPFactory = Callable[[str], torch.nn.Module]
VIBFactory = Callable[[], torch.nn.Module]
EnvFactory = Callable[[], Any]
SourceFactory = Callable[[], TransitionSource]
RetrainFn = Callable[
    [EasyDict, int, List[dict], str],  # cfg, round_idx, successful_rollouts, prev_dp_ckpt
    str,                                # new DP ckpt path
]


# --------------------------------------------------------------------------- #
# loop
# --------------------------------------------------------------------------- #
class SelfImprovementLoop:
    """Multi-round DP self-improvement via VIB-guided exploration.

    Args mirror the 5-step design (above). All factories are required so the
    loop has no hard dependency on robomimic / mujoco / a particular ckpt path.
    """

    def __init__(self,
                 cfg: EasyDict,
                 dp_factory: DPFactory,
                 scout_vib_factory: VIBFactory,
                 env_factory: EnvFactory,
                 retrain_fn: RetrainFn,
                 source_factory: Optional[SourceFactory] = None,
                 state_to_vec: Optional[Callable] = None,
                 device: Optional[torch.device] = None,
                 verbose: bool = True):
        self.cfg = cfg
        self.dp_factory = dp_factory
        self.scout_vib_factory = scout_vib_factory
        self.env_factory = env_factory
        self.retrain_fn = retrain_fn
        self.source_factory = source_factory
        self.state_to_vec = state_to_vec
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose

        # bookkeeping
        self.dp_path: str = cfg.base_dp.initial_ckpt_path
        self.history: List[dict] = []
        self.accumulated_rollouts: List[dict] = []   # grows each round

    def _log(self, *a, **kw):
        if self.verbose:
            print(*a, **kw)

    # ---- per-round steps ------------------------------------------------- #
    def _baseline_round(self, dp, init_states) -> List[Tuple[bool, dict]]:
        adapter = BaseDPAdapter(dp, self.device)
        return evaluate_baseline(adapter, self.env_factory, init_states,
                                 horizon=int(self.cfg.eval.horizon))

    def _exploration_round(self, dp, scout_vib, init_states,
                           baseline_results) -> List[dict]:
        if self.state_to_vec is None:
            raise ValueError(
                "GuidedAdapter requires `state_to_vec`; pass it to "
                "SelfImprovementLoop (env-obs -> (state_dim,) tensor).")
        adapter = GuidedAdapter(
            dp, self.device, scout_vib, self.state_to_vec,
            guidance_scale=float(self.cfg.exploration.guidance_scale),
            guidance_start_timestep=int(
                self.cfg.exploration.guidance_start_timestep),
            bridge=None,                                  # Identity in stage-1
        )
        return evaluate_exploration(
            adapter, self.env_factory, init_states,
            horizon=int(self.cfg.eval.horizon),
            try_times=int(self.cfg.eval.try_times),
            only_failed_of=baseline_results,
        )

    def _collect_successful(self,
                            exploration_results: List[dict],
                            obs_keys: Sequence[str],
                            action_dim: int) -> List[dict]:
        """Pull successful rollouts (+ their transitions) from a round's results.

        Returns the list of successful traj dicts (the same objects the loop
        forwards to ``retrain_fn``). As a side-effect, if ``source_factory`` was
        provided, the transitions are written back via ``source.add`` (Phase-1
        write-back entry) -- this is the SOE multi-round loop's retrain data.
        """
        succ_rollouts: List[dict] = []
        if self.source_factory is not None:
            source = self.source_factory()
        else:
            source = None
        for r in exploration_results:
            for traj in r["successful_trajs"]:
                succ_rollouts.append(traj)
                if source is not None:
                    trans = rollout_to_transitions(traj, obs_keys=obs_keys,
                                                   action_dim=action_dim)
                    if trans is not None:
                        source.add(trans)
        self._log(f"  round collected {len(succ_rollouts)} successful rollouts "
                  f"(source len now {len(source) if source else 'n/a'})")
        return succ_rollouts

    # ---- full loop ------------------------------------------------------- #
    def run(self, num_rounds: Optional[int] = None) -> List[dict]:
        """Run ``num_rounds`` rounds (defaults to ``cfg.self_improvement.num_rounds``).

        Per round: baseline DP_i -> guided exploration -> write-back -> retrain
        -> DP_{i+1}. The ScoutVIB is loaded ONCE (round 0) and reused (per
        design: VIB dynamics don't change across rounds).

        Returns the per-round summaries (also stored in ``self.history``).
        """
        n_rounds = int(num_rounds or self.cfg.self_improvement.num_rounds)
        scout_vib = self.scout_vib_factory()

        # N init states are FIXED across rounds for fair metric comparison.
        init_states = collect_initial_states(
            self.env_factory, n_init_states=int(self.cfg.eval.n_init_states))
        self._log(f"[loop] collected {len(init_states)} init states (fixed "
                  f"across {n_rounds} rounds)")

        action_dim = int(self.cfg.action_dim)
        obs_keys = list(self.cfg.eval.get("obs_keys",
                                          ["robot0_eef_pos", "object",
                                           "robot0_gripper_qpos"]))

        for r in range(n_rounds):
            self._log(f"\n=== round {r} ===  (dp_ckpt={self.dp_path})")
            dp = self.dp_factory(self.dp_path)

            baseline = self._baseline_round(dp, init_states)
            self._log(f"  baseline: {sum(1 for s,_ in baseline if s)}/"
                      f"{len(baseline)} solved")

            expl = self._exploration_round(dp, scout_vib, init_states, baseline)
            yield_this = sum(len(r_["successful_trajs"]) for r_ in expl)
            self._log(f"  exploration yield: {yield_this}")

            summary = summarize_round(baseline, expl,
                                      try_times=int(self.cfg.eval.try_times))
            summary["round"] = r
            summary["dp_ckpt"] = self.dp_path
            self.history.append(summary)
            self._log(f"  metrics: success_rate={summary['success_rate']:.4f} "
                      f"pass_at_k={summary['pass_at_k']:.4f} "
                      f"yield={summary['exploration_yield']} "
                      f"jerk_base={summary['jerk_baseline']:.4f} "
                      f"jerk_expl={summary['jerk_exploration']:.4f}")

            # write-back + retrain -> next dp ckpt path
            successful = self._collect_successful(expl, obs_keys, action_dim)
            self.accumulated_rollouts.extend(successful)
            if r + 1 < n_rounds:                  # no retrain needed after last round
                new_dp_path = self.retrain_fn(
                    self.cfg, r, self.accumulated_rollouts, self.dp_path)
                self.dp_path = new_dp_path
                self._log(f"  retrain -> new dp_ckpt={new_dp_path}")

        return self.history


# --------------------------------------------------------------------------- #
# default retrain_fn (writes augmented HDF5 + calls train_base_dp.train)
# --------------------------------------------------------------------------- #
def default_retrain_fn_factory(log_root: str) -> RetrainFn:
    """Build the default retrain callback.

    Returns a closure that, on each round, writes an augmented HDF5
    (core demos + accumulated successful rollouts as new demo_*) and invokes
    :func:`scout.train_base_dp.train` with the cfg's ``base_dp.cfg`` path
    overridden to the new hdf5. Real run only -- the dry-run swaps in a stub.

    .. note:: The HDF5 writer is structurally complete (robomimic ``data/demo_N``
       schema with ``obs/<key>``, ``actions``, ``done``, ``success`` per frame),
       but UNTESTED against a real robomimic loader (env install deferred). If
       the real run trips on schema details, this is the place to fix.
    """

    def retrain_fn(cfg: EasyDict, round_idx: int,
                   successful_rollouts: List[dict],
                   prev_dp_ckpt: str) -> str:
        from scout.train_base_dp import train       # lazy: keeps dry-run hermetic

        # 1. write augmented hdf5
        core_path = cfg.dataset.path
        round_dir = os.path.join(log_root, f"round_{round_idx + 1}")
        os.makedirs(round_dir, exist_ok=True)
        new_path = os.path.join(round_dir, "augmented.hdf5")
        aug_mask_key = cfg.self_improvement.get("scout_aug_mask", "scout_aug")
        _write_augmented_hdf5(core_path, new_path, successful_rollouts,
                              core_filter_key=cfg.self_improvement.core_filter_key,
                              aug_mask_key=aug_mask_key)

        # 2. retrain cfg -- clone, override dataset path + ckpt resume
        import copy
        base_dp_cfg = EasyDict(copy.deepcopy(dict(cfg.base_dp.train_cfg)))
        base_dp_cfg.dataset.path = new_path
        # The augmented HDF5 contains core demos + appended successful rollouts.
        # `train_filter_key` MUST point at a mask that includes BOTH, else the
        # retrain would ignore the new rollouts (the bug we avoid). The augmented
        # file writes such a mask (`mask/<scout_aug_mask>`); point at it here.
        base_dp_cfg.dataset.train_filter_key = cfg.self_improvement.scout_aug_mask
        base_dp_cfg.resume_ckpt = prev_dp_ckpt
        base_dp_cfg.log_dir = round_dir
        if "num_epochs" in cfg.self_improvement:
            base_dp_cfg.num_epochs = int(cfg.self_improvement.num_epochs)
        log_run = train(base_dp_cfg)
        new_ckpt = os.path.join(log_run, "ckpt",
                                f"policy_epoch_{base_dp_cfg.num_epochs}.ckpt")
        return new_ckpt

    return retrain_fn


def _write_augmented_hdf5(core_path: str, out_path: str,
                          rollouts: List[dict],
                          core_filter_key: str = "train",
                          aug_mask_key: str = "scout_aug"):
    """Write ``core_path``'s filtered demos + ``rollouts`` as a new HDF5.

    Mirrors robomimic's ``data/demo_N`` schema: per-demo ``obs/<key>`` (T, dim),
    ``actions`` (T, action_dim), ``done``/``success`` (T,) bool, ``states`` (T, D).
    Also writes ``mask/<aug_mask_key>`` = boolean over ALL ``data/`` demos that
    selects ``core_filter_key`` demos + the appended rollout demos, so the
    retrain step can pick up both via a single mask (otherwise it would
    re-train on core-only and silently ignore the new rollouts).

    .. warning:: UNTESTED against the real robomimic loader (env deferred). The
       schema is faithful to SOE's `run_full_multi_round.py` write path; if a
       future real run finds a missing attribute (e.g. ``model_file``, ``env``
       metadata in attrs), extend here.
    """
    import h5py
    import shutil

    # start from a copy of core (preserves env metadata / attrs / mask groups)
    shutil.copyfile(core_path, out_path)

    with h5py.File(out_path, "r+") as f:
        # filter core demos
        from scout.data.robomimic_lowdim import _demo_list, _discover_obs_keys
        core_demos = _demo_list(f, core_filter_key)
        if not core_demos:
            raise RuntimeError(f"no core demos under mask='{core_filter_key}' in {core_path}")
        obs_keys = _discover_obs_keys(f["data"], core_demos[0])

        # all demos present BEFORE we append (used for the augmented mask)
        all_demos_before = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        core_set = set(core_demos)

        # find next free demo id
        existing_ids = [int(d.split("_")[-1]) for d in f["data"].keys()
                        if d.startswith("demo_") and d.split("_")[-1].isdigit()]
        next_id = (max(existing_ids) + 1) if existing_ids else 0

        new_demo_names: List[str] = []
        for rollout in rollouts:
            demo_name = f"demo_{next_id}"
            next_id += 1
            ep_len = int(rollout.get("horizon", 0))
            if ep_len == 0:
                continue
            grp = f["data"].create_group(demo_name)
            # obs: per-key (T, dim) arrays reconstructed from rollout["obs"]
            obs_list = rollout.get("obs") or []
            if len(obs_list) < ep_len:
                # rollout wasn't recorded with obs -- can't recover per-key
                raise ValueError(
                    f"rollout for {demo_name} missing obs (need record_obs=True); "
                    f"got {len(obs_list)} frames, need {ep_len}")
            obs_grp = grp.create_group("obs")
            for k in obs_keys:
                obs_grp.create_dataset(
                    k, data=np.stack(
                        [np.asarray(o[k], dtype=np.float32).reshape(-1)
                         for o in obs_list[:ep_len]], axis=0))
            grp.create_dataset("actions",
                               data=np.asarray(rollout["actions"], dtype=np.float32))
            grp.create_dataset("done",
                               data=np.asarray(rollout["dones"], dtype=bool))
            grp.create_dataset("success",
                               data=np.full(ep_len, bool(rollout.get("success", True)),
                                            dtype=bool))
            # num_samples attr (robomimic convention)
            grp.attrs["num_samples"] = ep_len
            new_demo_names.append(demo_name)

        # write the augmented mask: True for core_<filter> demos + new rollouts.
        # `data/demo_list` order is the canonical mask index order (sorted).
        all_demos_after = sorted([k for k in f["data"].keys() if k.startswith("demo")])
        new_set = set(new_demo_names)
        mask = np.array([d in core_set or d in new_set for d in all_demos_after],
                        dtype=bool)
        if f"mask/{aug_mask_key}" in f:
            del f[f"mask/{aug_mask_key}"]
        aug_grp = f.create_group(f"mask/{aug_mask_key}")
        aug_grp.create_dataset("mask", data=mask)
        aug_grp.attrs["num"] = int(mask.sum())
        f.attrs["num_demos_added"] = len(new_demo_names)


# --------------------------------------------------------------------------- #
# config loader
# --------------------------------------------------------------------------- #
def load_cfg(path: str) -> EasyDict:
    with open(path, "r") as f:
        return EasyDict(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# dry-run with mocks (orchestration verification, ONE round)
# --------------------------------------------------------------------------- #
def _dry_run():
    """Mock-DP / mock-VIB / mock-env ONE-round orchestration check.

    Run via ``python -m scout.eval.self_improvement``. Verifies the loop wires
    end-to-end: round iterates -> baseline rollout -> guided rollout -> success
    filter -> source.add fires -> retrain_fn fires -> metric compare logged.

    NOTE: this is an ORCHESTRATION test (do the right components fire in the
    right order with the right arguments?), not a correctness test (the mock
    DP's actions are uncorrelated with success -- don't read into the absolute
    success_rate value).
    """
    from scout.policy.dp import DP
    from scout.model.scout_vib import ScoutVIB
    from scout.data.transition_source import ReplayBuffer

    state_dim = 8
    action_dim = 4
    num_action = 20

    cfg = EasyDict({
        "base_dp": {"initial_ckpt_path": "<mock-dp-0>"},
        "dataset": {"path": "<mock-core>"},
        "action_dim": action_dim,
        "eval": {"n_init_states": 4, "try_times": 3, "horizon": 10,
                 "obs_keys": ["low_dim_a", "low_dim_b"]},
        "exploration": {"guidance_scale": 5.0, "guidance_start_timestep": 50},
        "self_improvement": {"num_rounds": 1, "core_filter_key": "train"},
    })

    # factories that build FRESH modules per call (DP needs to be re-created
    # per round so the retrain "takes effect"; ScoutVIB is created once by the
    # loop itself).
    def dp_factory(ckpt_path: str) -> torch.nn.Module:
        dp = DP(num_action=num_action, action_dim=action_dim,
                obs_shape_meta={"low_dim_a": dict(shape=[action_dim], type="low_dim"),
                                "low_dim_b": dict(shape=[action_dim], type="low_dim")})
        dp.to(torch.device("cpu"))
        return dp

    def scout_vib_factory() -> torch.nn.Module:
        return ScoutVIB(state_dim=state_dim, action_dim=action_dim,
                        s_latent_dim=16, style_dim=8, hidden_dim=32,
                        beta=1e-3).to(torch.device("cpu"))

    # Mock env: scripted success threshold (same as rollout smoke).
    class MockEnv:
        def __init__(self, seed=0):
            self.action_dim = action_dim
            self.horizon = cfg.eval.horizon
            self._step = 0
            self._cum = 0.0
            self._state_dict = {"s": 0.0}
            self._rng = np.random.default_rng(seed)
            self.rollout_exceptions = ()

        def reset(self):
            self._step = 0
            self._cum = 0.0
            # Mid-range threshold: untrained-DP chunk magnitudes vary enough
            # across diffusion noise + guided-z that ~1-2 of 4 init states
            # solve in baseline; some of the remaining solve in exploration
            # (try_times=3) via different z draws. Exercises success-filter
            # AND write-back.
            self._state_dict = {"s": float(self._rng.uniform(1.5, 3.5))}
            return self._get_obs()

        def reset_to(self, state_dict):
            self._step = 0
            self._cum = 0.0
            self._state_dict = dict(state_dict)
            return self._get_obs()

        def _get_obs(self):
            return {"low_dim_a": np.zeros(self.action_dim, dtype=np.float32),
                    "low_dim_b": np.ones(self.action_dim, dtype=np.float32)}

        def step(self, action):
            self._step += 1
            # Magnify so untrained chunks (~|a[0]|~0.1) cumulate to >threshold:
            self._cum += float(np.abs(action).sum()) * 0.5
            r = float(np.linalg.norm(action))
            done = self.is_success()["task"] or (self._step >= self.horizon)
            return self._get_obs(), r, done, {}

        def is_success(self):
            return {"task": bool(self._cum >= self._state_dict["s"])}

        def get_state(self):
            return dict(self._state_dict)

        def close(self):
            pass

    def env_factory():
        return MockEnv()

    def state_to_vec(obs_dict):
        a = obs_dict["low_dim_a"]
        b = obs_dict["low_dim_b"]
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a); b = torch.as_tensor(b)
        v = torch.cat([a, b], dim=-1).float()
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v

    # mock retrain_fn: record calls + return a new fake path (no real training)
    retrain_calls: List[Tuple] = []

    def mock_retrain_fn(cfg, round_idx, successful_rollouts, prev_dp_ckpt):
        retrain_calls.append((round_idx, len(successful_rollouts), prev_dp_ckpt))
        return f"<mock-dp-{round_idx + 1}>"

    # mock source: a tiny ReplayBuffer we can inspect for write-back counts
    written_back = {"count": 0}
    mock_buffer = ReplayBuffer(state_dim=2 * action_dim, action_dim=action_dim,
                               capacity=1000)
    orig_add = mock_buffer.add

    def counting_add(trans):
        orig_add(trans)
        written_back["count"] += len(trans["S_t"])

    mock_buffer.add = counting_add                                  # type: ignore

    def source_factory():
        return mock_buffer

    loop = SelfImprovementLoop(
        cfg=cfg,
        dp_factory=dp_factory,
        scout_vib_factory=scout_vib_factory,
        env_factory=env_factory,
        retrain_fn=mock_retrain_fn,
        source_factory=source_factory,
        state_to_vec=state_to_vec,
        device=torch.device("cpu"),
    )
    history = loop.run()

    print("\n--- dry-run results ---")
    print(f"rounds run         : {len(history)}")
    print(f"history[0]         : {history[0]}")
    print(f"retrain calls      : {retrain_calls}")
    print(f"buffer.add rows    : {written_back['count']}")
    print(f"buffer size        : {len(mock_buffer)}")
    print(f"accumulated rollouts: {len(loop.accumulated_rollouts)}")

    # assertions
    assert len(history) == 1, "expected 1 round"
    h = history[0]
    assert "success_rate" in h and "pass_at_k" in h and "exploration_yield" in h \
        and "jerk_baseline" in h, "missing metric keys"
    assert 0.0 <= h["success_rate"] <= 1.0
    assert 0.0 <= h["pass_at_k"] <= 1.0
    assert h["exploration_yield"] >= 0
    # retrain was called for the not-last round (round 0 with num_rounds=1: NO
    # retrain expected -- last round skips it). This makes num_rounds=1 a pure
    # "does the baseline+exploration pipeline fire" check; for the full retrain
    # wiring verification, see _dry_run_two_rounds below.
    assert retrain_calls == [], "(num_rounds=1 -> no retrain expected)"

    # Airtight write-back assertion: directly exercise _collect_successful with
    # a synthetic successful rollout, so source.add coverage doesn't hinge on
    # mock-env stochasticity (untrained DPs have unpredictable success rates).
    synth_traj = {
        "actions": np.random.standard_normal((6, action_dim)).astype(np.float32),
        "rewards": np.zeros(6, dtype=np.float32),
        "dones": np.zeros(6, dtype=bool),
        "states": [{} for _ in range(6)],
        "obs": [{"low_dim_a": np.zeros(action_dim, dtype=np.float32),
                 "low_dim_b": np.ones(action_dim, dtype=np.float32)}
                for _ in range(6)],
        "next_obs": [{"low_dim_a": np.zeros(action_dim, dtype=np.float32),
                      "low_dim_b": np.ones(action_dim, dtype=np.float32)}
                     for _ in range(6)],
        "horizon": 6, "success": True, "initial_state_dict": None,
    }
    synth_expl = [{"solved": True, "n_tries": 1,
                   "successful_trajs": [synth_traj], "all_trajs": [synth_traj],
                   "baseline_solved": False}]
    before = len(mock_buffer)
    loop._collect_successful(synth_expl, obs_keys=cfg.eval.obs_keys,
                             action_dim=action_dim)
    after = len(mock_buffer)
    print(f"[write-back] synth rollout: buffer {before} -> {after} "
          f"(written_back count={written_back['count']})")
    assert after == before + 6, "source.add did not fire for synthetic successful rollout"
    print("[dry-run] self_improvement.py OK (orchestration + write-back)")


def _dry_run_two_rounds():
    """Two-round variant to exercise the retrain + DP_{i+1} wiring.

    Round 0's retrain fires (mock) -> its returned path becomes round 1's DP
    ckpt path. Verifies retrain_fn invocation count + path hand-off.
    """
    from scout.policy.dp import DP
    from scout.model.scout_vib import ScoutVIB

    state_dim = 8
    action_dim = 4
    num_action = 20
    cfg = EasyDict({
        "base_dp": {"initial_ckpt_path": "<mock-dp-0>"},
        "dataset": {"path": "<mock-core>"},
        "action_dim": action_dim,
        "eval": {"n_init_states": 3, "try_times": 2, "horizon": 8,
                 "obs_keys": ["low_dim_a", "low_dim_b"]},
        "exploration": {"guidance_scale": 1.0, "guidance_start_timestep": 50},
        "self_improvement": {"num_rounds": 2, "core_filter_key": "train"},
    })

    def dp_factory(ckpt_path):
        dp = DP(num_action=num_action, action_dim=action_dim,
                obs_shape_meta={"low_dim_a": dict(shape=[action_dim], type="low_dim"),
                                "low_dim_b": dict(shape=[action_dim], type="low_dim")})
        return dp.to(torch.device("cpu"))

    def vib_factory():
        return ScoutVIB(state_dim=state_dim, action_dim=action_dim,
                        s_latent_dim=16, style_dim=8, hidden_dim=32,
                        beta=1e-3).to(torch.device("cpu"))

    class MockEnv:
        def __init__(self, seed=0):
            self.action_dim = action_dim
            self.horizon = cfg.eval.horizon
            self._step = 0; self._cum = 0.0
            self._state_dict = {"s": 0.5}
            self._rng = np.random.default_rng(seed)
            self.rollout_exceptions = ()

        def reset(self):
            self._step = 0; self._cum = 0.0
            self._state_dict = {"s": float(self._rng.uniform(0.3, 1.5))}
            return self._get_obs()

        def reset_to(self, sd):
            self._step = 0; self._cum = 0.0
            self._state_dict = dict(sd)
            return self._get_obs()

        def _get_obs(self):
            return {"low_dim_a": np.zeros(self.action_dim, dtype=np.float32),
                    "low_dim_b": np.ones(self.action_dim, dtype=np.float32)}

        def step(self, a):
            self._step += 1
            self._cum += float(np.abs(a[0]).sum()) * 0.05
            done = self.is_success()["task"] or (self._step >= self.horizon)
            return self._get_obs(), 0.0, done, {}

        def is_success(self):
            return {"task": bool(self._cum >= self._state_dict["s"])}

        def get_state(self):
            return dict(self._state_dict)

        def close(self):
            pass

    def state_to_vec(obs):
        a = obs["low_dim_a"]; b = obs["low_dim_b"]
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a); b = torch.as_tensor(b)
        v = torch.cat([a, b], dim=-1).float()
        return v.unsqueeze(0) if v.dim() == 1 else v

    retrain_calls = []

    def retrain_fn(c, r_idx, rollouts, prev):
        retrain_calls.append((r_idx, prev))
        return f"<mock-dp-{r_idx + 1}>"

    loop = SelfImprovementLoop(
        cfg=cfg, dp_factory=dp_factory, scout_vib_factory=vib_factory,
        env_factory=lambda: MockEnv(), retrain_fn=retrain_fn,
        source_factory=None, state_to_vec=state_to_vec,
        device=torch.device("cpu"), verbose=False,
    )
    history = loop.run()

    print("\n--- two-round dry-run ---")
    print(f"rounds run: {len(history)}")
    print(f"retrain calls: {retrain_calls}")
    print(f"final dp_path: {loop.dp_path}")
    print(f"history: {[(h['round'], round(h['success_rate'], 3), h['exploration_yield']) for h in history]}")
    assert len(history) == 2, "expected 2 rounds"
    assert len(retrain_calls) == 1, "expected 1 retrain (after round 0)"
    assert retrain_calls[0][1] == "<mock-dp-0>", "round 0 should start from initial"
    assert loop.dp_path == "<mock-dp-1>", "round 1 should use retrain's output"
    print("[dry-run-2] self_improvement.py OK (retrain wiring)")


if __name__ == "__main__":
    _dry_run()
    _dry_run_two_rounds()
