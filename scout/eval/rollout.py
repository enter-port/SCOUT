"""Robomimic rollout harness for SCOUT (Phase 5; scout_design.md §5).

Wires SCOUT's eval onto the LPB base DP stack:

  * **unguided** path: :class:`BaseDPAdapter` wraps the LPB
    :class:`~diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy`
    and drives its ``predict_action`` (chunked replay, SOE ``RolloutDP`` shape).
  * **guided** path: :class:`GuidedAdapter` wraps
    :class:`scout.guidance.policy.ScoutPolicy` (an LPB
    ``DiffusionUnetHybridImagePolicy`` subclass with SCOUT guidance in its
    overridden ``guided_conditional_sample``) and drives its
    ``predict_action_dyn_guided``; the SCOUT planner (cost ``‖z−μ‖``, seam ②
    unnormalize-only bridge, seam ① obs-adapter) is attached once at construction.

Env interface (pluggable; real = :class:`RobomimicScoutEnvAdapter` around the LPB
``RobomimicImageWrapper``, mock = the ``MockEnv`` in the ``__main__`` smoke test).
Whatever the caller passes must expose:

    reset()                -> obs_dict            (fresh episode)
    reset_to(state_dict)   -> obs_dict            (deterministic replay)
    step(action)           -> (next_obs_dict, r, done, info)
    is_success()           -> {"task": bool, ...}
    get_state()            -> state_dict

``action`` is a 1-D numpy array ``(action_dim,)`` in env space (DP ``predict_action``
already unnormalizes; abs_action transform, if needed, is the env's job).
``obs_dict`` is a per-key dict of numpy arrays ``(n_obs_steps, *shape)`` -- exactly
what the LPB ``MultiStepWrapper`` emits.

Per scout_design.md §5: for each of N init states, run 1 baseline try, then up to
``try_times`` exploration tries on the failed ones (first-success stops).
Successful rollouts (``record_obs=True``) feed the augmented-hdf5 write-back
(:mod:`scout.eval.self_improvement`) -- no in-memory buffer.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from scout.normalizer import ActionNormalizerBridge, UnnormalizeOnlyBridge


# --------------------------------------------------------------------------- #
# obs / action bridges (seam ① + ②)
# --------------------------------------------------------------------------- #
def make_obs_adapter(
    view_names: Sequence[str],
    proprio_keys: Sequence[str],
) -> Callable[[dict], dict]:
    """seam ①: LPB raw keyed ``obs_dict`` -> E_s format (scout_design.md §2).

    Returns a callable ``current_obs -> {"visual": {view: (B,1,3,H,W)},
    "proprio": (B,1,P)}`` -- the layout :class:`scout.model.encoder.StateEncoder`
    expects. ``current_obs`` is the dict the LPB ``predict_action_dyn_guided``
    threads into ``guided_conditional_sample`` (i.e. the last obs frame,
    per-key ``(B, 1, *shape)`` tensors). RGB keys are already ``CHW`` under the
    LPB shape_meta; proprio keys are concatenated into a single ``(B,1,P)``.
    """
    view_names = list(view_names)
    proprio_keys = list(proprio_keys)

    def adapt(current_obs: dict) -> dict:
        visual = {v: current_obs[v].float() for v in view_names}
        proprio = torch.cat([current_obs[k].float() for k in proprio_keys],
                            dim=-1)
        return {"visual": visual, "proprio": proprio}

    return adapt


def make_action_bridge(dp) -> ActionNormalizerBridge:
    """seam ②: DP-normalized ``x̂_0`` -> raw (VIB action space).

    The LPB base DP carries a fitted ``dp.normalizer['action']``; inside
    ``guided_conditional_sample`` the trajectory is DP-normalized, so the SCOUT
    cost must unnormalize before evaluating the VIB encoder (trained on raw
    hdf5 actions). Returns :class:`UnnormalizeOnlyBridge` (differentiable affine
    -> ``autograd.grad(cost, x_t)`` flows through). If the DP somehow has no
    fitted normalizer yet, falls back to :class:`IdentityBridge`.
    """
    normalizer = getattr(dp, "normalizer", None)
    if normalizer is None or "action" not in normalizer:
        from scout.normalizer import IdentityBridge
        return IdentityBridge()
    return UnnormalizeOnlyBridge(normalizer["action"])


# --------------------------------------------------------------------------- #
# policy adapters (chunked replay; SOE RolloutDP shape)
# --------------------------------------------------------------------------- #
def _to_device_batched(obs_dict: dict, device) -> dict:
    """Per-key numpy/tensor -> ``(1, *shape)`` float tensors on ``device``.

    Accepts already-tensor or numpy inputs of any rank; numpy is converted and
    a leading batch dim of 1 is added (LPB ``predict_action`` expects
    ``(B, n_obs_steps, *shape)`` per key).
    """
    out = {}
    for k, v in obs_dict.items():
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        v = v.float().to(device)
        if v.dim() == 1:
            v = v.unsqueeze(0)                # (To,) -> (1, To) for low_dim keys
        else:
            v = v.unsqueeze(0)                # (To, ...) -> (1, To, ...)
        out[k] = v
    return out


class BaseDPAdapter:
    """Unguided frozen LPB base DP, chunked action replay.

    Args:
        dp               : the LPB ``DiffusionUnetHybridImagePolicy`` (frozen,
                           state loaded; normalizer fit). Put in ``.eval()`` here.
        device           : torch device.
        n_action_steps   : chunk size the DP emits (``dp.n_action_steps``).
                           Falls back to ``inference_horizon`` if absent.
        inference_horizon: env steps before re-planning. Defaults to
                           ``n_action_steps`` (re-plan once per chunk).
        obs_to_dict      : optional ``env_obs -> DP obs_dict`` converter; default
                           identity (env already returns a per-key dict).
    """

    def __init__(self, dp, device,
                 n_action_steps: Optional[int] = None,
                 inference_horizon: Optional[int] = None,
                 obs_to_dict: Optional[Callable[[Any], dict]] = None):
        self.dp = dp
        self.device = device
        self.n_action_steps = int(n_action_steps or getattr(dp, "n_action_steps", 1))
        self.inference_horizon = int(inference_horizon or self.n_action_steps)
        self._t = 0
        self._chunk: Optional[np.ndarray] = None
        self.obs_to_dict = obs_to_dict or (lambda o: o)
        dp.eval()

    def start_episode(self):
        self._t = 0
        self._chunk = None
        # LPB policies carry a per-episode reset hook (e.g. crop state); call it
        # when present (no-op for the plain DiffusionUnetHybridImagePolicy).
        reset = getattr(self.dp, "reset", None)
        if callable(reset):
            reset()

    @torch.no_grad()
    def __call__(self, obs) -> np.ndarray:
        if self._chunk is None or self._t >= self.inference_horizon:
            obs_dict = _to_device_batched(self.obs_to_dict(obs), self.device)
            result = self.dp.predict_action(obs_dict)
            action_chunk = result["action"]            # (1, n_action_steps, A)
            self._chunk = action_chunk[0].detach().cpu().numpy()
            self._t = 0
        a = self._chunk[self._t]
        self._t += 1
        return a


class GuidedAdapter(BaseDPAdapter):
    """Guided rollout via :class:`scout.guidance.policy.ScoutPolicy`.

    Same chunk/replay shell as :class:`BaseDPAdapter`, but each chunk is produced
    by :meth:`DiffusionUnetHybridImagePolicy.predict_action_dyn_guided`, which
    calls ScoutPolicy's overridden ``guided_conditional_sample`` (SCOUT cost +
    gate (b) dropped). A fresh ``z ~ N(0,I)`` is sampled inside the policy per
    inference call and held fixed across the chunk (scout_design.md §1, §4).

    The SCOUT planner (carrying the frozen ScoutVIB + seam ①/②) is attached to
    the policy ONCE at construction (caller does
    ``policy.initialize_scout_planner(planner, guidance_start_timestep, guidance_scale)``
    before passing the policy in). This class is agnostic to that -- it just
    calls ``predict_action_dyn_guided``.
    """

    def __call__(self, obs) -> np.ndarray:
        if self._chunk is None or self._t >= self.inference_horizon:
            obs_dict = _to_device_batched(self.obs_to_dict(obs), self.device)
            # NOTE: NOT @torch.no_grad at this level -- guidance needs grad on
            # the trajectory inside guided_conditional_sample. The grad is local
            # to the denoise loop (released before we read actions out).
            result = self.dp.predict_action_dyn_guided(obs_dict)
            action_chunk = result["action"]            # (1, n_action_steps, A)
            self._chunk = action_chunk[0].detach().cpu().numpy()
            self._t = 0
        a = self._chunk[self._t]
        self._t += 1
        return a


# --------------------------------------------------------------------------- #
# episode + multi-episode
# --------------------------------------------------------------------------- #
def rollout_episode(policy_adapter, env, horizon: int,
                    initial_state_dict: Optional[dict] = None,
                    record_obs: bool = False
                    ) -> Tuple[bool, dict]:
    """Run a single episode. Returns ``(success, traj)``.

    Mirrors SOE ``rollout_utils.rollout``: ``reset_to(state_dict)``, loop
    ``act -> env.step -> is_success``, break on success, ``success`` from
    ``env.is_success()["task"]``. ``record_obs=True`` stores per-step ``obs`` /
    ``next_obs`` (needed for the augmented-hdf5 write-back; off by default).

    ``env.rollout_exceptions`` (tuple) is swallowed if present (SOE behaviour
    for robomimic numerical instabilities).
    """
    policy_adapter.start_episode()
    obs = env.reset()
    if initial_state_dict is None:
        state_dict = env.get_state()
    else:
        state_dict = initial_state_dict
    obs = env.reset_to(state_dict)

    actions: List[np.ndarray] = []
    rewards: List[float] = []
    dones: List[bool] = []
    states: List[dict] = []
    obs_list: List[dict] = []
    next_obs_list: List[dict] = []
    success = False
    step_i = -1

    excs = getattr(env, "rollout_exceptions", ())
    try:
        for step_i in range(horizon):
            act = policy_adapter(obs)
            if not isinstance(act, np.ndarray):
                act = np.asarray(act)
            next_obs, r, done, _ = env.step(act)
            success = bool(env.is_success().get("task", False))
            done = bool(done) or success

            actions.append(act)
            rewards.append(float(r))
            dones.append(bool(done))
            states.append(copy.deepcopy(state_dict))
            if record_obs:
                obs_list.append(copy.deepcopy(obs))
                next_obs_list.append(copy.deepcopy(next_obs))

            if done or success:
                break
            obs = copy.deepcopy(next_obs)
            state_dict = env.get_state()
    except excs as e:  # empty tuple -> catches nothing (matches SOE behaviour)
        print(f"WARNING: rollout swallowed exception: {e}")

    traj = {
        "actions": np.stack(actions, axis=0) if actions else np.zeros((0,),
                                                                       dtype=np.float32),
        "rewards": np.asarray(rewards, dtype=np.float32),
        "dones": np.asarray(dones, dtype=bool),
        "states": states,
        "obs": obs_list,
        "next_obs": next_obs_list,
        "initial_state_dict": initial_state_dict,
        "horizon": step_i + 1 if step_i >= 0 else 0,
        "success": success,
    }
    return success, traj


def collect_initial_states(env_factory: Callable[[], Any],
                           n_init_states: int) -> List[dict]:
    """Generate N distinct init state_dicts via repeated ``env.reset()``.

    Robomimic ``reset()`` randomises the env; ``get_state()`` captures the
    deterministic-replay handle. ``env_factory`` is called once and the env is
    closed if it has a ``.close()`` method. The N states are fixed across all
    rounds in the self-improvement loop (fair metric comparison).
    """
    env = env_factory()
    try:
        states = []
        for _ in range(n_init_states):
            env.reset()
            states.append(env.get_state())
        return states
    finally:
        if hasattr(env, "close"):
            env.close()


def evaluate_baseline(policy_adapter, env_factory: Callable[[], Any],
                      init_states: Sequence[dict], horizon: int
                      ) -> List[Tuple[bool, dict]]:
    """One try per init state. Returns ``[(success, traj), ...]``."""
    env = env_factory()
    try:
        return [rollout_episode(policy_adapter, env, horizon,
                                initial_state_dict=sd)
                for sd in init_states]
    finally:
        if hasattr(env, "close"):
            env.close()


def evaluate_exploration(exploration_adapter, env_factory: Callable[[], Any],
                         init_states: Sequence[dict], horizon: int,
                         try_times: int = 5,
                         only_failed_of: Optional[Sequence[Tuple[bool, dict]]] = None,
                         ) -> List[dict]:
    """Exploration tries per init state.

    For each init state, run up to ``try_times`` episodes with
    ``exploration_adapter`` (z is resampled *inside* the policy on each chunk,
    so the same adapter instance is reused). Stop on first success (SOE pattern).

    Args:
        exploration_adapter : a :class:`BaseDPAdapter` (typically :class:`GuidedAdapter`).
        env_factory         : produces a fresh env for the whole sweep.
        init_states         : N init state_dicts.
        horizon             : per-episode step cap.
        try_times           : max exploration tries per init state.
        only_failed_of      : optional ``[(success, traj), ...]`` of the baseline
                              run; if given, init states already solved by baseline
                              are skipped (SOE pattern).
    Returns ``[{solved, n_tries, successful_trajs, all_trajs}, ...]``.
    """
    env = env_factory()
    try:
        results: List[dict] = []
        for i, sd in enumerate(init_states):
            if only_failed_of is not None and only_failed_of[i][0]:
                results.append({"solved": True, "n_tries": 0,
                                "successful_trajs": [], "all_trajs": [],
                                "baseline_solved": True})
                continue
            entry = {"solved": False, "n_tries": 0,
                     "successful_trajs": [], "all_trajs": [],
                     "baseline_solved": False}
            for _ in range(try_times):
                success, traj = rollout_episode(exploration_adapter, env, horizon,
                                                initial_state_dict=sd)
                entry["n_tries"] += 1
                entry["all_trajs"].append(traj)
                if success:
                    entry["solved"] = True
                    entry["successful_trajs"].append(traj)
                    break
            results.append(entry)
        return results
    finally:
        if hasattr(env, "close"):
            env.close()


# --------------------------------------------------------------------------- #
# robomimic env factory (LPB robomimic_image_runner reuse; lazy-imported)
# --------------------------------------------------------------------------- #
class RobomimicScoutEnvAdapter:
    """Adapt LPB :class:`RobomimicImageWrapper` to the SCOUT env contract.

    LPB's wrapper exposes ``reset``/``step``/``get_flattened_state``/
    ``get_success_label``; SCOUT's :func:`rollout_episode` also needs
    ``reset_to(state_dict)`` + ``get_state()`` + ``is_success()["task"]``. This
    thin shim wires those (used by :func:`make_robomimic_env_factory`).

    ``model_file`` (robomimic sim) is forwarded to ``env.reset_to`` when present
    (matches the wrapper's internal reset path).
    """

    def __init__(self, wrapper):
        self.wrapper = wrapper
        self.env = wrapper.env
        self.rollout_exceptions = ()  # robomimic numerical instability swallows

    def _model_file(self):
        return getattr(self.wrapper, "model_file", None)

    def reset(self):
        # random reset (init_state=None, no seed) -- the wrapper handles caching.
        self.wrapper.init_state = None
        self.wrapper._seed = None
        return self.wrapper.reset()

    def reset_to(self, state_dict):
        self.wrapper.init_state = None
        self.wrapper._seed = None
        mf = self._model_file()
        if mf is None:
            raw_obs = self.env.reset_to({"states": state_dict})
        else:
            raw_obs = self.env.reset_to({"states": state_dict, "model": mf})
        return self.wrapper.get_observation(raw_obs)

    def step(self, action):
        return self.wrapper.step(action)

    def is_success(self):
        return {"task": bool(self.wrapper.get_success_label())}

    def get_state(self):
        return self.env.get_state()["states"]

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_robomimic_env_factory(dataset_path: str, shape_meta: dict,
                               render_obs_key: str = "agentview_image"
                               ) -> Callable[[], Any]:
    """Build a :class:`RobomimicScoutEnvAdapter` factory (real robomimic run).

    Reuses the LPB ``robomimic_image_runner.create_env`` env-construction path
    (env_meta from dataset + ``EnvUtils.create_env_from_metadata``). Lazy-imported
    so this module imports cleanly without robomimic/mujoco installed.
    """
    def factory() -> Any:
        import collections
        import h5py
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
            RobomimicImageWrapper,
        )

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta["env_kwargs"]["use_object_obs"] = False

        modality_mapping = collections.defaultdict(list)
        for key, attr in shape_meta["obs"].items():
            modality_mapping[attr.get("type", "low_dim")].append(key)
        ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

        robomimic_env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta, render=False, render_offscreen=True,
            use_image_obs=True,
        )
        robomimic_env.env.hard_reset = False
        wrapper = RobomimicImageWrapper(
            env=robomimic_env, shape_meta=shape_meta, init_state=None,
            render_obs_key=render_obs_key,
        )
        return RobomimicScoutEnvAdapter(wrapper)

    return factory


# --------------------------------------------------------------------------- #
# smoke test -- MOCK env + MOCK policies (no robomimic / mujoco / LPB deps)
# --------------------------------------------------------------------------- #
def _smoke():
    """Mock-env + mock-policy smoke test. ``python -m scout.eval.rollout``.

    Three checks:
      1. ``evaluate_baseline`` on a scripted MockEnv + MockDPAdapter: collects
         N init states x 1 try, success/actions/states recorded.
      2. ``GuidedAdapter`` wiring against a MockGuidedPolicy exposing the LPB
         ``predict_action_dyn_guided`` interface: runs without raising and
         produces non-empty actions. (The real ScoutPolicy can't be built
         without robomimic; its import + guided_conditional_sample override are
         verified separately in scout/guidance/_verify.py.)
      3. record_obs=True yields per-frame obs/next_obs of the right shape --
         the contract the augmented-hdf5 write-back consumes.
    """
    import torch.nn as nn

    # ---------- check 1: scripted mock env + adapter ----------------------- #
    class MockEnv:
        """Scripted env: succeeds iff cumulative action[0] sum >= threshold."""

        def __init__(self, action_dim=4, horizon=20, seed=0):
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
            self._state_dict = {"s": float(self._rng.uniform(0.5, 2.5))}
            return self._get_obs()

        def reset_to(self, state_dict):
            self._step = 0
            self._cum = 0.0
            self._state_dict = dict(state_dict)
            return self._get_obs()

        def _get_obs(self):
            return {"agentview_image": np.zeros((2, 3, 4, 4), dtype=np.float32),
                    "robot0_eye_in_hand_image": np.ones((2, 3, 4, 4), dtype=np.float32),
                    "robot0_eef_pos": np.zeros((2, 3), dtype=np.float32),
                    "robot0_eef_quat": np.zeros((2, 4), dtype=np.float32),
                    "robot0_gripper_qpos": np.zeros((2, 2), dtype=np.float32)}

        def step(self, action):
            self._step += 1
            self._cum += float(action[0]) * 0.2
            next_obs = self._get_obs()
            r = float(action[0])
            done = self.is_success()["task"] or (self._step >= self.horizon)
            return next_obs, r, done, {}

        def is_success(self):
            return {"task": bool(self._cum >= self._state_dict["s"])}

        def get_state(self):
            return dict(self._state_dict)

        def close(self):
            pass

    class MockDPAdapter:
        """Unguided stand-in: always emits action=[0.5,0,0,0]. Mirrors the
        BaseDPAdapter chunk-replay shape without needing the LPB stack."""

        def __init__(self, n_action_steps=4):
            self.n_action_steps = n_action_steps
            self._t = 0
            self._chunk = None

        def start_episode(self):
            self._t = 0
            self._chunk = None

        def __call__(self, obs):
            if self._chunk is None or self._t >= self.n_action_steps:
                self._chunk = np.stack(
                    [np.array([0.5, 0, 0, 0], dtype=np.float32)]
                    * self.n_action_steps, axis=0)
                self._t = 0
            a = self._chunk[self._t]
            self._t += 1
            return a

    env_factory = lambda: MockEnv()
    init_states = collect_initial_states(env_factory, n_init_states=5)
    print(f"[1] collected {len(init_states)} init states: thresholds="
          f"{[round(s['s'], 2) for s in init_states]}")
    base = evaluate_baseline(MockDPAdapter(), env_factory, init_states, horizon=20)
    n_succ = sum(1 for s, _ in base if s)
    print(f"[1] baseline: {n_succ}/{len(init_states)} succeeded; "
          f"horizons={[t['horizon'] for _, t in base]}")
    assert all(t["horizon"] > 0 for _, t in base), "empty traj"
    assert all(t["actions"].shape[1] == 4 for _, t in base), "wrong action_dim"
    expl = evaluate_exploration(MockDPAdapter(), env_factory, init_states,
                                horizon=20, try_times=3, only_failed_of=base)
    print(f"[1] exploration: solved={[r['solved'] for r in expl]}, "
          f"yields={[len(r['successful_trajs']) for r in expl]}")

    # ---------- check 2: GuidedAdapter against a mock LPB-style policy ------ #
    # The real ScoutPolicy can't be instantiated without robomimic; verify the
    # adapter <-> policy contract (predict_action_dyn_guided + chunk replay)
    # with a mock that mimics that interface.
    class MockScoutPolicy(nn.Module):
        """Mimics LPB predict_action_dyn_guided: returns a fixed chunk in
        env-normalized action space. Exposes n_action_steps (LPB policy attr)."""

        def __init__(self, n_action_steps=4, action_dim=4):
            super().__init__()
            self.n_action_steps = n_action_steps
            self.action_dim = action_dim

        def reset(self):
            pass

        def predict_action_dyn_guided(self, obs_dict):
            B = next(iter(obs_dict.values())).shape[0]
            chunk = torch.zeros((B, self.n_action_steps, self.action_dim),
                                dtype=torch.float32)
            chunk[..., 0] = 0.5
            return {"action": chunk, "action_pred": chunk}

        def eval(self):
            return self

    device = torch.device("cpu")
    guided_policy = MockScoutPolicy(n_action_steps=4, action_dim=4)
    guided = GuidedAdapter(guided_policy, device, n_action_steps=4)
    env = MockEnv(action_dim=4, horizon=10)
    succ, traj = rollout_episode(guided, env, horizon=10,
                                 initial_state_dict={"s": 100.0},
                                 record_obs=True)
    a = traj["actions"]
    print(f"[2] guided: success={succ} horizon={traj['horizon']} "
          f"actions.shape={a.shape} |a[0]|={float(np.linalg.norm(a[0])):.3f}")
    assert a.shape == (traj["horizon"], 4), "action shape"
    assert traj["horizon"] > 0, "guided rollout produced no steps"

    # ---------- check 3: record_obs contract for write-back --------------- #
    assert len(traj["obs"]) == traj["horizon"], "obs frames mismatch horizon"
    assert len(traj["next_obs"]) == traj["horizon"], "next_obs frames mismatch"
    print(f"[3] record_obs: {len(traj['obs'])} frames, "
          f"keys={sorted(traj['obs'][0].keys())}")
    print("[smoke] rollout.py OK")


if __name__ == "__main__":
    _smoke()
