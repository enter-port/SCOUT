"""Robomimic rollout harness for SCOUT (Phase 5, scout_impl_plan Task 5.1).

Transplants the env-interaction shape from SOE ``simulation/rollout_utils.py``
but strips robomimic-specific hard deps so a MOCK env / MOCK policy drives the
same code path. Real robomimic run is deferred (needs mujoco/robomimic install
+ trained ckpts + dataset -- none live in the dev env).

Env interface (pluggable; real = robomimic ``EnvBase``, mock = ``MockEnv`` in
the ``__main__`` smoke test). Whatever the caller passes must expose:

    reset()                -> obs_dict            (fresh episode)
    reset_to(state_dict)   -> obs_dict            (deterministic replay)
    step(action)           -> (next_obs_dict, r, done, info)
    is_success()           -> {"task": bool, ...}
    get_state()            -> state_dict
    rollout_exceptions      : tuple[Exception, ...]   (optional; default ())

``action`` is a 1-D numpy array (``action_dim``,``) in env space. ``obs_dict``
is whatever the policy adapter's ``obs_to_dict`` consumes -- for stage-1 low_dim
that's a per-key dict of 1-D numpy arrays.

Policy adapters (replace SOE's monolithic ``RolloutDP``):

    BaseDPAdapter  : wraps :class:`scout.policy.dp.DP` (frozen). Chunked:
                     every ``inference_horizon`` env steps, re-runs
                     ``DP(obs)`` to get a fresh ``(num_action, action_dim)``
                     chunk and replays one action per step (SOE ``RolloutDP``
                     pattern). No normalizer (stage-1 -- scout/normalizer.py).
    GuidedAdapter  : same chunk/replay shell, but each chunk is produced by
                     :meth:`DiffusionUNetPolicy.guided_conditional_sample` with
                     a fresh ``z ~ N(0,I)`` (held fixed within the chunk per
                     scout_design.md §4) and a per-adapter ``guidance_scale``.

Per scout_design.md §5 / evaluation_plan.md §一.3: for each of N init states,
run 1 baseline try, then up to ``try_times`` exploration tries on the failed
ones (first-success stops). See :func:`evaluate_baseline` /
:func:`evaluate_exploration`.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from scout.normalizer import ActionNormalizerBridge, IdentityBridge


# --------------------------------------------------------------------------- #
# policy adapters
# --------------------------------------------------------------------------- #
def _to_device_batched(obs_dict: Dict[str, Any], device) -> Dict[str, torch.Tensor]:
    """Per-key numpy/tensor -> ``(1, dim)`` float tensors on ``device``.

    Accepts already-tensor or numpy inputs of shape ``(dim,)`` or ``(B, dim)``;
    numpy is converted and 1-D inputs are unsqueezed to batch size 1.
    """
    out = {}
    for k, v in obs_dict.items():
        if not isinstance(v, torch.Tensor):
            v = torch.as_tensor(v)
        v = v.float().to(device)
        if v.dim() == 1:
            v = v.unsqueeze(0)
        out[k] = v
    return out


class BaseDPAdapter:
    """Unguided frozen base DP, chunked action replay (SOE ``RolloutDP`` shape).

    Args:
        dp               : :class:`scout.policy.dp.DP` (already on ``device``,
                           state loaded). Set to ``.eval()`` here.
        device           : torch device.
        inference_horizon: env steps before re-planning. Defaults to the DP's
                           ``num_action`` (re-plan once per chunk).
        obs_to_dict      : optional ``env_obs -> DP obs_dict`` converter; default
                           identity (env already returns a per-key dict). Useful
                           when the env emits e.g. a single flat state vector.
    """

    def __init__(self, dp, device, inference_horizon: Optional[int] = None,
                 obs_to_dict: Optional[Callable[[Any], Dict[str, Any]]] = None):
        self.dp = dp
        self.device = device
        self.num_action = int(dp.action_decoder.horizon)
        self.inference_horizon = int(inference_horizon or self.num_action)
        self._t = 0
        self._chunk: Optional[np.ndarray] = None
        self.obs_to_dict = obs_to_dict or (lambda o: o)
        dp.eval()

    def start_episode(self):
        self._t = 0
        self._chunk = None

    @torch.no_grad()
    def __call__(self, obs) -> np.ndarray:
        if self._chunk is None or self._t >= self.inference_horizon:
            obs_dict = _to_device_batched(self.obs_to_dict(obs), self.device)
            action_chunk = self.dp(obs_dict)        # (1, T, A) in env space
            self._chunk = action_chunk[0].cpu().numpy()
            self._t = 0
        a = self._chunk[self._t]
        self._t += 1
        return a


class GuidedAdapter(BaseDPAdapter):
    """Guided rollout: per-chunk ``z ~ N(0,I)`` + SCOST guidance via
    :meth:`DiffusionUNetPolicy.guided_conditional_sample` (Phase-4 path).

    Extra args:
        scout_vib           : :class:`scout.model.scout_vib.ScoutVIB` (already on
                              ``device``, state loaded). ``E_s`` provides
                              ``s_bar_t = E_s(S_t)``; ``vib_enc`` provides ``μ``.
        state_to_vec        : ``obs_dict -> (1, state_dim)`` torch tensor on any
                              device; the adapter moves it to ``device``. Required
                              because the cost needs a single flat state vector
                              (the encoder's input), which is NOT what the DP
                              consumes (per-key dict). For stage-1 low_dim this is
                              typically ``concat(sorted low_dim keys))``.
        guidance_scale      : η in ``η·√(1−ᾱ_t)`` (Phase-4 E2 picks this).
        guidance_start_timestep : gate (a); only guide when ``t < this``.
        bridge              : :class:`ActionNormalizerBridge` (Identity in
                              stage-1; see scout/normalizer.py).
        z_seed              : optional int; if given, z draws are reproducible.
    """

    def __init__(self, dp, device, scout_vib, state_to_vec: Callable,
                 guidance_scale: float = 1.0, guidance_start_timestep: int = 50,
                 bridge: Optional[ActionNormalizerBridge] = None,
                 inference_horizon: Optional[int] = None,
                 obs_to_dict: Optional[Callable[[Any], Dict[str, Any]]] = None,
                 z_seed: Optional[int] = None):
        super().__init__(dp, device, inference_horizon=inference_horizon,
                         obs_to_dict=obs_to_dict)
        self.scout_vib = scout_vib
        self.style_dim = int(scout_vib.style_dim)
        self.guidance_scale = float(guidance_scale)
        self.guidance_start_timestep = int(guidance_start_timestep)
        self.bridge = bridge or IdentityBridge()
        self.state_to_vec = state_to_vec
        self.scout_vib.eval()
        self._gen: Optional[torch.Generator] = None
        if z_seed is not None:
            self._gen = torch.Generator(device=device).manual_seed(int(z_seed))

    # NOTE: NOT @torch.no_grad -- guided_conditional_sample needs grad on the
    # trajectory (``autograd.grad(cost, trajectory)``). Phase-4 path.
    def _plan_chunk(self, obs) -> np.ndarray:
        obs_dict = _to_device_batched(self.obs_to_dict(obs), self.device)
        dp = self.dp
        ad = dp.action_decoder
        # 1. obs -> global_cond (mirror DP.predict_action, no grad on the encoder)
        with torch.no_grad():
            if dp.img_encoder is not None:
                readout = dp.img_encoder(obs_dict)
                if dp.bottleneck is not None:
                    readout = dp.bottleneck(readout)
            else:
                readout = None
            B = (readout.shape[0] // ad.n_obs_steps) if readout is not None else 1
            global_cond = readout.reshape(B, -1) if readout is not None else None
            cond_data = torch.zeros((B, ad.horizon, ad.action_dim),
                                    device=self.device)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
            # 2. s_bar_t = E_s(S_t)  (fixed across chunk)
            s_vec = self.state_to_vec(obs_dict).float().to(self.device)
            if s_vec.dim() == 1:
                s_vec = s_vec.unsqueeze(0)
            s_bar_t = self.scout_vib.E_s(s_vec)                   # (B, s_bar_dim)
        # 3. z ~ N(0,I), fixed across chunk
        z = torch.randn(B, self.style_dim, device=self.device, generator=self._gen)
        # 4. guided denoise (grad on inside; outer scope keeps default grad state)
        sample = ad.guided_conditional_sample(
            cond_data, cond_mask, global_cond=global_cond,
            classifier_guidance=self.guidance_scale > 0.0,
            s_bar_t=s_bar_t, z=z, vib_enc=self.scout_vib.vib_enc,
            bridge=self.bridge, guidance_scale=self.guidance_scale,
            guidance_start_timestep=self.guidance_start_timestep,
        )
        return sample[..., : ad.action_dim][0].detach().cpu().numpy()

    def __call__(self, obs) -> np.ndarray:
        if self._chunk is None or self._t >= self.inference_horizon:
            self._chunk = self._plan_chunk(obs)
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

    Mirrors SOE ``rollout_utils.rollout`` step-by-step: ``reset_to(state_dict)``,
    loop ``act -> env.step -> is_success``, break on success, ``success`` from
    ``env.is_success()["task"]``. ``record_obs=True`` stores per-step ``obs`` /
    ``next_obs`` (needed for self-improvement write-back; off by default).

    ``env.rollout_exceptions`` (tuple) is swallowed if present -- SOE behaviour
    for robomimic numerical instabilities.
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
    closed if it has a ``.close()`` method.
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
    ``exploration_adapter`` (z is resampled *inside* the adapter on each chunk,
    so the same adapter instance is reused). Stop on first success (SOE pattern).

    Args:
        exploration_adapter : a :class:`BaseDPAdapter` (typically :class:`GuidedAdapter`).
        env_factory         : produces a fresh env for the whole sweep.
        init_states         : N init state_dicts.
        horizon             : per-episode step cap.
        try_times           : max exploration tries per init state.
        only_failed_of      : optional ``[(success, traj), ...]`` of the baseline
                              run; if given, init states already solved by baseline
                              are skipped (their result entry is a trivial solved=
                              True with 0 tries) -- SOE pattern.
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
# write-back helpers
# --------------------------------------------------------------------------- #
def rollout_to_transitions(traj: dict, obs_keys: Sequence[str],
                           action_dim: Optional[int] = None) -> Optional[dict]:
    """Convert one successful rollout dict to ``{S_t, A_t, S_tp1}`` arrays.

    The Phase-1 :class:`scout.data.transition_source.ReplayBuffer` /
    :class:`scout.data.robomimic_lowdim.RobomimicLowdimSource` write-back path
    consumes these. ``S_t`` = concat of sorted ``obs_keys`` per frame (matching
    RobomimicLowdimSource's convention).

    Returns ``None`` if the traj has no recorded obs or is too short.

    Args:
        traj      : a successful episode dict from :func:`rollout_episode`
                    (must have been recorded with ``record_obs=True``).
        obs_keys  : sorted low-dim keys to concat into ``S`` (must match the
                    encoder's training-time state vector layout).
        action_dim: optional sanity-check dim.
    """
    actions = np.asarray(traj["actions"], dtype=np.float32)
    if actions.ndim == 1:
        # edge case: 0-d or single-step rollout
        if actions.size == 0:
            return None
        actions = actions.reshape(-1, 1) if action_dim is None else \
            actions.reshape(-1, action_dim)
    if action_dim is not None and actions.shape[1] != action_dim:
        raise ValueError(f"action_dim mismatch: {actions.shape[1]} vs {action_dim}")
    n_steps = actions.shape[0]
    obs_list = traj.get("obs")
    next_obs_list = traj.get("next_obs")
    if not obs_list or not next_obs_list or len(obs_list) < n_steps:
        return None
    keys = sorted(obs_keys)

    def stack(frames):
        return np.stack(
            [np.concatenate([np.asarray(f[k], dtype=np.float32).reshape(-1)
                             for k in keys], axis=0) for f in frames],
            axis=0)

    S_t = stack(obs_list[:n_steps])
    S_tp1 = stack(next_obs_list[:n_steps])
    return {"S_t": S_t, "A_t": actions, "S_tp1": S_tp1}


# --------------------------------------------------------------------------- #
# smoke test -- MOCK env (no robomimic / mujoco deps)
# --------------------------------------------------------------------------- #
def _smoke():
    """Mock-env smoke test for the rollout harness. Run via ``python -m scout.eval.rollout``.

    Two checks:
      1. ``evaluate_baseline`` on a scripted MockEnv + MockDPAdapter: collects
         N init states x 1 try, success/actions/states recorded.
      2. ``GuidedAdapter`` wiring: a tiny real DP + ScoutVIB + MockEnv, with
         ``guidance_scale>0``, runs without raising and produces non-empty
         actions -- proving the guided path is plumbed end-to-end.
    """
    import torch.nn as nn
    from scout.policy.dp import DP
    from scout.model.scout_vib import ScoutVIB

    # ---------- check 1: scripted mock env + adapter ----------------------- #
    class MockEnv:
        """Scripted env: succeeds iff the cumulative action sum in dim 0 >= 1.0.

        Per-step reward = action[0]. ``reset(i)`` seeds the success threshold so
        ~half of the init states are trivially solvable.
        """

        def __init__(self, action_dim=4, horizon=20, seed=0):
            self.action_dim = action_dim
            self.horizon = horizon
            self._step = 0
            self._cum = 0.0
            self._state_dict = {"s": 0.0}
            self._rng = np.random.default_rng(seed)
            self.rollout_exceptions = ()  # none

        def reset(self):
            self._step = 0
            self._cum = 0.0
            # randomised init state -- threshold ∈ [0.5, 2.5]
            self._state_dict = {"s": float(self._rng.uniform(0.5, 2.5))}
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
            self._cum += float(action[0]) * 0.2  # per-step contribution
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

    class MockAdapter:
        """Always emits action = [0.5, 0, 0, 0] -> succeeds iff threshold <= 0.5*0.2*horizon."""

        def __init__(self):
            self.calls = 0

        def start_episode(self):
            pass

        def __call__(self, obs):
            self.calls += 1
            a = np.zeros(4, dtype=np.float32)
            a[0] = 0.5
            return a

    env_factory = lambda: MockEnv()
    init_states = collect_initial_states(env_factory, n_init_states=5)
    print(f"[1] collected {len(init_states)} init states: thresholds="
          f"{[round(s['s'], 2) for s in init_states]}")
    base = evaluate_baseline(MockAdapter(), env_factory, init_states, horizon=20)
    n_succ = sum(1 for s, _ in base if s)
    print(f"[1] baseline: {n_succ}/{len(init_states)} succeeded; "
          f"horizons={[t['horizon'] for _, t in base]}")
    assert all(t["horizon"] > 0 for _, t in base), "empty traj"
    assert all(t["actions"].shape[1] == 4 for _, t in base), "wrong action_dim"
    expl = evaluate_exploration(MockAdapter(), env_factory, init_states,
                                horizon=20, try_times=3, only_failed_of=base)
    print(f"[1] exploration: solved={[r['solved'] for r in expl]}, "
          f"yields={[len(r['successful_trajs']) for r in expl]}")

    # ---------- check 2: GuidedAdapter wiring ------------------------------ #
    # NOTE: num_action=20 matches the real SOE / scout base_dp config; small
    # horizons (e.g. 5) hit ConditionalUnet1D conv-padding edge cases (output
    # length != input), which never arise at the real config.
    state_dim = 8
    action_dim = 4
    num_action = 20
    device = torch.device("cpu")
    dp = DP(
        num_action=num_action, action_dim=action_dim,
        obs_shape_meta={"low_dim_a": dict(shape=[action_dim], type="low_dim"),
                        "low_dim_b": dict(shape=[action_dim], type="low_dim")},
    ).to(device)
    svib = ScoutVIB(state_dim=state_dim, action_dim=action_dim,
                    style_dim=8, hidden_dim=32, beta=1e-3).to(device)

    def state_to_vec(obs_dict):
        # concat sorted low-dim keys (a, b) -> (1, 2*action_dim) == (1, state_dim) if state_dim==8
        a = obs_dict["low_dim_a"]
        b = obs_dict["low_dim_b"]
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a); b = torch.as_tensor(b)
        v = torch.cat([a, b], dim=-1).float()
        if v.dim() == 1:
            v = v.unsqueeze(0)
        return v

    # assert dims align: state_dim must equal concatenated obs dim
    assert state_dim == 2 * action_dim, "state_dim/obs_dim mismatch in smoke test"

    guided0 = GuidedAdapter(dp, device, svib, state_to_vec,
                            guidance_scale=0.0, guidance_start_timestep=50)
    guided_g = GuidedAdapter(dp, device, svib, state_to_vec,
                             guidance_scale=5.0, guidance_start_timestep=50)
    for label, adapter in [("scale=0", guided0), ("scale=5", guided_g)]:
        env = MockEnv(action_dim=action_dim, horizon=10)
        succ, traj = rollout_episode(adapter, env, horizon=10,
                                     initial_state_dict={"s": 100.0},
                                     record_obs=True)
        a = traj["actions"]
        print(f"[2] guided {label}: success={succ} horizon={traj['horizon']} "
              f"actions.shape={a.shape} |a[0]|={float(np.linalg.norm(a[0])):.3f}")
        assert a.shape == (traj["horizon"], action_dim), "action shape"
        assert traj["horizon"] > 0, "guided rollout produced no steps"

    # write-back smoke -- run a full-horizon trajectory (no early success) so
    # the conversion produces a non-trivial transition count.
    succ, traj = rollout_episode(guided_g, MockEnv(action_dim=action_dim,
                                                    horizon=10), horizon=10,
                                 initial_state_dict={"s": 100.0}, record_obs=True)
    trans = rollout_to_transitions(traj, obs_keys=["low_dim_a", "low_dim_b"],
                                   action_dim=action_dim)
    print(f"[3] rollout_to_transitions: keys={list(trans.keys())} "
          f"S_t.shape={trans['S_t'].shape} A_t.shape={trans['A_t'].shape}")
    assert trans["S_t"].shape == (traj["horizon"], 2 * action_dim)
    assert trans["A_t"].shape == (traj["horizon"], action_dim)
    assert traj["horizon"] == 10, "write-back smoke should use a full-horizon traj"
    print("[smoke] rollout.py OK")


if __name__ == "__main__":
    _smoke()
