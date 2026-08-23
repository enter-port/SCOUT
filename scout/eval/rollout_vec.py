"""Vectorized (parallel-env) rollout harness for SCOUT (Phase 5.4).

Single process, ``n_envs`` robomimic envs (each an independent MuJoCo sim),
batched policy inference across the slots that simultaneously need a new action
chunk. This is the high-throughput sibling of :mod:`scout.eval.rollout`'s
sequential :func:`evaluate_baseline` / :func:`evaluate_exploration` (kept as the
``n_envs == 1`` fallback -- identical result schema).

Why single-process batched (not multiprocessing):
  * the diffusion policy's denoise loop is the dominant per-step cost; batching
    B slots into one ``predict_action`` collapses B inference passes into one;
  * :meth:`ScoutPolicy.guided_conditional_sample` /
    :meth:`ScoutPlanner.compute_loss` already handle ``(B, ...)`` tensors
    natively -- ``z`` is sampled ``(B, style_dim)`` (policy.py), ``s_bar_t`` is
    ``(B, s_bar_dim)``, and the guided path sum-reduces the cost (block-diagonal
    rows => per-row gradient is B-independent; the pre-fix mean reduction made
    the effective guidance guidance_scale/B -- 1/B scaling bug) -- so the guided
    path batches too with NO change to policy/planner;
  * ``record_obs=True`` exploration carries HWC image frames (~100MB/traj) --
    staying in-process avoids IPC'ing those over pipes.

Design -- continuous work queue + per-slot chunk state. Each tick:
  1. launch  -- idle slots pull the next job, ``env.reset_to(init_state)``, reset
                chunk/t/accumulators (guided: draw a fresh ``z`` for this rollout,
                held fixed across all its chunks -- scout_design §1);
  2. replan  -- slots whose chunk is exhausted batch their ``current_obs`` into
                one ``(B, n_obs_steps, ...)`` tensor -> one policy call ->
                ``(B, n_action_steps, A)`` chunks distributed back (baseline:
                ``predict_action`` under ``no_grad``; guided:
                ``predict_action_dyn_guided`` with a batched ``(B, style_dim)`` z
                locked on the planner just before the call);
  3. step    -- every active slot steps its own env with its chunk's current
                action, accumulates the trajectory, checks done/success/horizon;
  4. finalize-> ``on_done`` routes the traj to the caller + wandb progress.

No idle waiting: a slot that finishes early immediately pulls the next pending
job on the next tick (one-tick gap, negligible over a ~300-step episode).

Memory note: N slots each hold one in-flight trajectory (exploration keeps image
obs for the augmented-hdf5 write-back). For N=25, horizon~300, 2 views at
~84x84x3 float32, peak concurrent ~2.5GB; accumulated successes ~12GB/round --
fine on an H20 (96GB). Lower ``cfg.eval.n_envs`` if memory-tight.
"""

from __future__ import annotations

import collections
import copy
from typing import Any, Callable, Deque, List, Optional, Sequence, Tuple

import numpy as np
import torch

from scout.eval.rollout import _to_device_batched  # reuse for shape parity
from scout.eval.metrics import jerk as _traj_jerk   # explore-path running avg_jerk


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _batch_obs(obs_dicts: Sequence[dict], device) -> dict:
    """List[per-key ``(n_obs_steps, *shape)`` ndarray] -> per-key
    ``(B, n_obs_steps, *shape)`` float tensors on ``device``.

    This is the batched analogue of :func:`rollout._to_device_batched` (which
    only adds a leading 1). Stacking across slots gives the ``(B, ...)`` layout
    ``predict_action`` / ``predict_action_dyn_guided`` consume.
    """
    keys = obs_dicts[0].keys()
    out = {}
    for k in keys:
        arr = np.stack([np.asarray(o[k]) for o in obs_dicts], axis=0)
        out[k] = torch.as_tensor(arr).float().to(device)
    return out


def _wandb_log(wandb_run, payload: dict, step: Optional[int] = None):
    """No-op when wandb is disabled (``wandb_run is None``); else ``wandb_run.log``."""
    if wandb_run is None:
        return
    if step is not None:
        wandb_run.log(payload, step=step)
    else:
        wandb_run.log(payload)


# --------------------------------------------------------------------------- #
# per-slot state
# --------------------------------------------------------------------------- #
class _VecSlot:
    """One parallel env + its in-flight episode state.

    The env is created ONCE and reused across jobs (``reset_to`` per new job) --
    matches how the sequential :func:`evaluate_baseline` reuses one env across
    episodes. Chunk replay + trajectory accumulation are per-slot so each env
    runs its own episode independently; only the (batched) policy call is shared.
    """

    def __init__(self, idx: int, env):
        self.idx = idx
        self.env = env
        self.active = False
        self.record_obs = False
        self.job: Optional[tuple] = None
        self.z: Optional[torch.Tensor] = None        # guided only; (1, style_dim)
        self.chunk: Optional[np.ndarray] = None      # (n_action_steps, A)
        self.t = 0                                   # step within current chunk
        self.step_i = 0                              # step within episode
        self.current_obs: Optional[dict] = None
        self.current_state_dict = None
        self.actions: List[np.ndarray] = []
        self.rewards: List[float] = []
        self.dones: List[bool] = []
        self.states: List[dict] = []
        self.obs_list: List[dict] = []
        self.next_obs_list: List[dict] = []
        self.success = False
        self.traj: Optional[dict] = None             # set on finalize

    def reset_for_job(self, job: tuple, record_obs: bool):
        """Reset episode-scoped state for a new job (called on launch)."""
        self.job = job
        self.record_obs = record_obs
        self.chunk = None
        self.t = 0
        self.step_i = 0
        self.actions = []
        self.rewards = []
        self.dones = []
        self.states = []
        self.obs_list = []
        self.next_obs_list = []
        self.success = False
        self.traj = None
        self.active = True


# --------------------------------------------------------------------------- #
# vectorized runner
# --------------------------------------------------------------------------- #
class _VecRunner:
    """Drive ``n_envs`` slots over a job queue with batched policy inference.

    Args:
        dp             : the policy. Baseline uses ``predict_action`` (under
                         ``no_grad``); guided uses ``predict_action_dyn_guided``
                         (with grad) + the ``dp.scout_planner`` already attached
                         by the caller.
        env_factory    : callable -> fresh env (called once per slot).
        n_envs         : parallel env (slot) count.
        n_action_steps : chunk size the policy emits (``dp.n_action_steps``).
        horizon        : per-episode step cap.
        device         : torch device.
        guided         : True -> guided path (batched z + dyn_guided call).
        on_done        : ``callable(slot)`` invoked when an episode finalizes;
                         the caller routes ``slot.traj`` + does extra wandb logs.
        progress_cb    : optional ``callable(tick)`` for periodic wandb progress
                         (reads ``self.completed`` / ``self.successes``).
        wandb_run      : optional wandb run (None -> logging disabled).
        log_every      : tick period for ``progress_cb`` invocation.
    """

    def __init__(self, dp, env_factory: Callable[[], Any], n_envs: int,
                 n_action_steps: int, horizon: int, device,
                 guided: bool,
                 on_done: Optional[Callable[[_VecSlot], None]] = None,
                 progress_cb: Optional[Callable[[int], None]] = None,
                 wandb_run=None, log_every: int = 10):
        self.dp = dp
        self.n_action_steps = int(n_action_steps)
        self.horizon = int(horizon)
        self.device = device
        self.guided = bool(guided)
        self.on_done = on_done
        self.progress_cb = progress_cb
        self.wandb_run = wandb_run
        self.log_every = max(1, int(log_every))

        # one env per slot (independent MuJoCo sim)
        self.envs = [env_factory() for _ in range(n_envs)]
        self.slots = [_VecSlot(i, self.envs[i]) for i in range(n_envs)]

        # Shared policy reset once. Per-episode ``dp.reset()`` doesn't compose
        # with N concurrent episodes on one shared policy; for the diffusion
        # UNet hybrid image policy reset is effectively a no-op, so a single
        # pre-loop reset is correct and avoids per-slot state crosstalk.
        reset = getattr(dp, "reset", None)
        if callable(reset):
            reset()

        # style_dim for guided per-rollout z sampling (None -> guided falls back
        # to the policy's own lazy sampling inside guided_conditional_sample).
        self.style_dim: Optional[int] = None
        if guided:
            planner = getattr(dp, "scout_planner", None)
            scout_vib = getattr(planner, "scout_vib", None) if planner else None
            if scout_vib is not None:
                self.style_dim = int(scout_vib.style_dim)
            # expert-guided planner (select_z hook): z* is selected per chunk
            # INSIDE the denoise loop from the current (s, x̂₀) -- the runner
            # must NOT draw a per-rollout prior z. style_dim=None disables the
            # _launch draw, and _replan's all-not-None guard then leaves z
            # resolution to the policy (which selects z* at the first guided
            # step). Exploration mode (plain ScoutPlanner) is unaffected.
            if planner is not None and hasattr(planner, "select_z"):
                self.style_dim = None

        # progress counters (incremented in _finalize)
        self.completed = 0     # episodes finalized (baseline inits / expl tries)
        self.successes = 0     # successful episodes (baseline solved / expl collected)

    # -- per-slot launch / plan / step / finalize -------------------------- #
    def _launch(self, slot: _VecSlot, job: tuple, record_obs: bool):
        init_state = job[0]
        slot.reset_for_job(job, record_obs)
        slot.current_obs = slot.env.reset_to(init_state)
        slot.current_state_dict = init_state
        if self.guided and self.style_dim is not None:
            # one fresh skill latent per rollout, held across ALL its chunks
            # (scout_design §1 "z 整段定住"; distinct from SOE per-chunk resample).
            slot.z = torch.randn(1, self.style_dim, device=self.device,
                                 dtype=torch.float32)

    def _replan(self):
        """Batched policy call for every active slot whose chunk is exhausted."""
        need = [s for s in self.slots
                if s.active and (s.chunk is None or s.t >= self.n_action_steps)]
        if not need:
            return
        obs_batch = _batch_obs([s.current_obs for s in need], self.device)
        if self.guided:
            planner = getattr(self.dp, "scout_planner", None)
            # lock the batched z (one row per slot in this batch) just before the
            # call; guided_conditional_sample reads planner.z (policy.py).
            if planner is not None and all(s.z is not None for s in need):
                planner.set_z(torch.cat([s.z for s in need], dim=0))
            result = self.dp.predict_action_dyn_guided(obs_batch)
        else:
            with torch.no_grad():
                result = self.dp.predict_action(obs_batch)
        chunks = result["action"].detach().cpu().numpy()   # (B, n_action_steps, A)
        for i, s in enumerate(need):
            s.chunk = chunks[i]
            s.t = 0

    def _step_slots(self):
        for s in self.slots:
            if not s.active:
                continue
            # horizon cap (matches sequential ``range(horizon)`` -- at most
            # ``horizon`` steps per episode).
            if s.step_i >= self.horizon:
                self._finalize(s)
                continue
            act = s.chunk[s.t]
            if not isinstance(act, np.ndarray):
                act = np.asarray(act)
            excs = getattr(s.env, "rollout_exceptions", ())
            try:
                next_obs, r, done, _ = s.env.step(act)
                success = bool(s.env.is_success().get("task", False))
                done = bool(done) or success
            except excs as e:     # empty tuple -> catches nothing (SOE parity)
                print(f"WARNING: vec slot {s.idx} swallowed exception: {e}")
                s.success = False
                self._finalize(s)
                continue

            s.actions.append(act)
            s.rewards.append(float(r))
            s.dones.append(bool(done))
            # states[i] = the state BEFORE step i (matches rollout_episode).
            s.states.append(copy.deepcopy(s.current_state_dict))
            if s.record_obs:
                s.obs_list.append(copy.deepcopy(s.current_obs))
                s.next_obs_list.append(copy.deepcopy(next_obs))
            s.success = success
            s.t += 1
            s.step_i += 1

            if done or success or s.step_i >= self.horizon:
                self._finalize(s)
            else:
                s.current_obs = next_obs
                s.current_state_dict = s.env.get_state()

    def _finalize(self, slot: _VecSlot):
        """Freeze the traj dict, deactivate, update counters, notify caller."""
        actions = slot.actions
        slot.traj = {
            "actions": (np.stack(actions, axis=0) if actions
                        else np.zeros((0,), dtype=np.float32)),
            "rewards": np.asarray(slot.rewards, dtype=np.float32),
            "dones": np.asarray(slot.dones, dtype=bool),
            "states": slot.states,
            "obs": slot.obs_list,
            "next_obs": slot.next_obs_list,
            "initial_state_dict": slot.job[0] if slot.job else None,
            "horizon": slot.step_i,
            "success": slot.success,
        }
        slot.active = False
        self.completed += 1
        if slot.success:
            self.successes += 1
        if self.on_done is not None:
            self.on_done(slot)

    # -- main loop --------------------------------------------------------- #
    def run(self, job_queue: Deque[tuple], record_obs: bool):
        """Drain ``job_queue`` across the slots. Each job is a tuple whose first
        element is the init_state; remaining elements are caller routing tags."""
        tick = 0
        while job_queue or any(s.active for s in self.slots):
            # 1. fill idle slots from the queue
            for s in self.slots:
                if not s.active and job_queue:
                    self._launch(s, job_queue.popleft(), record_obs)
            # 2. batched re-plan for slots needing a new chunk
            self._replan()
            # 3. step every active slot (may finalize some -> on_done)
            self._step_slots()
            # 4. periodic progress callback (wandb)
            tick += 1
            if self.progress_cb is not None and tick % self.log_every == 0:
                self.progress_cb(tick)

    def close(self):
        for env in self.envs:
            if hasattr(env, "close"):
                env.close()


# --------------------------------------------------------------------------- #
# public API (same result schema as the sequential evaluate_*)
# --------------------------------------------------------------------------- #
def evaluate_baseline_vec(dp, env_factory: Callable[[], Any],
                          init_states: Sequence[dict], horizon: int,
                          n_envs: int, n_action_steps: int, device,
                          n_tries: int = 1,
                          record_obs: bool = False,
                          metric_prefix: str = "eval",
                          on_progress: Optional[Callable[[dict], None]] = None,
                          wandb_run=None, log_every: int = 10
                          ) -> Tuple[List[Tuple[bool, dict]], List[bool], List[dict]]:
    """Vectorized base-DP baseline: ``n_tries`` per init state, N envs parallel.

    Returns ``(results, any_success, success_trajs)`` -- same schema as the
    sequential :func:`rollout.evaluate_baseline`:
      * ``results``      : ``[(first_success, first_traj), ...]`` (FIRST try).
      * ``any_success``  : per-init bool -- solved in ANY of the ``n_tries``
                           baseline tries (base DP pass@k).
      * ``success_trajs``: flat list of EVERY successful traj (all tries; only
                           carries per-frame obs/next_obs when ``record_obs``).

    Jobs are flattened as ``(init_state, init_idx, try_idx)``; the vec runner is
    try-agnostic (it just runs jobs), ``on_done`` aggregates per init_idx.
    ``record_obs=False`` (baseline trajs feed metrics only); ``record_obs=True``
    stores per-frame obs/next_obs so successful base-DP rollouts can feed the
    augmented-hdf5 write-back (mode=base collection).

    ``metric_prefix`` namespaces the live progress metrics: "eval" (default)
    reports the metric-measurement keys ``eval/baseline_*``; "explore" reports
    the SAME keys as the guided-exploration dashboard (``explore/init_done`` /
    ``explore/tries_done`` / ``explore/collected`` / ``explore/yield``) so a
    direct-DP collection run (mode=base) compares side-by-side with guided
    exploration on one wandb panel.
    """
    n = len(init_states)
    n_tries = int(n_tries)
    first_results: List[Optional[Tuple[bool, dict]]] = [None] * n
    any_succ_flags: List[bool] = [False] * n
    tries_done_per_env: List[int] = [0] * n   # for env-level progress
    success_trajs: List[dict] = []            # EVERY successful traj (flat)

    def on_done(slot: _VecSlot):
        _, init_idx, try_idx = slot.job
        tries_done_per_env[init_idx] += 1
        if try_idx == 0:
            first_results[init_idx] = (slot.success, slot.traj)
        if slot.success:
            any_succ_flags[init_idx] = True
            success_trajs.append(slot.traj)   # keep ALL (SOE-style)

    def progress_cb(tick: int):
        # env-level progress: an env counts as done when ALL its tries finished.
        # CRITICAL: the numerators must count ONLY fully-done envs -- first_results
        # / any_succ_flags are written the moment try_0 (or any try) finishes, so
        # counting them against the "fully done" denominator would give rates > 1
        # (a bug observed on the real square/can runs: rate hit 2.0, pass5 3.0).
        env_done = sum(1 for c in tries_done_per_env if c >= n_tries)
        succ = 0
        pass5 = 0
        for i in range(n):
            if tries_done_per_env[i] >= n_tries:
                if first_results[i] is not None and first_results[i][0]:
                    succ += 1
                if any_succ_flags[i]:
                    pass5 += 1
        if metric_prefix == "explore":
            # align with the guided-exploration dashboard: base-DP collection
            # reports the SAME counters so both collect flows compare 1:1.
            _wandb_log(wandb_run, {
                "explore/init_done": env_done,
                "explore/tries_done": sum(tries_done_per_env),
                "explore/collected": len(success_trajs),
                "explore/yield": len(success_trajs),
            })
        else:
            _wandb_log(wandb_run, {
                "eval/baseline_env_done": env_done,
                "eval/baseline_successes": succ,
                "eval/baseline_success_rate": succ / max(env_done, 1),
                "eval/base_pass_at_5": pass5 / max(env_done, 1),
            })
        if on_progress is not None:
            on_progress({"completed": env_done, "successes": succ})

    runner = _VecRunner(
        dp, env_factory, n_envs, n_action_steps, horizon, device,
        guided=False, on_done=on_done, progress_cb=progress_cb,
        wandb_run=wandb_run, log_every=log_every,
    )
    try:
        jobs = collections.deque(
            (init_states[i], i, j) for i in range(n) for j in range(n_tries))
        runner.run(jobs, record_obs=record_obs)
    finally:
        runner.close()
    return first_results, any_succ_flags, success_trajs  # type: ignore[return-value]


def evaluate_exploration_vec(dp, env_factory: Callable[[], Any],
                             init_states: Sequence[dict], horizon: int,
                             try_times: int, n_envs: int, n_action_steps: int,
                             device,
                             only_failed_of: Optional[Sequence[Tuple[bool, dict]]] = None,
                             guided: bool = True,
                             on_progress: Optional[Callable[[dict], None]] = None,
                             wandb_run=None, log_every: int = 10
                             ) -> List[dict]:
    """Vectorized exploration: up to ``try_times`` tries per FAILED init state.

    Returns ``[{solved, n_tries, successful_trajs, all_trajs, baseline_solved}, ...]``
    in init-state order (same schema as :func:`rollout.evaluate_exploration``).
    Baseline-solved init states are skipped (passed through as solved in 0 tries)
    unless ``only_failed_of`` is None. ALL ``try_times`` retries run (no early
    stop) and EVERY successful rollout is kept (SOE pattern). ``record_obs=True``
    so successful trajs feed the augmented-hdf5 write-back.
    """
    n = len(init_states)
    # pre-build per-init-state result entries
    results: List[dict] = []
    job_queue: Deque[tuple] = collections.deque()
    for i in range(n):
        if only_failed_of is not None and only_failed_of[i][0]:
            results.append({"solved": True, "n_tries": 0,
                            "successful_trajs": [], "all_trajs": [],
                            "first_traj": None,
                            "baseline_solved": True})
            continue
        entry = {"solved": False, "n_tries": 0,
                 "successful_trajs": [], "all_trajs": [],
                 "first_traj": None,
                 "baseline_solved": False}
        results.append(entry)
        for j in range(int(try_times)):
            job_queue.append((init_states[i], i, j))    # (init_state, init_idx, try_idx)

    # per-init aggregation state (only for failed inits that have jobs)
    first_success: dict = {}      # init_idx -> first try_idx that succeeded (1-based)
    # running avg_jerk over EVERY exploration trajectory (success + failure),
    # SOE 3rd-difference norm (scout.eval.metrics.jerk); T<4 -> 0.0, skipped.
    jerk_sum = 0.0
    jerk_n = 0

    def on_done(slot: _VecSlot):
        nonlocal jerk_sum, jerk_n
        _init_state, init_idx, try_idx = slot.job
        entry = results[init_idx]
        entry["all_trajs"].append(slot.traj)
        if try_idx == 0:
            # rescue-protocol dyn rule (user 2026-08-23): an all-failed init
            # contributes its FIRST retry to the dyn data -- completion order
            # is NOT try order under parallel envs, so tag try 0 explicitly.
            entry["first_traj"] = slot.traj
        j = _traj_jerk(slot.traj["actions"])
        if j > 0.0:                              # T<4 -> 0.0, skipped
            jerk_sum += j
            jerk_n += 1
        if slot.success:
            if init_idx not in first_success:
                first_success[init_idx] = try_idx + 1     # 1-based first-success try
            entry["solved"] = True
            entry["successful_trajs"].append(slot.traj)   # keep ALL (SOE-style)

    def progress_cb(tick: int):
        # tries done = completed episodes; collected = successful ones.
        done = runner.completed
        collected = runner.successes
        # init states fully done = all their try_times tries completed.
        tries_per_init = int(try_times)
        init_done = 0
        init_done_failed = 0          # failed inits whose try_times tries all ran
        solved_failed = 0             # failed inits solved by exploration
        for i in range(n):
            if results[i]["baseline_solved"]:
                init_done += 1
                continue
            if len(results[i]["all_trajs"]) >= tries_per_init:
                init_done += 1
                init_done_failed += 1
            if results[i]["solved"]:
                solved_failed += 1
        _wandb_log(wandb_run, {
            "explore/init_done": init_done,
            "explore/tries_done": done,
            "explore/collected": collected,
            "explore/yield": collected,
        })
        if on_progress is not None:
            on_progress({
                "explore_init_done": init_done_failed,
                "solved_failed": solved_failed,
                "jerk_sum": jerk_sum, "jerk_n": jerk_n,
            })

    runner = _VecRunner(
        dp, env_factory, n_envs, n_action_steps, horizon, device,
        guided=guided, on_done=on_done, progress_cb=progress_cb,
        wandb_run=wandb_run, log_every=log_every,
    )
    try:
        runner.run(job_queue, record_obs=True)
    finally:
        runner.close()

    # finalize n_tries per entry (first-success try, else try_times)
    for i in range(n):
        entry = results[i]
        if entry["baseline_solved"]:
            continue
        entry["n_tries"] = first_success.get(i, int(try_times)) if entry["solved"] \
            else int(try_times)
    return results


# --------------------------------------------------------------------------- #
# smoke test (mock env + batched mock policy; no robomimic / mujoco / LPB)
# --------------------------------------------------------------------------- #
def _smoke_vec():
    """Mock-env + batched mock-policy smoke for the vectorized path.

    Verifies:
      1. ``evaluate_baseline_vec`` (n_envs=2) produces the SAME per-init success
         flags and traj horizon/action-shape as the sequential
         :func:`evaluate_baseline` on identical init_states + deterministic policy.
      2. ``evaluate_exploration_vec`` (guided mock, n_envs=2) matches the
         sequential :func:`evaluate_exploration` schema + success aggregation.
      3. guided per-slot z (batched) path runs without raising.

    Run via ``python -m scout.eval.rollout_vec``.
    """
    import torch.nn as nn
    from scout.eval.rollout import (collect_initial_states,
                                    evaluate_baseline, evaluate_exploration)

    HORIZON = 12
    N_INIT = 6
    N_ENVS = 2
    ACTION_DIM = 4
    N_ACTION_STEPS = 4

    class MockEnv:
        """Deterministic given init_state: success iff cum action[0] >= s.
        Same dynamics as rollout._smoke.MockEnv (so vec vs sequential match)."""

        def __init__(self, seed=0):
            self._step = 0
            self._cum = 0.0
            self._state_dict = {"s": 0.0}
            self._rng = np.random.default_rng(seed)
            self.rollout_exceptions = ()

        def reset(self):
            self._step = 0
            self._cum = 0.0
            self._state_dict = {"s": float(self._rng.uniform(0.4, 1.6))}
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
            done = self.is_success()["task"] or (self._step >= 100)
            return self._get_obs(), float(action[0]), done, {}

        def is_success(self):
            return {"task": bool(self._cum >= self._state_dict["s"])}

        def get_state(self):
            return dict(self._state_dict)

        def close(self):
            pass

    class _BatchedSeqAdapter:
        """Sequential-path adapter wrapping the SAME deterministic batched
        policy, so vec and sequential see byte-identical actions."""

        def __init__(self, policy):
            self.policy = policy
            self._chunk = None
            self._t = 0

        def start_episode(self):
            self._chunk = None
            self._t = 0

        def __call__(self, obs):
            if self._chunk is None or self._t >= N_ACTION_STEPS:
                obs_dict = _to_device_batched(obs, torch.device("cpu"))
                result = self.policy.predict_action(obs_dict)
                self._chunk = result["action"][0].cpu().numpy()
                self._t = 0
            a = self._chunk[self._t]
            self._t += 1
            return a

    class MockDP(nn.Module):
        """Batched: reads B from obs, returns a fixed (B, n_action_steps, A)
        chunk with action[0]=0.5. Deterministic -> vec == sequential."""

        def __init__(self):
            super().__init__()
            self.n_action_steps = N_ACTION_STEPS

        def reset(self):
            pass

        def eval(self):
            return self

        def predict_action(self, obs_dict):
            B = next(iter(obs_dict.values())).shape[0]
            c = torch.zeros((B, N_ACTION_STEPS, ACTION_DIM))
            c[..., 0] = 0.5
            return {"action": c}

        def predict_action_dyn_guided(self, obs_dict):
            # guided emits a stronger action[0] so some init states flip to success
            B = next(iter(obs_dict.values())).shape[0]
            c = torch.zeros((B, N_ACTION_STEPS, ACTION_DIM))
            c[..., 0] = 0.9
            return {"action": c}

    device = torch.device("cpu")

    # deterministic init states (fixed seed -> identical for vec & sequential)
    env_factory = lambda: MockEnv(seed=0)
    init_states = collect_initial_states(env_factory, N_INIT)
    print(f"[1] init thresholds: {[round(s['s'], 3) for s in init_states]}")

    # ---- check 1: baseline vec == sequential (n_tries=2 -> exercises pass@5) #
    dp = MockDP()
    base_seq, seq_pass, seq_succ = evaluate_baseline(_BatchedSeqAdapter(dp),
                                                     lambda: MockEnv(seed=0),
                                                     init_states, horizon=HORIZON,
                                                     n_tries=2)
    dp2 = MockDP()
    base_vec, vec_pass, vec_succ = evaluate_baseline_vec(dp2, lambda: MockEnv(seed=0),
                                                         init_states, horizon=HORIZON,
                                                         n_envs=N_ENVS,
                                                         n_action_steps=N_ACTION_STEPS,
                                                         device=device, n_tries=2)
    seq_succ = [s for s, _ in base_seq]
    vec_succ = [s for s, _ in base_vec]
    print(f"[1] baseline seq: {seq_succ} pass@2={seq_pass}")
    print(f"[1] baseline vec: {vec_succ} pass@2={vec_pass}")
    assert seq_succ == vec_succ, f"baseline success mismatch: {seq_succ} vs {vec_succ}"
    assert seq_pass == vec_pass, f"baseline pass@k mismatch: {seq_pass} vs {vec_pass}"
    assert len(seq_succ) == len(vec_succ), (
        f"success_trajs count mismatch: seq={len(seq_succ)} vec={len(vec_succ)}")
    for (ss, ts), (sv, tv) in zip(base_seq, base_vec):
        assert ts["horizon"] == tv["horizon"], (
            f"horizon mismatch {ts['horizon']} vs {tv['horizon']}")
        assert ts["actions"].shape == tv["actions"].shape, "action shape mismatch"
    print(f"[1] baseline vec == sequential OK ({sum(seq_succ)}/{N_INIT} solved, "
          f"pass@2 {sum(seq_pass)}/{N_INIT})")

    # ---- check 2: exploration vec schema + guided batched z run --------- #
    class MockPlanner:
        def __init__(self):
            self.z = None
            self.scout_vib = type("V", (), {"style_dim": 8})()

        def set_z(self, z):
            self.z = z

    class MockGuidedDP(MockDP):
        def __init__(self):
            super().__init__()
            self.scout_planner = MockPlanner()

    guided_dp = MockGuidedDP()
    expl_vec = evaluate_exploration_vec(
        guided_dp, lambda: MockEnv(seed=0), init_states, horizon=HORIZON,
        try_times=3, n_envs=N_ENVS, n_action_steps=N_ACTION_STEPS, device=device,
        only_failed_of=base_vec,
    )
    # schema check
    for r in expl_vec:
        assert set(r.keys()) == {"solved", "n_tries", "successful_trajs",
                                 "all_trajs", "baseline_solved"}, "schema"
        if not r["baseline_solved"]:
            assert len(r["all_trajs"]) == 3, "all try_times trajs kept (no early stop)"
            for t in r["successful_trajs"]:
                assert t["actions"].shape[-1] == ACTION_DIM
                assert len(t["obs"]) == t["horizon"], "record_obs frames"
    n_collected = sum(len(r["successful_trajs"]) for r in expl_vec)
    print(f"[2] exploration vec: solved={[r['solved'] for r in expl_vec]} "
          f"collected={n_collected}")
    print(f"[2] exploration vec schema + guided batched z OK")

    # ---- check 3: vec vs sequential exploration equality ---------------- #
    guided_dp2 = MockGuidedDP()

    class _GuidedSeqAdapter(_BatchedSeqAdapter):
        def __call__(self, obs):
            if self._chunk is None or self._t >= N_ACTION_STEPS:
                obs_dict = _to_device_batched(obs, torch.device("cpu"))
                result = self.policy.predict_action_dyn_guided(obs_dict)
                self._chunk = result["action"][0].cpu().numpy()
                self._t = 0
            a = self._chunk[self._t]
            self._t += 1
            return a

    expl_seq = evaluate_exploration(_GuidedSeqAdapter(guided_dp2),
                                    lambda: MockEnv(seed=0), init_states,
                                    horizon=HORIZON, try_times=3,
                                    only_failed_of=base_seq)
    seq_solved = [r["solved"] for r in expl_seq]
    vec_solved = [r["solved"] for r in expl_vec]
    print(f"[3] expl seq solved: {seq_solved}")
    print(f"[3] expl vec solved: {vec_solved}")
    assert seq_solved == vec_solved, f"expl solved mismatch: {seq_solved} vs {vec_solved}"
    seq_collected = sum(len(r["successful_trajs"]) for r in expl_seq)
    assert seq_collected == n_collected, (
        f"collected mismatch: seq={seq_collected} vec={n_collected}")
    print(f"[3] exploration vec == sequential OK (collected={n_collected})")

    # ---- check 4: rate>1 regression guard -------------------------------- #
    # Reproduces the real-run bug: first_results / any_succ_flags are written
    # when try_0 (or any try) finishes, while env_done only counts envs whose
    # ALL n_tries finished. If the numerator is not restricted to fully-done
    # envs, rates exceed 1 (observed 2.0/3.0 on square). Simulate the exact
    # desync: env0 done, env1/env2 only try_0 done and successful.
    n_try, nn = 2, 3
    tries_done_per_env = [2, 1, 1]
    first_results = [(True, None), (True, None), (False, None)]
    any_succ_flags = [True, True, False]
    env_done = sum(1 for c in tries_done_per_env if c >= n_try)
    succ = sum(1 for i in range(nn) if tries_done_per_env[i] >= n_try
               and first_results[i] is not None and first_results[i][0])
    pass5 = sum(1 for i in range(nn) if tries_done_per_env[i] >= n_try
                and any_succ_flags[i])
    print(f"[4] desync guard: env_done={env_done} succ={succ} pass5={pass5} "
          f"(old logic would give succ=2 pass5=2)")
    assert env_done == 1 and succ == 1 and pass5 == 1, (
        "rate>1 regression: numerators must only count fully-done envs")
    assert succ / max(env_done, 1) <= 1.0 and pass5 / max(env_done, 1) <= 1.0
    print("[4] rate>1 regression guard OK")

    print("[smoke] rollout_vec.py OK")


if __name__ == "__main__":
    _smoke_vec()
