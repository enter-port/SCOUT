"""Robomimic rollout harness for SCOUT (Phase 5; scout_design.md §5).

Wires SCOUT's eval onto the LPB base DP stack:

  * **unguided** path: :class:`BaseDPAdapter` wraps the LPB
    :class:`~diffusion_policy.policy.diffusion_unet_hybrid_image_policy.DiffusionUnetHybridImagePolicy`
    and drives its ``predict_action`` (chunked replay, SOE ``RolloutDP`` shape).
  * **guided** path: :class:`GuidedAdapter` wraps
    :class:`scout.guidance.policy.ScoutPolicy` (an LPB
    ``DiffusionUnetHybridImagePolicy`` subclass with SCOUT guidance in its
    overridden ``guided_conditional_sample``) and drives its
    ``predict_action_dyn_guided``; the SCOUT planner (cost ``‖z−z_θ‖``, z_θ=reparam, seam ②
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
    img_scale: Optional[float] = 1.0 / 255.0,
    crop_size: Optional[int] = 76,
) -> Callable[[dict], dict]:
    """seam ①: LPB raw keyed ``obs_dict`` -> E_s format (scout_design.md §2).

    Returns a callable ``current_obs -> {"visual": {view: (B,1,3,H,W)},
    "proprio": (B,1,P)}`` -- the layout :class:`scout.model.encoder.StateEncoder`
    expects. ``current_obs`` is the dict the LPB ``predict_action_dyn_guided``
    threads into ``guided_conditional_sample`` (i.e. the last obs frame,
    per-key ``(B, 1, *shape)`` tensors). RGB keys are already ``CHW`` under the
    LPB shape_meta; proprio keys are concatenated into a single ``(B,1,P)``.

    Preprocessing (must match VIB training inputs exactly):
      - ``img_scale``: the env returns raw uint8-scale [0,255] images (robomimic
        ``postprocess_visual_obs=False``), but E_s was trained on [0,1] images
        (``RobomimicImageDynamicsModelDataset`` applies ``/255``) -- so divide
        by 255 here.
      - ``crop_size``: VIB training crops 84x84 -> 76x76 (random crop at train,
        center crop at val); inference uses the center crop to match the val /
        base-DP eval_fixed_crop transform. Applied only when the incoming
        spatial size is larger than ``crop_size`` (no-op for 76 or smaller).
    """
    view_names = list(view_names)
    proprio_keys = list(proprio_keys)

    def adapt(current_obs: dict) -> dict:
        visual = {}
        for v in view_names:
            img = current_obs[v].float()
            if img_scale is not None and img_scale != 1.0:
                img = img * img_scale
            if crop_size is not None:
                h, w = img.shape[-2], img.shape[-1]
                if h > crop_size and w > crop_size:
                    top = (h - crop_size) // 2
                    left = (w - crop_size) // 2
                    img = img[..., top:top + crop_size, left:left + crop_size]
            visual[v] = img
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
    # LinearNormalizer has __getitem__ but no __contains__/__iter__, so
    # `"action" in normalizer` falls back to integer-indexed iteration and
    # crashes (ParameterDict rejects int keys). Test params_dict directly.
    params = getattr(normalizer, "params_dict", None)
    if normalizer is None or params is None or "action" not in params:
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
    gate (b) dropped). A fresh skill latent ``z ~ N(0,I)`` is sampled per ROLLOUT
    in :meth:`start_episode` and locked on the planner, held fixed across ALL
    chunks of that rollout (scout_design.md §1; distinct from SOE which
    resamples z per chunk).

    The SCOUT planner (carrying the frozen ScoutVIB + seam ①/②) is attached to
    the policy ONCE at construction (caller does
    ``policy.initialize_scout_planner(planner, guidance_start_timestep, guidance_scale)``
    before passing the policy in). This class is agnostic to that -- it just
    calls ``predict_action_dyn_guided``.
    """

    def start_episode(self):
        """Reset chunk state + sample/lock a fresh skill latent for this rollout.

        z is fixed across the whole rollout (one committed skill per trajectory,
        scout_design §1); a new z is drawn on the next call. Real ScoutPolicy
        only -- mock policies without a ``scout_planner`` are left untouched
        (so rollout.py's mock smoke + self_improvement's dry-run still pass).
        """
        super().start_episode()
        planner = getattr(self.dp, "scout_planner", None)
        scout_vib = getattr(planner, "scout_vib", None) if planner is not None else None
        if scout_vib is not None:
            z = torch.randn(1, scout_vib.style_dim,
                            device=self.device, dtype=torch.float32)
            planner.set_z(z)

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
                           n_init_states: int,
                           base_seed: Optional[int] = None) -> List[dict]:
    """Generate N distinct init state_dicts via repeated ``env.reset()``.

    Robomimic ``reset()`` randomises the env; ``get_state()`` captures the
    deterministic-replay handle. ``env_factory`` is called once and the env is
    closed if it has a ``.close()`` method. The N states are fixed across all
    rounds in the self-improvement loop (fair metric comparison).

    ``base_seed`` (int): if given, init ``i`` is seeded with ``base_seed + i``
    (e.g. base_seed=42 -> seeds 42..141 for N=100) so the N init states are
    reproducible and SHARED across runs (same seed -> same 100 scenes -> a
    controlled DP-vs-SCOUT comparison). ``None`` -> unseeded (legacy default).
    """
    env = env_factory()
    try:
        states = []
        for i in range(n_init_states):
            seed = (int(base_seed) + i) if base_seed is not None else None
            try:
                env.reset(seed=seed)
            except TypeError:
                env.reset()           # mock adapters without a seed kwarg
            states.append(env.get_state())
        return states
    finally:
        if hasattr(env, "close"):
            env.close()


def _wandb_log(wandb_run, payload: dict):
    """No-op when wandb disabled (``wandb_run is None``); else ``wandb_run.log``.

    Mirrors :func:`scout.eval.rollout_vec._wandb_log` (kept local to avoid the
    rollout_vec -> rollout -> rollout_vec import cycle). Same metric names so the
    sequential fallback path reports the SAME live dashboard as the vectorized one.
    """
    if wandb_run is not None:
        wandb_run.log(payload)


def evaluate_baseline(policy_adapter, env_factory: Callable[[], Any],
                      init_states: Sequence[dict], horizon: int,
                      n_tries: int = 1,
                      record_obs: bool = False,
                      metric_prefix: str = "eval",
                      wandb_run=None
                      ) -> Tuple[List[Tuple[bool, dict]], List[bool], List[dict]]:
    """Base-DP baseline: ``n_tries`` tries per init state (default 1).

    Returns ``(results, any_success, success_trajs)``:
      * ``results``      : ``[(first_success, first_traj), ...]`` -- the FIRST
                           try per init state (single-attempt success rate is
                           computed from this; exploration's ``only_failed_of``
                           also keys off the first try).
      * ``any_success``  : per-init-state bool -- True if the base DP solved
                           that init state in ANY of its ``n_tries`` tries
                           (base DP pass@k). Equal to ``first_success`` when
                           ``n_tries == 1``.
      * ``success_trajs``: flat list of EVERY successful traj (all tries; only
                           carries per-frame obs/next_obs when ``record_obs``).

    ``record_obs=True`` stores per-frame obs/next_obs so successful base-DP
    rollouts can feed the augmented-hdf5 write-back (mode=base collection).

    ``metric_prefix`` namespaces the live progress metrics: "eval" (default)
    reports ``eval/baseline_*``; "explore" reports the SAME keys as the guided
    exploration dashboard (``explore/init_done`` / ``explore/tries_done`` /
    ``explore/collected`` / ``explore/yield``) so direct-DP collection
    (mode=base) compares side-by-side with guided exploration. Same metric names
    as the vectorized :func:`scout.eval.rollout_vec.evaluate_baseline_vec`.
    """
    env = env_factory()
    try:
        results: List[Tuple[bool, dict]] = []
        any_success: List[bool] = []
        success_trajs: List[dict] = []
        tries_done = 0
        for sd in init_states:
            first_success = False
            first_traj: Optional[dict] = None
            any_succ = False
            for j in range(int(n_tries)):
                success, traj = rollout_episode(policy_adapter, env, horizon,
                                                initial_state_dict=sd,
                                                record_obs=record_obs)
                tries_done += 1
                if j == 0:
                    first_success, first_traj = success, traj
                if success:
                    any_succ = True
                    success_trajs.append(traj)            # keep ALL (SOE-style)
            results.append((first_success, first_traj))   # type: ignore[arg-type]
            any_success.append(any_succ)
            done = len(results)
            succ = sum(1 for s, _ in results if s)
            if metric_prefix == "explore":
                _wandb_log(wandb_run, {
                    "explore/init_done": done,
                    "explore/tries_done": tries_done,
                    "explore/collected": len(success_trajs),
                    "explore/yield": len(success_trajs),
                })
            else:
                _wandb_log(wandb_run, {
                    "eval/baseline_env_done": done,
                    "eval/baseline_successes": succ,
                    "eval/baseline_success_rate": succ / max(done, 1),
                    "eval/base_pass_at_5": sum(1 for a in any_success if a) / max(done, 1),
                })
        return results, any_success, success_trajs
    finally:
        if hasattr(env, "close"):
            env.close()


def evaluate_exploration(exploration_adapter, env_factory: Callable[[], Any],
                         init_states: Sequence[dict], horizon: int,
                         try_times: int = 5,
                         only_failed_of: Optional[Sequence[Tuple[bool, dict]]] = None,
                         wandb_run=None,
                         ) -> List[dict]:
    """Exploration tries per init state.

    For each init state, run **all** ``try_times`` episodes with
    ``exploration_adapter`` (z is resampled *inside* the policy on each chunk,
    so the same adapter instance is reused) and keep **every** successful
    rollout (SOE pattern: ``run.py:128`` has no break; ``extract_useful_data_v2``
    keeps all successes). ``n_tries`` records the first-success try (for pass_at_k).

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
        tries_done = 0
        collected = 0
        for i, sd in enumerate(init_states):
            if only_failed_of is not None and only_failed_of[i][0]:
                results.append({"solved": True, "n_tries": 0,
                                "successful_trajs": [], "all_trajs": [],
                                "baseline_solved": True})
                _wandb_log(wandb_run, {
                    "explore/init_done": i + 1,
                    "explore/tries_done": tries_done,
                    "explore/collected": collected,
                    "explore/yield": collected,
                })
                continue
            entry = {"solved": False, "n_tries": 0,
                     "successful_trajs": [], "all_trajs": [],
                     "baseline_solved": False}
            first_success_try = None
            for j in range(try_times):
                # record obs/next_obs: successful exploration rollouts feed the
                # augmented-hdf5 write-back. SOE runs ALL try_times retries per
                # failed init (no early stop) and keeps EVERY successful rollout
                # (run.py:128 has no break; extract_useful_data_v2 keeps all
                # successes) -- match that here so collection is SOE-aligned.
                success, traj = rollout_episode(exploration_adapter, env, horizon,
                                                initial_state_dict=sd,
                                                record_obs=True)
                entry["all_trajs"].append(traj)
                tries_done += 1
                if success:
                    if first_success_try is None:
                        first_success_try = j + 1
                    entry["solved"] = True
                    entry["successful_trajs"].append(traj)   # keep ALL (SOE-style)
                    collected += 1
                _wandb_log(wandb_run, {
                    "explore/init_done": i,
                    "explore/tries_done": tries_done,
                    "explore/collected": collected,
                    "explore/yield": collected,
                })
            entry["n_tries"] = first_success_try if entry["solved"] else try_times
            results.append(entry)
            _wandb_log(wandb_run, {
                "explore/init_done": i + 1,
                "explore/tries_done": tries_done,
                "explore/collected": collected,
                "explore/yield": collected,
            })
        return results
    finally:
        if hasattr(env, "close"):
            env.close()


# --------------------------------------------------------------------------- #
# robomimic env factory (LPB robomimic_image_runner reuse; lazy-imported)
# --------------------------------------------------------------------------- #
class RobomimicScoutEnvAdapter:
    """Adapt :class:`RobomimicImageWrapper` to the SCOUT env contract
    (:func:`rollout_episode`): ``reset`` / ``reset_to`` / ``step`` /
    ``is_success`` / ``get_state``, with obs returned as per-key
    ``(n_obs_steps, *shape)`` numpy arrays -- the layout the LPB
    ``DiffusionUnetHybridImagePolicy.predict_action`` expects.

    The n_obs_steps stacking is managed HERE (a bounded obs buffer) rather than
    via a :class:`MultiStepWrapper`, so the per-step rollout + the SOE
    ``reset_to(start_state)`` replay both work on the same env
    (MultiStepWrapper's reset/step API has no reset_to). On reset the first frame
    is repeated to fill the buffer (matches MultiStepWrapper's pad behaviour).

    abs_action: the policy emits 10-dim abs_6drot actions (3 trans + 6 rot_6d +
    1 gripper) but the robomimic controller (``control_delta=False``, set in the
    factory) consumes 7-dim axis-angle -- this adapter converts per step,
    mirroring the LPB runner's ``undo_transform_action``.
    """

    def __init__(self, wrapper, n_obs_steps: int = 2, abs_action: bool = False):
        self.wrapper = wrapper
        self.env = wrapper.env
        self.n_obs_steps = int(n_obs_steps)
        self.abs_action = bool(abs_action)
        self.rollout_exceptions = ()  # robomimic numerical-instability swallows
        if self.abs_action:
            from diffusion_policy.model.common.rotation_transformer import (
                RotationTransformer,)
            # forward = aa->6d (dataset); .inverse = 6d->aa (policy->env), same
            # convention as the LPB runner's undo_transform_action.
            self._rot = RotationTransformer("axis_angle", "rotation_6d")
        else:
            self._rot = None
        self._obs_buffer: List[dict] = []

    # -- obs stacking: per-key (n_obs_steps, *shape) ----------------------- #
    def _stack(self) -> dict:
        from diffusion_policy.gym_util.multistep_wrapper import stack_last_n_obs
        keys = self._obs_buffer[0].keys()
        return {k: stack_last_n_obs([o[k] for o in self._obs_buffer],
                                    self.n_obs_steps)
                for k in keys}

    def _seed_buffer(self, obs: dict):
        self._obs_buffer = [obs] * self.n_obs_steps   # pad with the first frame

    # -- SCOUT env contract ------------------------------------------------ #
    def reset(self, seed: Optional[int] = None) -> dict:
        # random reset. ``seed`` (int) -> the wrapper seeds the env RNG so the
        # sampled init state is deterministic for that seed (None -> random,
        # the default; matches the previous unseeded behaviour).
        self.wrapper.init_state = None
        self.wrapper._seed = seed
        obs = self.wrapper.reset()
        self._seed_buffer(obs)
        return self._stack()

    def reset_to(self, state) -> dict:
        # deterministic replay: wrapper.reset() uses init_state -> env.reset_to.
        self.wrapper.init_state = np.asarray(state)
        self.wrapper._seed = None
        obs = self.wrapper.reset()
        self._seed_buffer(obs)
        return self._stack()

    def _to_env_action(self, action) -> np.ndarray:
        a = np.asarray(action)
        if not self.abs_action:
            return a
        raw_shape = a.shape
        if raw_shape[-1] == 20:
            # dual arm (transport): per-arm 10 (pos3|rot6|grip1) -> per-arm 7,
            # mirroring the training loader's undo_transform_action.
            a = a.reshape(-1, 2, 10)
        d_rot = a.shape[-1] - 4                        # 10 -> 6 (rot_6d portion)
        pos = a[..., :3]
        rot = np.asarray(self._rot.inverse(a[..., 3:3 + d_rot]))  # rot_6d -> aa
        gripper = a[..., [-1]]
        out = np.concatenate([pos, rot, gripper], axis=-1)
        if raw_shape[-1] == 20:
            out = out.reshape(raw_shape[:-1] + (14,))
        return out

    def step(self, action):
        env_action = self._to_env_action(action)
        obs, reward, done, info = self.wrapper.step(env_action)
        self._obs_buffer.append(obs)
        self._obs_buffer = self._obs_buffer[-self.n_obs_steps:]
        return self._stack(), reward, done, info

    def is_success(self):
        return {"task": bool(self.wrapper.get_success_label())}

    def get_state(self):
        # Fast path: flattened sim state ONLY. robomimic's
        # EnvRobosuite.get_state() also serializes the whole model to XML via
        # temp files (get_xml -> mkdtemp) on EVERY call -- profiling showed
        # that dominates rollout wall time when called per env step. Nothing
        # downstream consumes the model (reset_to replays from the flattened
        # states array), so bypass it. Value-identical to
        # ``self.env.get_state()["states"]`` (robomimic does exactly this
        # flatten() internally before attaching the xml).
        return np.array(self.env.env.sim.get_state().flatten())

    def close(self):
        if hasattr(self.env, "close"):
            self.env.close()


def make_robomimic_env_factory(dataset_path: str, shape_meta: dict,
                               n_obs_steps: int = 2,
                               abs_action: bool = False,
                               render_obs_key: str = "agentview_image"
                               ) -> Callable[[], Any]:
    """Build a :class:`RobomimicScoutEnvAdapter` factory (real robomimic run).

    Reuses the LPB env-construction path (env_meta from dataset +
    ``EnvUtils.create_env_from_metadata``). For ``abs_action`` it sets
    ``controller_configs.control_delta=False`` (absolute controller) -- mirrors
    the LPB ``robomimic_image_runner``. Lazy-imported so this module imports
    cleanly without robomimic/mujoco installed.
    """
    def factory() -> Any:
        import collections
        import robomimic.utils.file_utils as FileUtils
        import robomimic.utils.env_utils as EnvUtils
        import robomimic.utils.obs_utils as ObsUtils
        from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
            RobomimicImageWrapper,
        )

        env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
        env_meta["env_kwargs"]["use_object_obs"] = False
        if abs_action:
            env_meta["env_kwargs"]["controller_configs"]["control_delta"] = False

        # offscreen-render device (2026-08-22): the dataset's env_meta pins
        # render_gpu_device_id=0, so EVERY chain rendered on EGL device 0 --
        # under 4 concurrent chains that overloaded it and corrupted frames
        # (2333-DP r1, both SCOUT arms r4-6). round.sh exports
        # SCOUT_RENDER_GPU=<chain gpu> to spread rendering one GPU per chain;
        # robosuite 1.4.1's egl_context picks the device directly by index.
        import os as _os
        _rgpu = int(_os.environ.get("SCOUT_RENDER_GPU", -1))
        if _rgpu >= 0:
            env_meta["env_kwargs"]["render_gpu_device_id"] = _rgpu

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
        return RobomimicScoutEnvAdapter(wrapper, n_obs_steps=n_obs_steps,
                                        abs_action=abs_action)

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
    base, base_pass, base_succ = evaluate_baseline(MockDPAdapter(), env_factory,
                                                   init_states, horizon=20, n_tries=3)
    assert all(t["success"] for t in base_succ), "collected trajs must be successful"
    assert len(base_succ) >= sum(1 for s, _ in base if s), "success_trajs covers first-try"
    n_succ = sum(1 for s, _ in base if s)
    print(f"[1] baseline: {n_succ}/{len(init_states)} succeeded "
          f"(pass@{3}: {sum(base_pass)}/{len(init_states)}); "
          f"horizons={[t['horizon'] for _, t in base]}")
    assert all(t["horizon"] > 0 for _, t in base), "empty traj"
    assert all(t["actions"].shape[1] == 4 for _, t in base), "wrong action_dim"
    assert len(base_pass) == len(init_states), "any_success length"
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
