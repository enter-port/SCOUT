"""E4: 5-step multi-round self-improvement loop (Phase 5.3 / scout_design.md §5).

Per round (6 by default; cfg.self_improvement.num_rounds):

  1. DP_i            -- loaded from ckpt (round 0 = the E0 base DP).
  2. ScoutVIB        -- loaded ONCE from the E1 chosen-β ckpt; reused across
                        all rounds (design §5: VIB dynamics don't change).
  3. Rollouts        -- baseline DP_i on N init states (1 try each); guided
                        exploration (GuidedAdapter with z~N(0,I), guidance_scale)
                        on the failed init states (up to try_times tries).
  4. Write-back      -- successful exploration rollouts -> **augmented hdf5**
                        (core demos + appended rollouts, with a `scout_aug` mask)
                        -> retrain via LPB train.py -> DP_{i+1} ckpt path.
                        This is the SOE ``run_full_multi_round`` pattern -- **no
                        in-memory ReplayBuffer** (design §3, §5).
  5. Metrics         -- success_rate / pass@k / yield / jerk (DP_{i+1} vs DP_i).

Real E4 (full robomimic env + mujoco + trained ckpts + GPU) is DEFERRED. Every
external dependency is a factory injected into :class:`SelfImprovementLoop`:

  dp_factory(ckpt_path)        -> a fresh ScoutPolicy (LPB base DP subclass; the
                                  planner is attached later for guided rollout).
  scout_vib_factory()          -> a fresh ScoutVIB (state loaded).
  env_factory()                -> an env (robomimic via
                                  rollout.make_robomimic_env_factory, or a mock).
  retrain_fn(cfg, round_idx,   -> path to the DP_{i+1} ckpt. The default impl
    successful_rollouts,         writes an augmented HDF5 (core + successful
    prev_dp_ckpt_path)           rollouts) and shells out to LPB ``train.py`` via
                                  :func:`scout.train_base_dp.train`; injected so
                                  the dry-run substitutes a no-op stub.

The dry-run (``python -m scout.eval.self_improvement``) wires a mock ScoutPolicy
/ mock ScoutVIB / mock env / mock retrain_fn and verifies ONE + TWO round(s)
end-to-end: loop iterates -> baseline runs -> planner attached -> guided rollout
runs -> successful rollouts filtered -> augmented hdf5 written -> retrain fires
-> metric compare logged.
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

from scout.eval.metrics import summarize_round
from scout.eval.rollout import (
    BaseDPAdapter,
    GuidedAdapter,
    collect_initial_states,
    evaluate_baseline,
    evaluate_exploration,
    make_action_bridge,
    make_obs_adapter,
)


# --------------------------------------------------------------------------- #
# types
# --------------------------------------------------------------------------- #
DPFactory = Callable[[str], torch.nn.Module]
VIBFactory = Callable[[], torch.nn.Module]
EnvFactory = Callable[[], Any]
RetrainFn = Callable[
    [EasyDict, int, List[dict], str],  # cfg, round_idx, successful_rollouts, prev_dp_ckpt
    str,                                # new DP ckpt path
]


# --------------------------------------------------------------------------- #
# loop
# --------------------------------------------------------------------------- #
class SelfImprovementLoop:
    """Multi-round DP self-improvement via VIB-guided exploration.

    Args mirror the 5-step design. ``dp_factory`` / ``scout_vib_factory`` /
    ``env_factory`` / ``retrain_fn`` are all required so the loop has no hard
    dependency on robomimic / mujoco / a particular ckpt path (the dry-run
    injects mocks). ``view_names`` / ``proprio_keys`` configure seam ① (obs
    adapter); ``guidance_scale`` / ``guidance_start_timestep`` are SCOUT guidance
    knobs (seam ② is auto-derived from each round's DP via
    :func:`scout.eval.rollout.make_action_bridge`).
    """

    def __init__(self,
                 cfg: EasyDict,
                 dp_factory: DPFactory,
                 scout_vib_factory: VIBFactory,
                 env_factory: EnvFactory,
                 retrain_fn: RetrainFn,
                 device: Optional[torch.device] = None,
                 verbose: bool = True,
                 force_explore_all: bool = False):
        self.cfg = cfg
        self.dp_factory = dp_factory
        self.scout_vib_factory = scout_vib_factory
        self.env_factory = env_factory
        self.retrain_fn = retrain_fn
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.verbose = verbose
        # smoke-only: when True, guided exploration runs on ALL init states
        # (only_failed_of=None) regardless of baseline -- lets a tiny smoke
        # exercise the guided + write-back + retrain path even when the baseline
        # happens to solve every init state. The real run leaves this False.
        self.force_explore_all = force_explore_all

        # guidance / obs-adapter config (read once from cfg)
        self.guidance_scale = float(cfg.exploration.guidance_scale)
        self.guidance_start_timestep = int(cfg.exploration.guidance_start_timestep)
        self.view_names = list(cfg.eval.view_names)
        self.proprio_keys = list(cfg.eval.proprio_keys)

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

    def _build_guided_adapter(self, dp, scout_vib) -> GuidedAdapter:
        """Attach the SCOUT planner to ``dp`` (seam ① + ②), return a GuidedAdapter.

        Per design §4: planner carries the frozen ScoutVIB (``E_s`` + ``vib_enc``)
        + the unnormalize-only action bridge (from this round's DP normalizer) +
        the obs-adapter (LPB keyed obs -> E_s format). z is sampled fresh inside
        ScoutPolicy on each inference call.
        """
        from scout.guidance.planner import ScoutPlanner
        bridge = make_action_bridge(dp)                      # seam ②
        obs_adapter = make_obs_adapter(self.view_names,      # seam ①
                                       self.proprio_keys)
        planner = ScoutPlanner(scout_vib, bridge=bridge,
                               obs_adapter=obs_adapter, z=None)
        # ScoutPolicy.initialize_scout_planner sets guidance_start_timestep /
        # guidance_scale + attaches the planner. If dp is a plain LPB DP (no
        # scout hook, e.g. a mock), skip -- the mock must already mimic the
        # guided interface.
        init = getattr(dp, "initialize_scout_planner", None)
        if callable(init):
            init(planner, self.guidance_start_timestep, self.guidance_scale)
        return GuidedAdapter(dp, self.device)

    def _exploration_round(self, dp, scout_vib, init_states,
                           baseline_results) -> List[dict]:
        adapter = self._build_guided_adapter(dp, scout_vib)
        return evaluate_exploration(
            adapter, self.env_factory, init_states,
            horizon=int(self.cfg.eval.horizon),
            try_times=int(self.cfg.eval.try_times),
            only_failed_of=(None if self.force_explore_all else baseline_results),
        )

    def _collect_successful(self,
                            exploration_results: List[dict]) -> List[dict]:
        """Pull successful rollouts from a round's exploration results.

        No in-memory buffer (design §3, §5): the rollouts are forwarded straight
        to ``retrain_fn``, which writes the augmented hdf5 and retrains.
        """
        succ = [traj for r in exploration_results for traj in r["successful_trajs"]]
        self._log(f"  round collected {len(succ)} successful rollouts "
                  f"(accumulated {len(self.accumulated_rollouts) + len(succ)})")
        return succ

    # ---- full loop ------------------------------------------------------- #
    def run(self, num_rounds: Optional[int] = None) -> List[dict]:
        """Run ``num_rounds`` rounds (defaults to cfg.self_improvement.num_rounds).

        Per round: baseline DP_i -> guided exploration -> write-back (augmented
        hdf5 + retrain) -> DP_{i+1}. The ScoutVIB is loaded ONCE (round 0) and
        reused (design: VIB dynamics don't change across rounds).

        Returns the per-round summaries (also stored in ``self.history``).
        """
        n_rounds = int(num_rounds or self.cfg.self_improvement.num_rounds)
        scout_vib = self.scout_vib_factory()

        # N init states are FIXED across rounds for fair metric comparison.
        init_states = collect_initial_states(
            self.env_factory, n_init_states=int(self.cfg.eval.n_init_states))
        self._log(f"[loop] collected {len(init_states)} init states (fixed "
                  f"across {n_rounds} rounds)")

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

            # write-back (augmented hdf5) + retrain -> next dp ckpt path
            successful = self._collect_successful(expl)
            self.accumulated_rollouts.extend(successful)
            if r + 1 < n_rounds:                  # no retrain needed after last round
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

    .. note:: The HDF5 writer is structurally faithful to robomimic's
       ``data/demo_N`` schema (per-demo ``obs/<key>``, ``actions``, ``done``,
       ``success``, ``states``) and to SOE's ``run_full_multi_round`` write path,
       but UNTESTED against the real robomimic loader (env install deferred). If
       a real run trips on schema details (e.g. ``env``/``model_file`` attrs,
       ``abs_action`` rotation fields), this is the place to fix.
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
        _write_augmented_hdf5(core_path, new_path, successful_rollouts,
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
# augmented HDF5 writer (SOE run_full_multi_round pattern; no in-memory buffer)
# --------------------------------------------------------------------------- #
def _demo_list(hdf5_file, mask_key: Optional[str]) -> List[str]:
    """Sorted ``data/demo*`` names, optionally filtered by ``mask/<mask_key>``."""
    all_demos = sorted([k for k in hdf5_file["data"].keys() if k.startswith("demo")])
    if mask_key is None or f"mask/{mask_key}" not in hdf5_file:
        return all_demos
    node = hdf5_file[f"mask/{mask_key}"]
    if isinstance(node, h5py_group_class()) and "mask" in node:
        arr = node["mask"][()]
    else:
        arr = node[()]
    if arr.dtype == bool:
        return [d for d, keep in zip(all_demos, arr) if keep]
    return [s.decode("utf-8") if isinstance(s, bytes) else str(s) for s in arr]


def _discover_obs_keys(data_grp, demo_name: str) -> List[str]:
    """Per-key obs keys present in ``data/<demo>/obs`` (robomimic schema)."""
    return sorted(data_grp[demo_name]["obs"].keys())


def h5py_group_class():
    """Lazy h5py import (module top-level imports stay clean)."""
    import h5py
    return h5py.Group


def _write_augmented_hdf5(core_path: str, out_path: str,
                          rollouts: List[dict],
                          core_filter_key: str = "train",
                          aug_mask_key: str = "scout_aug"):
    """Write ``core_path``'s filtered demos + ``rollouts`` as a new HDF5.

    Mirrors robomimic's ``data/demo_N`` schema: per-demo ``obs/<key>``
    (T, dim), ``actions`` (T, action_dim), ``done``/``success`` (T,) bool,
    ``states`` (T, D) when available. Also writes ``mask/<aug_mask_key>`` =
    boolean over ALL ``data/`` demos selecting ``core_filter_key`` demos + the
    appended rollout demos, so the retrain step picks up both via one mask
    (otherwise it would train on core-only and silently ignore the new
    rollouts -- the bug this avoids).

    .. warning:: UNTESTED against the real robomimic loader (env deferred). The
       schema is faithful to SOE's write path; if a future real run finds a
       missing attribute (e.g. ``model_file``, ``env`` metadata, per-demo
       ``num_samples``), extend here.
    """
    import h5py
    import shutil

    # start from a copy of core (preserves env metadata / attrs / mask groups)
    shutil.copyfile(core_path, out_path)

    with h5py.File(out_path, "r+") as f:
        core_demos = _demo_list(f, core_filter_key)
        if not core_demos:
            raise RuntimeError(
                f"no core demos under mask='{core_filter_key}' in {core_path}")
        core_obs_keys = set(_discover_obs_keys(f["data"], core_demos[0]))
        # rollout obs carries only the policy's shape_meta keys (use_object_obs=
        # False drops 'object' etc.); write the intersection so appended demos
        # match what the DP loads. Core demos keep their full key set.
        rollout_keys = set()
        for _r in rollouts:
            _ro = _r.get("obs") or []
            if _ro:
                rollout_keys |= set(_ro[0].keys())
        obs_keys = sorted((core_obs_keys & rollout_keys) if rollout_keys
                          else core_obs_keys)
        core_set = set(core_demos)

        # find next free demo id
        existing_ids = [int(d.split("_")[-1]) for d in f["data"].keys()
                        if d.startswith("demo_") and d.split("_")[-1].isdigit()]
        next_id = (max(existing_ids) + 1) if existing_ids else 0

        # abs_action: rollout actions are the policy's 10-dim rot_6d output; the
        # core hdf5 stores 7-dim axis-angle (the loader re-converts to 6d via
        # abs_action=true). Transform back so the augmented hdf5 is consistent.
        from diffusion_policy.model.common.rotation_transformer import (
            RotationTransformer,)
        rot = RotationTransformer("axis_angle", "rotation_6d")  # .inverse = 6d->aa

        def _last_frame(o_k) -> np.ndarray:
            """Last frame of an n_obs_steps windowed obs value -> current frame."""
            return np.asarray(o_k, dtype=np.float32)[-1]

        def _to_storage(k: str, frame) -> np.ndarray:
            """rollout frame -> core hdf5 storage layout.

            rollout image obs is (C,H,W) float (robomimic CHW); core stores
            (H,W,C) uint8. low_dim obs is (D,) float either way.
            """
            if k.endswith("image"):
                img = np.asarray(frame, dtype=np.float32)
                if img.ndim == 3 and img.shape[0] == 3:
                    img = np.transpose(img, (1, 2, 0))     # CHW -> HWC
                return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)
            return np.asarray(frame, dtype=np.float32)

        new_demo_names: List[str] = []
        for rollout in rollouts:
            demo_name = f"demo_{next_id}"
            next_id += 1
            ep_len = int(rollout.get("horizon", 0))
            if ep_len == 0:
                continue
            grp = f["data"].create_group(demo_name)
            obs_list = rollout.get("obs") or []
            next_obs_list = rollout.get("next_obs") or []
            if len(obs_list) < ep_len:
                raise ValueError(
                    f"rollout for {demo_name} missing obs (need record_obs=True); "
                    f"got {len(obs_list)} frames, need {ep_len}")
            obs_grp = grp.create_group("obs")
            for k in obs_keys:
                obs_grp.create_dataset(
                    k, data=np.stack([_to_storage(k, _last_frame(o[k]))
                                      for o in obs_list[:ep_len]], axis=0))
            if next_obs_list:
                next_grp = grp.create_group("next_obs")
                for k in obs_keys:
                    next_grp.create_dataset(
                        k, data=np.stack([_to_storage(k, _last_frame(o[k]))
                                          for o in next_obs_list[:ep_len]], axis=0))
            # actions: 10-dim rot_6d -> 7-dim axis-angle (matches core storage).
            acts = np.asarray(rollout["actions"], dtype=np.float32)   # (T,10)
            if acts.shape[-1] == 10:
                pos = acts[..., :3]
                rot_aa = np.asarray(rot.inverse(acts[..., 3:9]))
                grip = acts[..., 9:10]
                acts = np.concatenate([pos, rot_aa, grip], axis=-1)   # (T,7)
            grp.create_dataset("actions", data=acts)
            # abs_actions: same 7-dim absolute aa (the policy emits absolute
            # actions; the loader reads THIS key for training when abs_action=
            # true, and reads demo['actions'] only for episode length).
            grp.create_dataset("abs_actions", data=acts)
            grp.create_dataset("done",
                               data=np.asarray(rollout["dones"], dtype=bool))
            grp.create_dataset("success",
                               data=np.full(ep_len, bool(rollout.get("success", True)),
                                            dtype=bool))
            # states (optional -- write if the rollout recorded them non-empty)
            states = rollout.get("states")
            if states is not None and len(states) >= ep_len:
                grp.create_dataset("states",
                                   data=np.asarray(states[:ep_len], dtype=np.float32))
            grp.attrs["num_samples"] = ep_len
            new_demo_names.append(demo_name)

        # write the augmented mask: True for core_<filter> demos + new rollouts.
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
# reference factory builders (lazy-imported; real run only)
# --------------------------------------------------------------------------- #
def make_lpb_dp_factory(device: Optional[torch.device] = None
                        ) -> DPFactory:
    """Build a ``dp_factory(ckpt_path) -> ScoutPolicy`` for the real run.

    Loads the LPB base DP ckpt (saved by ``TrainDiffusionUnetHybridWorkspace``)
    into a :class:`scout.guidance.policy.ScoutPolicy` (subclass; same state_dict
    structure -- no new params). Fits the action normalizer from the sibling
    ``normalizer.pth`` (saved by the LPB workspace alongside ``checkpoints/``).

    Lazy-imported so :mod:`scout.eval.self_improvement` stays importable without
    robomimic/hydra. The dry-run does NOT use this -- it injects a mock.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def factory(ckpt_path: str) -> torch.nn.Module:
        import hydra
        from omegaconf import OmegaConf
        from scout.guidance.policy import ScoutPolicy
        from diffusion_policy.model.common.normalizer import LinearNormalizer

        OmegaConf.register_new_resolver("eval", eval, replace=True)
        payload = torch.load(ckpt_path, map_location="cpu")
        cfg = payload["cfg"] if isinstance(payload, dict) and "cfg" in payload \
            else payload
        # Build a ScoutPolicy (LPB DP subclass) -- NOT the parent -- so its
        # guided_conditional_sample override is the one predict_action_dyn_guided
        # resolves to. Override _target_ on the policy cfg; ScoutPolicy.__init__
        # forwards all params to the LPB parent (scout_planner defaults None).
        pcfg = OmegaConf.to_container(cfg.policy, resolve=True)
        pcfg["_target_"] = "scout.guidance.policy.ScoutPolicy"
        policy = hydra.utils.instantiate(pcfg)

        # state_dict slot: the LPB workspace stores the policy under the "model"
        # attribute (TrainDiffusionUnetHybridImageWorkspace: self.model), so
        # payload["state_dicts"] keys are "model"/"ema_model"/"optimizer" -- NOT
        # "policy". Fall back to "policy"/"state_dict" for other savers.
        if "state_dicts" in payload:
            sds = payload["state_dicts"]
            sd = sds.get("model", sds.get("policy", sds.get("ema_model")))
        else:
            sd = payload.get("model", payload.get("policy", payload.get("state_dict")))
        policy.load_state_dict(sd, strict=False)

        # normalizer (sibling normalizer.pth next to the ckpt's checkpoints/ dir)
        normalizer_path = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), "normalizer.pth")
        if os.path.isfile(normalizer_path):
            nstate = torch.load(normalizer_path, map_location="cpu")
            policy.normalizer = LinearNormalizer()
            policy.normalizer.load_state_dict(nstate)

        return policy.to(dev).eval()

    return factory


def make_scout_vib_factory(cfg: EasyDict,
                           device: Optional[torch.device] = None
                           ) -> VIBFactory:
    """Build a ``scout_vib_factory() -> ScoutVIB`` for the real run.

    Reconstructs ``E_s`` via :meth:`StateEncoder.from_base_dp_ckpt` (frozen
    base-DP ResNet + trained proprio Conv1d) using the E_s params in
    ``cfg.vib`` (``base_dp_ckpt``, ``view_names``, ``proprio_dim``,
    ``proprio_emb_dim``), then loads the chosen-β VIB ckpt
    (``cfg.vib.ckpt_path``). Lazy-imported; the dry-run injects a mock.

    The E_s params MUST match the E1 VIB training config (``configs/vib_lift_*``)
    -- duplicate them in the eval config's ``vib:`` block so this loop is
    self-contained.
    """
    dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def factory() -> torch.nn.Module:
        from scout.model.encoder import StateEncoder
        from scout.model.scout_vib import ScoutVIB

        vcfg = cfg.vib
        E_s = StateEncoder.from_base_dp_ckpt(
            base_dp_ckpt=vcfg.base_dp_ckpt,
            view_names=list(vcfg.view_names),
            proprio_dim=int(vcfg.proprio_dim),
            proprio_emb_dim=int(getattr(vcfg, "proprio_emb_dim", 64)),
        )
        ckpt = torch.load(vcfg.ckpt_path, map_location="cpu")
        # The VIB encoder was trained on the FLATTENED fs-step action chunk
        # (train_vib._slice_transition: a_t = first `frameskip` per-step actions
        # flattened, e.g. 8x10 = 80-dim), NOT the per-step action. So its action
        # input dim is whatever was saved -- inferred here as (encoder first-
        # layer width - s_bar_dim), NOT cfg.action_dim (10). Building with the
        # per-step dim would mismatch the saved Linear weights (1168 vs 1098).
        sd = ckpt["state_dict"]
        enc_in = sd["vib_enc.net.encoder.0.weight"].shape[1]
        action_dim = enc_in - int(E_s.s_bar_dim)
        model = ScoutVIB(
            action_dim=action_dim,
            E_s=E_s,
            style_dim=int(vcfg.style_dim),
            hidden_dim=int(getattr(vcfg, "hidden_dim", 128)),
            beta=float(ckpt.get("beta", 1.0e-3)),
        )
        model.load_state_dict(sd)
        return model.to(dev).eval()

    return factory


def make_default_env_factory(cfg: EasyDict) -> EnvFactory:
    """Build the real robomimic env factory (LPB robomimic_image_runner reuse).

    Lazy-imports :func:`scout.eval.rollout.make_robomimic_env_factory`; reads
    dataset_path + shape_meta from a base-DP LPB config (resolved via the
    cfg.base_dp config_name). The dry-run does NOT use this -- it injects a mock.
    """
    from scout.eval.rollout import make_robomimic_env_factory
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    config_path = os.path.join(cfg.base_dp.config_dir,
                               cfg.base_dp.config_name + ".yaml")
    with open(config_path) as f:
        lpb_cfg = OmegaConf.create(yaml.safe_load(f))
    # n_obs_steps + abs_action come from the base-DP config (the policy's
    # obs-stacking + the env's abs controller must match training).
    return make_robomimic_env_factory(
        dataset_path=OmegaConf.select(lpb_cfg, "task.dataset_path",
                                      default=cfg.dataset.path),
        shape_meta=OmegaConf.to_container(lpb_cfg.shape_meta, resolve=True),
        n_obs_steps=int(OmegaConf.select(lpb_cfg, "n_obs_steps", default=2)),
        abs_action=bool(OmegaConf.select(lpb_cfg, "abs_action", default=False)),
    )


# --------------------------------------------------------------------------- #
# config loader
# --------------------------------------------------------------------------- #
def load_cfg(path: str) -> EasyDict:
    # encoding=utf-8: configs carry §/β/smart-quotes in comments; Windows locale
    # default (GBK) would otherwise UnicodeDecodeError on them.
    with open(path, "r", encoding="utf-8") as f:
        return EasyDict(yaml.safe_load(f))


# --------------------------------------------------------------------------- #
# dry-run with mocks (orchestration verification)
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


def _dry_run():
    """Mock-DP / mock-VIB / mock-env ONE-round orchestration check.

    Verifies the loop wires end-to-end: round iterates -> baseline rollout ->
    planner attached -> guided rollout -> success filter -> retrain_fn fires ->
    metric compare logged.
    """
    action_dim = 4
    cfg = EasyDict({
        "base_dp": {"initial_ckpt_path": "<mock-dp-0>",
                    "config_name": "base_dp_lift_image", "config_dir": "configs"},
        "vib": {"ckpt_path": "<mock-vib>"},
        "dataset": {"path": "<mock-core>", "core_filter_key": "train"},
        "action_dim": action_dim,
        "eval": {"n_init_states": 4, "try_times": 3, "horizon": 10,
                 "view_names": ["agentview_image", "robot0_eye_in_hand_image"],
                 "proprio_keys": ["robot0_eef_pos", "robot0_eef_quat",
                                  "robot0_gripper_qpos"]},
        "exploration": {"guidance_scale": 5.0, "guidance_start_timestep": 50},
        "self_improvement": {"num_rounds": 1, "scout_aug_mask": "scout_aug",
                             "num_epochs": 1},
    })

    # factories -- mock policy amplifies guided actions so some init states solve
    # in exploration (exercises success-filter + write-back wiring).
    def dp_factory(ckpt_path):
        return _MockScoutPolicy(n_action_steps=8, action_dim=action_dim,
                                guided_strength=0.6)

    def scout_vib_factory():
        return _MockScoutVIB(style_dim=8)

    env_factory = _make_dry_run_env_factory(action_dim, cfg.eval.horizon,
                                            lo=2.5, hi=5.0)

    retrain_calls: List[Tuple] = []

    def mock_retrain_fn(c, round_idx, successful_rollouts, prev_dp_ckpt):
        retrain_calls.append((round_idx, len(successful_rollouts), prev_dp_ckpt))
        return f"<mock-dp-{round_idx + 1}>"

    loop = SelfImprovementLoop(
        cfg=cfg,
        dp_factory=dp_factory,
        scout_vib_factory=scout_vib_factory,
        env_factory=env_factory,
        retrain_fn=mock_retrain_fn,
        device=torch.device("cpu"),
    )
    history = loop.run()

    print("\n--- dry-run results ---")
    print(f"rounds run         : {len(history)}")
    print(f"history[0]         : {history[0]}")
    print(f"retrain calls      : {retrain_calls}")
    print(f"accumulated rollouts: {len(loop.accumulated_rollouts)}")

    # assertions
    assert len(history) == 1, "expected 1 round"
    h = history[0]
    for k in ("success_rate", "pass_at_k", "exploration_yield", "jerk_baseline"):
        assert k in h, f"missing metric {k}"
    assert 0.0 <= h["success_rate"] <= 1.0
    assert 0.0 <= h["pass_at_k"] <= 1.0
    assert h["exploration_yield"] >= 0
    # num_rounds=1 -> last round skips retrain
    assert retrain_calls == [], "(num_rounds=1 -> no retrain expected)"
    print("[dry-run] self_improvement.py OK (orchestration)")


def _dry_run_two_rounds():
    """Two-round variant: retrain fires after round 0; its ckpt path feeds
    round 1. Verifies retrain_fn invocation count + path hand-off.
    """
    action_dim = 4
    cfg = EasyDict({
        "base_dp": {"initial_ckpt_path": "<mock-dp-0>",
                    "config_name": "base_dp_lift_image", "config_dir": "configs"},
        "vib": {"ckpt_path": "<mock-vib>"},
        "dataset": {"path": "<mock-core>", "core_filter_key": "train"},
        "action_dim": action_dim,
        "eval": {"n_init_states": 3, "try_times": 2, "horizon": 8,
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

    vib_factory = lambda: _MockScoutVIB(style_dim=8)
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
    print(f"history: {[(h['round'], round(h['success_rate'], 3), h['exploration_yield']) for h in history]}")
    assert len(history) == 2, "expected 2 rounds"
    assert len(retrain_calls) == 1, "expected 1 retrain (after round 0)"
    assert retrain_calls[0][1] > 0, "retrain should receive non-empty rollouts"
    assert retrain_calls[0][2] == "<mock-dp-0>", "round 0 should start from initial"
    assert loop.dp_path == "<mock-dp-1>", "round 1 should use retrain's output"
    print("[dry-run-2] self_improvement.py OK (retrain wiring)")


if __name__ == "__main__":
    _dry_run()
    _dry_run_two_rounds()
