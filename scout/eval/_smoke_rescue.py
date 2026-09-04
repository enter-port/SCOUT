"""Hermetic smoke for the SOE rescue protocol (scout.eval.rollout_pipeline.
_run_rescue + rollout_vec first_traj tagging). Pure mocks -- no robomimic.

Env design: each init state carries ``need`` = the attempt count at which it
starts succeeding (deterministic; attempt counter is global per need value).
  need=1 -> solved on the baseline try      (eval-solved: no data)
  need=2..5 -> rescued on retry (need-1)    (successes -> DP + dyn)
  need=6 -> all 5 retries fail              (FIRST retry only -> dyn)
The current attempt number is exposed through obs eef_pos[..., 0] so the
"first retry" selection is verifiable from the recorded frames.

Run:  python -m scout.eval._smoke_rescue
"""
from collections import defaultdict

import numpy as np
import torch
from easydict import EasyDict

from scout.eval.rollout_pipeline import RolloutPipeline

HORIZON, N_ENVS, N_ACT = 30, 3, 4
NEEDS = [1, 2, 3, 4, 5, 7]          # per-init difficulty (7 > 1+5 retries -> all-fail)
ATTEMPTS = defaultdict(int)         # need -> attempts so far


class MockEnv:
    def __init__(self, seed=0):
        self._need = 0

    def _obs(self, attempt):
        return {
            "agentview_image": np.zeros((2, 3, 4, 4), dtype=np.float32),
            "robot0_eye_in_hand_image": np.ones((2, 3, 4, 4), dtype=np.float32),
            "robot0_eef_pos": np.full((2, 3), float(attempt), dtype=np.float32),
            "robot0_eef_quat": np.zeros((2, 4), dtype=np.float32),
            "robot0_gripper_qpos": np.zeros((2, 2), dtype=np.float32),
        }

    def reset_to(self, state):
        self._need = state["need"]
        self._attempt = ATTEMPTS[self._need] = ATTEMPTS[self._need] + 1
        return self._obs(self._attempt)

    def step(self, action):
        return self._obs(self._attempt), 0.0, True, {}   # done after 1 step

    def is_success(self):
        return {"task": self._attempt >= self._need}

    def get_state(self):
        return {"need": self._need}

    def rollout_exceptions(self):
        return ()

    def close(self):
        pass


class MockDP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.n_action_steps = N_ACT

    def reset(self):
        pass

    def eval(self):
        return self

    def predict_action(self, obs):
        B = next(iter(obs.values())).shape[0]
        return {"action": torch.zeros((B, N_ACT, 7))}


def main():
    cfg = EasyDict({
        "eval": {"horizon": HORIZON, "n_init_states": len(NEEDS),
                 "n_envs": N_ENVS, "seed": 42, "log_every": 2,
                 "view_names": ["agentview_image"],
                 "proprio_keys": ["robot0_eef_pos", "robot0_eef_quat",
                                  "robot0_gripper_qpos"]},
        "exploration": {"guidance_scale": 0.01,
                        "guidance_start_timestep": 50},
    })
    pipe = RolloutPipeline(
        cfg=cfg, dp_factory=lambda ckpt: MockDP().eval(),
        scout_vib_factory=None, env_factory=lambda: MockEnv(),
        device=torch.device("cpu"), guided=False,
    )
    init_states = [{"need": n} for n in NEEDS]
    # pre-register the baseline attempts exactly as collect_initial_states
    # would (it resets each scene once); we call the internal pieces directly
    # via run(), so instead feed the states by monkey-patching the collector.
    import scout.eval.rollout_pipeline as rp
    orig_collect = rp.collect_initial_states
    rp.collect_initial_states = lambda ef, n_init_states, base_seed=None: \
        init_states[:int(n_init_states)]
    try:
        result = pipe.run("mock-ckpt", explore_mode="rescue",
                          explore_try_times=5)
    finally:
        rp.collect_initial_states = orig_collect

    m = result["metrics"]
    trajs, all_trajs = result["trajs"], result["all_trajs"]

    # ---- expected bookkeeping (see module docstring) ---------------------- #
    # baseline: only need=1 solved. rescued: needs 2..5. all-fail: need=6.
    assert m["baseline_solved"] == 1, m
    assert m["n_failed"] == 5, m
    assert m["exploration_rescued"] == 4, m
    assert abs(m["pass_at_5"] - 5 / 6) < 1e-9, m
    # DP data: successes of needs 2,3,4,5 -> 5+4+3+2 = 14
    assert len(trajs) == 14, len(trajs)
    assert m["collected_trajs"] == 14, m
    # dyn data: those 14 + FIRST retry of need=6 -> 15
    assert len(all_trajs) == 15, len(all_trajs)
    assert m["n_all_trajs"] == 15, m
    # every collected traj carries per-frame obs (hdf5 write-back needs them)
    assert all("obs" in t and len(t["obs"]) > 0 for t in trajs)

    # ---- the need=7 init contributed exactly ONE dyn traj = its FIRST retry
    # (attempt 2: baseline was attempt 1). Identify it via the obs marker. --- #
    # successful trajs started at attempts >= their need; the all-fail first
    # retry is the single traj whose start attempt == 2 and which is a retry
    # of need=7: it is the only dyn traj that is NOT in trajs.
    dyn_only = [t for t in all_trajs if id(t) not in {id(x) for x in trajs}]
    assert len(dyn_only) == 1, len(dyn_only)
    start_attempt = float(dyn_only[0]["obs"][0]["robot0_eef_pos"][0, 0])
    assert start_attempt == 2.0, (
        f"all-failed init must contribute its FIRST retry (attempt 2), "
        f"got attempt {start_attempt}")

    print("[smoke_rescue] OK:",
          {k: m[k] for k in ("baseline_solved", "n_failed",
                             "exploration_rescued", "pass_at_5",
                             "collected_trajs", "n_all_trajs")})

    _smoke_rescue_spool(cfg, result)


def _smoke_rescue_spool(cfg, result_ref):
    """End-to-end TrajSpool parity (2026-09-04 OOM fix): rerun the SAME
    deterministic mock protocol with ``traj_sink=TrajSpool.on_traj`` and
    assert (a) metrics/counts identical to the sink-less run, (b) the
    spool-assembled success/all hdf5 are value-identical to a one-shot
    ``write_rollouts_to_hdf5`` on the reference lists. Needs pytorch3d
    (RotationTransformer) -- skipped where unavailable (local dev boxes).
    """
    try:
        import pytorch3d  # noqa: F401
    except ImportError:
        print("[smoke_rescue] spool scenario SKIPPED (pytorch3d unavailable)")
        return
    import os
    import tempfile
    import h5py
    from scout.eval.hdf5_writer import write_rollouts_to_hdf5
    from scout.eval.traj_spool import TrajSpool

    tmp = tempfile.mkdtemp(prefix="smoke_rescue_spool_")
    core = os.path.join(tmp, "core.hdf5")
    with h5py.File(core, "w") as f:
        d = f.create_group("data")
        g = d.create_group("demo_0")
        for k, shape in (("obs/agentview_image", (2, 4, 4, 3)),
                         ("obs/robot0_eye_in_hand_image", (2, 4, 4, 3))):
            g.create_dataset(k, data=np.zeros(shape, dtype=np.uint8))
        for k, shape in (("obs/robot0_eef_pos", (2, 3)),
                         ("obs/robot0_eef_quat", (2, 4)),
                         ("obs/robot0_gripper_qpos", (2, 2))):
            g.create_dataset(k, data=np.zeros(shape, dtype=np.float32))
        g.attrs["num_samples"] = 2

    ref_s = os.path.join(tmp, "ref_success.hdf5")
    ref_a = os.path.join(tmp, "ref_all.hdf5")
    write_rollouts_to_hdf5(core, ref_s, result_ref["trajs"],
                           core_filter_key="train")
    write_rollouts_to_hdf5(core, ref_a, result_ref["all_trajs"],
                           core_filter_key="train")

    s_path = os.path.join(tmp, "sp_success.hdf5")
    a_path = os.path.join(tmp, "sp_all.hdf5")
    spool = TrajSpool(core, s_path, a_path, core_filter_key="train",
                      rule="rescue", try_times=5, flush_every=3, verbose=False)
    ATTEMPTS.clear()          # replay the identical deterministic attempt stream
    pipe2 = RolloutPipeline(
        cfg=cfg, dp_factory=lambda ckpt: MockDP().eval(),
        scout_vib_factory=None, env_factory=lambda: MockEnv(),
        device=torch.device("cpu"), guided=False,
    )
    import scout.eval.rollout_pipeline as rp
    init_states = [{"need": n} for n in NEEDS]
    orig_collect = rp.collect_initial_states
    rp.collect_initial_states = lambda ef, n_init_states, base_seed=None: \
        init_states[:int(n_init_states)]
    try:
        result2 = pipe2.run("mock-ckpt", explore_mode="rescue",
                            explore_try_times=5, traj_sink=spool.on_traj)
    finally:
        rp.collect_initial_states = orig_collect
    res = spool.finalize()

    assert result2["metrics"] == result_ref["metrics"], (
        "sink-mode metrics diverged from the sink-less run")
    assert res["success_demos"] == len(result_ref["trajs"]), res
    assert res["all_demos"] == len(result_ref["all_trajs"]), res
    assert spool.n_success == len(result_ref["trajs"])
    assert spool.n_all == len(result_ref["all_trajs"])

    for got, ref, tag in ((s_path, ref_s, "success"), (a_path, ref_a, "all")):
        with h5py.File(got, "r") as fg, h5py.File(ref, "r") as fr:
            assert sorted(fg["data"].keys()) == sorted(fr["data"].keys()), tag
            for demo in sorted(fr["data"].keys()):
                for sub in ("obs", "next_obs"):
                    for k in fr["data"][demo].get(sub, {}):
                        np.testing.assert_array_equal(
                            fg[f"data/{demo}/{sub}/{k}"][()],
                            fr[f"data/{demo}/{sub}/{k}"][()],
                            err_msg=f"{tag}:{demo}/{sub}/{k}")
                for ds in ("actions", "done", "success"):
                    np.testing.assert_array_equal(
                        fg[f"data/{demo}/{ds}"][()],
                        fr[f"data/{demo}/{ds}"][()],
                        err_msg=f"{tag}:{demo}/{ds}")
            np.testing.assert_array_equal(
                fg["mask/scout_aug/mask"][()], fr["mask/scout_aug/mask"][()])
    print(f"[smoke_rescue] spool parity OK: success={res['success_demos']} "
          f"all={res['all_demos']} value-identical to one-shot (flush_every=3)")


if __name__ == "__main__":
    main()
