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


if __name__ == "__main__":
    main()
