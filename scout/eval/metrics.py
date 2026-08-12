"""SOE-aligned evaluation metrics for SCOUT (Phase 5.2).

Implements the four metrics from scout_design.md §5 / evaluation_plan.md §一.4:

  success_rate_per_round : per-round baseline success rate over N init states.
  pass_at_k              : fraction of init states solved within k tries
                           (k = try_times = 5; baseline-solved counts as 0 tries).
  exploration_yield      : # successful exploration rollouts in the round.
  jerk                   : mean per-frame 3rd-difference norm of action chunks
                           (reused from Phase-4 ``guidance_checks.jerk`` -- the
                           single implementation; SCOUT stage-1 always reports
                           jerk for trajectory snippets, not per-chunk).

All metrics take the outputs of :mod:`scout.eval.rollout` directly:
``baseline_results`` = list of ``(success, traj)``;
``exploration_results`` = list of dicts from :func:`evaluate_exploration`.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

# Single source of truth for the jerk formula -- Phase-4 guidance_checks.jerk.
# Re-imported here so callers compute eval/jerk from one place.
from scout.eval.guidance_checks import jerk as _chunk_jerk


# --------------------------------------------------------------------------- #
# SOE four metrics
# --------------------------------------------------------------------------- #
def success_rate_per_round(baseline_results: Sequence[Tuple[bool, dict]]) -> float:
    """Fraction of init states solved by the baseline DP in 1 try each."""
    if not baseline_results:
        return 0.0
    solved = sum(1 for s, _ in baseline_results if s)
    return solved / len(baseline_results)


def pass_at_k(exploration_results: Sequence[dict],
              baseline_results: Optional[Sequence[Tuple[bool, dict]]] = None,
              k: Optional[int] = None) -> float:
    """Fraction of init states solved within ``k`` tries.

    An init state solved by the baseline counts as solved in 0 tries (i.e.
    always within ``k``) -- evaluation_plan.md §一.3: baseline runs first, then
    exploration only on the failures.

    Args:
        exploration_results: per-init-state dict from
                             :func:`rollout.evaluate_exploration`
                             (``solved``, ``n_tries``).
        baseline_results   : optional per-init-state ``(success, traj)``.
        k                  : defaults to ``max(n_tries)``.
    """
    N = len(exploration_results)
    if N == 0:
        return 0.0
    if k is None:
        k = max((r["n_tries"] for r in exploration_results), default=1)
    solved = 0
    for i, r in enumerate(exploration_results):
        baseline_solved = bool(baseline_results and baseline_results[i][0])
        if baseline_solved or (r["solved"] and r["n_tries"] <= k):
            solved += 1
    return solved / N


def exploration_yield(exploration_results: Sequence[dict]) -> int:
    """Total # successful exploration rollouts in the round.

    With first-success stop (the default in :func:`rollout.evaluate_exploration`),
    this is 0 or 1 per init state; summing gives the round's yield.
    """
    return sum(len(r["successful_trajs"]) for r in exploration_results)


# --------------------------------------------------------------------------- #
# jerk (trajectory-level; reuses Phase-4 chunk formula)
# --------------------------------------------------------------------------- #
def jerk(actions) -> float:
    """Mean per-frame 3rd-difference norm of an action sequence.

    Accepts ``(T, A)`` or ``(B, T, A)`` (numpy or tensor). Returns 0.0 for
    sequences shorter than 4 frames. Formula matches Phase-4
    :func:`scout.eval.guidance_checks.jerk`:

        d3[t] = a[t+3] − 3 a[t+2] + 3 a[t+1] − a[t]
        jerk  = mean_t ‖d3[t]‖₂
    """
    a = torch.as_tensor(actions, dtype=torch.float32)
    if a.dim() == 1:
        a = a.unsqueeze(-1)                # (T,) -> (T, 1)
    if a.dim() == 2:
        a = a.unsqueeze(0)                 # (T, A) -> (1, T, A)
    return _chunk_jerk(a)


def jerk_of_results(results: Sequence[Tuple[bool, dict]],
                    only_successful: bool = False) -> float:
    """Aggregate ``jerk`` over many rollout results.

    Args:
        results         : ``[(success, traj), ...]`` (baseline) or a flat list
                          of traj dicts (each entry treated as a rollout).
        only_successful : if True, skip failed rollouts (jerk on fails is
                          misleading -- the action sequence is incomplete).
    Returns the mean jerk over all included rollouts with >= 4 frames (0.0 if
    none qualify).
    """
    jerks: List[float] = []
    for entry in results:
        if isinstance(entry, tuple) and len(entry) == 2:
            success, traj = entry
        else:
            success, traj = True, entry           # plain traj dict
        if only_successful and not success:
            continue
        j = jerk(traj["actions"])
        if j > 0.0:                               # skips T<4
            jerks.append(j)
    return float(np.mean(jerks)) if jerks else 0.0


def jerk_of_exploration(exploration_results: Sequence[dict]) -> float:
    """Mean jerk over all *successful* exploration rollouts (jerks of failures
    are not informative -- they're truncated)."""
    jerks: List[float] = []
    for r in exploration_results:
        for traj in r["successful_trajs"]:
            j = jerk(traj["actions"])
            if j > 0.0:
                jerks.append(j)
    return float(np.mean(jerks)) if jerks else 0.0


# --------------------------------------------------------------------------- #
# round summary
# --------------------------------------------------------------------------- #
def summarize_round(baseline_results: Sequence[Tuple[bool, dict]],
                    exploration_results: Sequence[dict],
                    try_times: int = 5,
                    base_pass5: Optional[Sequence[bool]] = None) -> dict:
    """One-line round summary: all four metrics + counts.

    ``try_times`` is the k for pass_at_k. Returns a plain dict (yaml-safe) so
    the self-improvement loop can append it to a per-round log.

    ``base_pass5`` (optional): per-init-state bool -- whether the base DP solved
    that init state in ANY of its baseline tries (base DP pass@k, distinct from
    the guided-exploration ``pass_at_k``). When given, the summary carries
    ``base_pass_at_5`` = fraction of init states solved at least once across the
    baseline tries. ``success_rate`` still uses the FIRST try only
    (single-attempt, paper-comparable).
    """
    summary = {
        "n_init_states": len(baseline_results),
        "success_rate": success_rate_per_round(baseline_results),
        "pass_at_k": pass_at_k(exploration_results, baseline_results, k=try_times),
        "exploration_yield": exploration_yield(exploration_results),
        "jerk_baseline": jerk_of_results(baseline_results, only_successful=True),
        "jerk_exploration": jerk_of_exploration(exploration_results),
        "baseline_solved": int(sum(1 for s, _ in baseline_results if s)),
        "exploration_solved": int(sum(1 for r in exploration_results if r["solved"]
                                      and not r.get("baseline_solved", False))),
    }
    if base_pass5 is not None:
        n = len(base_pass5)
        summary["base_pass_at_5"] = (
            float(sum(1 for s in base_pass5 if s) / n) if n else 0.0)
    return summary


# --------------------------------------------------------------------------- #
# smoke test on dummy data
# --------------------------------------------------------------------------- #
def _smoke():
    """Dummy-rollout metrics smoke test. Run via ``python -m scout.eval.metrics``.

    Checks (on synthetic data): all four metrics are finite & in valid ranges
    (success_rate / pass_at_k in [0,1]; yield >= 0; jerk > 0 for T>=4); the
    baseline-vs-exploration interaction (baseline-solved init states count as
    pass_at_k solved) holds.
    """
    rng = np.random.default_rng(0)

    def make_traj(T=15, A=4, succeeded=True):
        return {"actions": rng.standard_normal((T, A)).astype(np.float32),
                "rewards": np.zeros(T, dtype=np.float32),
                "dones": np.zeros(T, dtype=bool),
                "states": [{} for _ in range(T)],
                "obs": [], "next_obs": [],
                "horizon": T, "success": succeeded,
                "initial_state_dict": None}

    # 6 init states: 3 baseline-successful, 3 baseline-failed; of the failed,
    # 2 eventually solved in exploration (yields = 2).
    S, F = dict(succeeded=True), dict(succeeded=False)
    baseline_results = [
        (True, make_traj(**S)),
        (True, make_traj(**S)),
        (True, make_traj(**S)),
        (False, make_traj(**F)),
        (False, make_traj(**F)),
        (False, make_traj(**F)),
    ]
    exploration_results = [
        {"solved": True, "n_tries": 0, "successful_trajs": [],
         "all_trajs": [], "baseline_solved": True},   # baseline-solved
        {"solved": True, "n_tries": 0, "successful_trajs": [],
         "all_trajs": [], "baseline_solved": True},
        {"solved": True, "n_tries": 0, "successful_trajs": [],
         "all_trajs": [], "baseline_solved": True},
        {"solved": True, "n_tries": 2, "successful_trajs": [make_traj(**S)],
         "all_trajs": [make_traj(**F), make_traj(**S)], "baseline_solved": False},
        {"solved": True, "n_tries": 4, "successful_trajs": [make_traj(**S)],
         "all_trajs": [make_traj(**F)] * 4 + [make_traj(**S)], "baseline_solved": False},
        {"solved": False, "n_tries": 5, "successful_trajs": [],
         "all_trajs": [make_traj(**F)] * 5, "baseline_solved": False},
    ]

    sr = success_rate_per_round(baseline_results)
    pk = pass_at_k(exploration_results, baseline_results, k=5)
    ey = exploration_yield(exploration_results)
    jb = jerk_of_results(baseline_results)
    je = jerk_of_exploration(exploration_results)

    print(f"success_rate_per_round = {sr:.4f}  (expect 0.5000)")
    print(f"pass_at_k (k=5)         = {pk:.4f}  (expect 0.8333 = 5/6)")
    print(f"exploration_yield       = {ey}      (expect 2)")
    print(f"jerk_baseline           = {jb:.4f}  (expect > 0)")
    print(f"jerk_exploration        = {je:.4f}  (expect > 0)")

    assert 0.0 <= sr <= 1.0, "success_rate out of [0,1]"
    assert 0.0 <= pk <= 1.0, "pass_at_k out of [0,1]"
    assert ey == 2, f"yield expected 2, got {ey}"
    assert jb > 0.0, "baseline jerk must be > 0"
    assert je > 0.0, "exploration jerk must be > 0"
    assert abs(sr - 0.5) < 1e-6, "success_rate value"
    assert abs(pk - 5.0 / 6.0) < 1e-6, "pass_at_k value"

    summary = summarize_round(baseline_results, exploration_results, try_times=5)
    print(f"summary = {summary}")
    assert summary["exploration_yield"] == 2
    assert summary["baseline_solved"] == 3
    print("[smoke] metrics.py OK")


if __name__ == "__main__":
    _smoke()
