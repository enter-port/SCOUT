"""entropy-random-dev idea registry (user long-term campaign, 2026-08-27).

Goal: inject SOE-like randomness INTO the guidance cost.  Stop condition =
pass@10 (pure rescue on base DP+dyn, scenes 42-141, NO data write-back)
> 0.85.  Current best = 方案三 atypical, pass@10 0.76-0.78 (rescued 14/19).

CONFLICT-MANAGEMENT CONTRACT (parallel subagents, do not violate):
  * each idea = ONE new module file in this package; NEVER edit shared files
    (entropy_costs.py / rollout_vec.py / rollout_pipeline.py / run_rollout.py
    are owned by the orchestrator);
  * the module must define
        NAME: str                                            # token
        def make_planner(scout_vib, bridge=None, obs_adapter=None, ek: dict)
    and is auto-discovered here via pkgutil -- creating the file is the ONLY
    step needed to make ``--guide rand_<NAME>`` work;
  * ek = run_rollout's entropy_kwargs dict (CLI params land there; add your
    own CLI flags in run_rollout only via the orchestrator, or read scalar
    knobs from ek strings);
  * module-level imports must stay torch/numpy-only (no robomimic/env deps);
    lazy-import anything heavy inside functions;
  * randomness MUST be deterministic given (shell_seed, init_idx, try_idx)
    (seed from ek["shell_seed"]-style kwargs); use the hooks:
        select_z(x0_hat, current_obs)     # per chunk, first guided step --
                                          # capture the intent baseline here
        set_row_jobs(jobs)                # per replan batch, jobs =
                                          # [(state, init_idx, try_idx), ...]
        compute_loss(x0_hat, current_obs, reduction="mean"|"sum")
    and respect ``reduction`` exactly (guided path uses "sum").

Reference result to beat (all on can base DP 599.ckpt + dyn-base, seed42
x100 scenes, rescue x10, env50): 方案三 rescued 19 / pass@10 0.78 /
mean_inject ~1.0.  The failed 方案A (kappa-shell random Gaussian target)
is documented in entropy_costs.py:ShellTargetCostPlanner -- its diagnosis:
uniform random directions are projected back onto the low-rank reachable
manifold of the encoder Jacobian; randomize in a space whose direction
spectrum you understand.
"""
import importlib
import pkgutil

REGISTRY = {}


def _discover():
    REGISTRY.clear()
    for m in pkgutil.iter_modules(__path__):
        if m.name.startswith("_"):
            continue
        mod = importlib.import_module(f"{__name__}.{m.name}")
        name = getattr(mod, "NAME", None)
        maker = getattr(mod, "make_planner", None)
        if name and callable(maker):
            REGISTRY[str(name)] = maker


_discover()
