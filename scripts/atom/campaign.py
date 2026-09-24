"""Config-driven SCOUT campaign runner.

The runner expands setup stages and typed four-step rounds into the existing
atom functions.  Calibration results are runtime inputs to eval+explore; a
campaign config can choose the calibration algorithm and its targets, but it
cannot duplicate the resulting eta/kappa values.
"""
from __future__ import annotations

from pathlib import Path

from . import base_train, calib, dp_train, dyn_train, eval_explore, grid_search
from .common import Context, finish, lock, parser, write_json


def _round_spec(ctx, number):
    """Resolve the explicit round policy for one round."""
    plan = ctx.c.get("round_plan")
    if not plan:
        return {"calib": dict(ctx.c.get("calib", {})), "train": number < int(ctx.c.get("rounds", 6))}
    for entry in plan:
        numbers = entry.get("rounds", [])
        if isinstance(numbers, int):
            numbers = [numbers]
        if number in numbers:
            spec = dict(entry)
            spec.setdefault("calib", {})
            spec.setdefault("train", number < int(ctx.c.get("rounds", 6)))
            return spec
    raise ValueError(f"round {number} is missing from round_plan")


def run(ctx):
    arms = ctx.c.get("arms", ["ATY"])
    rounds = int(ctx.c.get("rounds", 6))
    if rounds < 1 or not arms or len(set(arms)) != len(arms) or any(
            arm not in {"ATY", "DP", "ORBIT"} for arm in arms):
        raise ValueError("require positive rounds and unique arms from ATY, DP, ORBIT")

    base = base_train.prepare(ctx)
    doses = grid_search.search(ctx, base, arms)

    # RC uses a fixed base reference.  P6 and DP-KL are fully current-pair
    # methods and deliberately do not call this compatibility branch.
    reference = dict(base)
    if any(_round_spec(ctx, n).get("calib", {}).get("mode", "rc") == "rc"
           for n in range(1, rounds + 1)) and "ATY" in arms:
        initial = doses["ATY"]
        reference.update(calib.calibrate(ctx, ctx.root / "base/calib", base["dp"], base["dyn"],
                                         initial, mode="r", options=ctx.c.get("calib", {})))
        if not ctx.dry:
            write_json(ctx.root / "base/reference.json", reference)

    results = []
    for arm in arms:
        pair = dict(base)
        dose = dict(doses[arm])
        successes, trajectories = [], []
        for number in range(1, rounds + 1):
            spec = _round_spec(ctx, number)
            rdir = ctx.root / "rounds" / arm / f"round-{number}"
            calib_spec = dict(ctx.c.get("calib", {}))
            calib_spec.update(spec.get("calib", {}))
            mode = calib_spec.pop("mode", "none")

            # A round's dose is produced here and passed directly to the
            # rollout atom.  Round 6 can set mode=none and therefore carries
            # round 5's measured dose without a second calibration.
            if mode != "none":
                dose = calib.calibrate(ctx, rdir / "calib", pair["dp"], pair["dyn"], dose,
                                       reference if mode == "rc" else None,
                                       mode=mode, options=calib_spec)

            rollout_options = dict(ctx.c.get("rollout", {}))
            rollout_options.update(spec.get("rollout", {}))
            result = eval_explore.run(ctx, rdir / "rollout", pair["dp"], pair["dyn"],
                                     arm, number, dose, options=rollout_options)
            results.append({"arm": arm, "round": number, "calib_mode": mode,
                            "dose": dose, **result})
            if result.get("success"):
                successes.append(result["success"])
            if result.get("all"):
                trajectories.append(result["all"])

            if not spec.get("train", number < rounds):
                continue

            resume_id = result.get("wandb_run_id")
            dp_options = spec.get("dp", {})
            dp_result = dp_train.train(ctx, rdir / "dp", successes,
                                       f"{arm}-round{number}-DP", options=dp_options,
                                       resume_run_id=resume_id)
            pair["dp"] = dp_result["dp"]
            dyn_options = spec.get("dyn", {})
            if arm != "DP" and number <= int(ctx.c.get("dyn_freeze_after", rounds)):
                dyn_result = dyn_train.train(ctx, rdir / "dyn", pair["dp"], trajectories,
                                             f"{arm}-round{number}-dyn", options=dyn_options,
                                             resume_run_id=resume_id)
                pair["dyn"] = dyn_result["dyn"]

    if not ctx.dry:
        write_json(ctx.root / "summary.json", results)
    return results


def main():
    args = parser(__doc__).parse_args()
    ctx = Context.from_args(args)
    if ctx.dry:
        finish(run(ctx))
    else:
        ctx.root.mkdir(parents=True, exist_ok=True)
        with lock(ctx.root / ".chain.lock"):
            finish(run(ctx))


if __name__ == "__main__":
    main()
