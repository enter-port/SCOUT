"""Explicit eta/kappa grid on one frozen base failure set; select per-arm pass@K peak."""
import itertools
import math

from . import eval_explore
from .common import Context, finish, parser, write_json


def search(ctx, base, arms):
    cfg = ctx.c.get("grid", {})
    initial = ctx.c.get("dose", {"eta": 3., "kappa": 2.5})
    if not cfg.get("enabled", True):
        return {a: dict(initial) for a in arms}
    etas, kappas = cfg.get("etas", [initial["eta"]]), cfg.get("kappas", [initial["kappa"]])
    if not etas or not kappas or any(not math.isfinite(float(x)) or float(x) <= 0 for x in [*etas, *kappas]):
        raise ValueError("grid requires nonempty positive eta/kappa lists")
    frozen, winners, rows = None, {}, []
    for arm in arms:
        cells = [dict(eta=e, kappa=k) for e, k in itertools.product(etas, kappas)] if arm != "DP" else [dict(initial)]
        if arm == "ORBIT":
            cells = [{**d, "orbit": {"lam": lam, "sigma": sigma}} for d in cells
                     for lam, sigma in itertools.product(cfg.get("orbit_lams", [.5]), cfg.get("orbit_sigmas", [.05]))]
        arm_rows = []
        for i, dose in enumerate(cells):
            result = eval_explore.run(ctx, ctx.root / "grid" / arm / f"cell-{i:03d}",
                                      base["dp"], base["dyn"], arm, 1, dose, frozen)
            if frozen is None:
                frozen = {k: result[k] for k in ("failed", "eval")}
                frozen["artifacts"] = list(frozen.values())
            row = {"arm": arm, **dose, "pass_at_k": result.get("pass_at_k", 0),
                   "metrics": result["metrics"]}
            rows.append(row)
            arm_rows.append(row)
        # Stable ties prefer smaller eta, then smaller kappa; preserve configured orbit order.
        best = max(arm_rows, key=lambda r: (r["pass_at_k"], -r["eta"], -r["kappa"]))
        winners[arm] = {k: best[k] for k in ("eta", "kappa", "orbit") if k in best}
    if not ctx.dry:
        write_json(ctx.root / "grid/verdict.json", {"selection": "per-arm pass@K; ties: lower eta, lower kappa",
                                                   "winners": winners, "cells": rows})
    else:
        print("DRY_RUN: downstream doses are placeholders; actual values come from grid/calibration outputs")
    return winners


def main():
    p = parser(__doc__)
    p.add_argument("--dp-ckpt", required=True)
    p.add_argument("--dyn-ckpt", required=True)
    p.add_argument("--arms", nargs="+", choices=["ATY", "ORBIT", "DP"], default=["ATY", "DP"])
    args = p.parse_args()
    ctx = Context.from_args(args)
    finish(search(ctx, {"dp": str(ctx.path(args.dp_ckpt)), "dyn": str(ctx.path(args.dyn_ckpt))}, args.arms))


if __name__ == "__main__":
    main()
