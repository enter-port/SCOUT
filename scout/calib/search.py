"""Bounded scalar calibration with measured results and explicit convergence."""
import math


def solve_kappa(measure, target, initial, *, band=.1, max_probes=12,
                lower=1e-3, upper=100.0, direction="increasing"):
    """Bracket a response, then bisect in log(kappa).

    Only evaluated values can be returned. A flat response, a discontinuity,
    or a depleted probe budget returns the closest measurement with
    converged=False. A bracket can tolerate local nonmonotonicity; no global
    monotonicity or existence of a solution is asserted by this routine.
    direction="unknown" explores both sides of the initial cap when there
    is no bracket. This supports decreasing responses and local extrema.
    """
    if not all(math.isfinite(x) for x in [target, initial, band, lower, upper]):
        raise ValueError("Calibration inputs must be finite")
    if not (target > 0 and 0 <= band < 1 and 0 < lower <= initial <= upper):
        raise ValueError("Invalid target, band, or kappa bounds")
    if not isinstance(max_probes, int) or max_probes < 1:
        raise ValueError("max_probes must be a positive integer")
    if direction not in ("increasing", "unknown"):
        raise ValueError("direction must be increasing or unknown")
    history = []
    kappa = initial
    reason = "probe_budget"
    for _ in range(max_probes):
        value = float(measure(kappa))
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid response at kappa={kappa}: {value}")
        history.append({"kappa": kappa, "C_mean": value})
        if abs(value - target) <= band * target:
            reason = "target_band"
            break
        points = sorted(history, key=lambda item: item["kappa"])
        brackets = [(a, b) for a, b in zip(points, points[1:])
                    if (a["C_mean"] - target) * (b["C_mean"] - target) < 0]
        if brackets:
            a, b = min(brackets, key=lambda pair: math.log(pair[1]["kappa"] / pair[0]["kappa"]))
            next_kappa = math.exp(.5 * (math.log(a["kappa"]) + math.log(b["kappa"])))
        elif direction == "unknown":
            candidates = [max(lower, points[0]["kappa"] / 2),
                          min(upper, points[-1]["kappa"] * 2)]
            if len(history) % 2 == 0:
                candidates.reverse()
            unused = [candidate for candidate in candidates
                      if not any(math.isclose(candidate, p["kappa"], rel_tol=1e-12) for p in history)]
            if not unused:
                reason = "bound_or_resolution"
                break
            next_kappa = unused[0]
        elif value < target:
            next_kappa = min(upper, points[-1]["kappa"] * 2)
        else:
            next_kappa = max(lower, points[0]["kappa"] / 2)
        if any(math.isclose(next_kappa, item["kappa"], rel_tol=1e-12) for item in history):
            reason = "bound_or_resolution"
            break
        kappa = next_kappa
    selected = min(range(len(history)), key=lambda i: abs(history[i]["C_mean"] - target))
    best = history[selected]
    return {"kappa": best["kappa"], "C_mean": best["C_mean"], "history": history,
            "selected_index": selected, "converged": abs(best["C_mean"] - target) <= band * target,
            "termination_reason": reason}
