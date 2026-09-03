#!/usr/bin/env python
"""Offline mixing simulation for the SQUARE beat-SOE campaign (2026-08-31).

Reads wave-1 arm jsons (explore_detail: per-failed-init solved + 1-based
first_success_try), estimates per-arm rescue curves, and searches budget
splits (n_arm1 + n_arm2 + ... = 10) for the best pooled rescue count.

Model: tries within an arm are exchangeable -> a scene is rescued by an arm
given k tries iff its observed first_success_try <= k (one realization;
optimistic estimator -- confirmation run with a fresh explore seed is the
gate, not this number).

Usage: python sq2_mix_sim.py <json1> <json2> [...] [--target N]
  (arms are named by their filename stem)
"""
import json
import sys
from itertools import product

TARGETS = {"233": 40, "2333": 43, "23333": 42}  # rescued scenes needed


def load_arm(path):
    name = path.split("/")[-1].replace(".json", "")
    d = json.load(open(path))
    det = d.get("explore_detail")
    if not det:
        raise SystemExit(f"{path}: no explore_detail (pre-2026-08-31 json?)")
    # init -> first-success try (None if never solved in 10 tries)
    return name, {e["init"]: (e["first_success_try"] if e["solved"] else None)
                  for e in det}


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    target = None
    for a in sys.argv[1:]:
        if a.startswith("--target"):
            target = int(a.split("=")[1])
    arms = dict(load_arm(p) for p in args)
    seeds = {s for arm in arms.values() for s in arm}
    # sanity: all arms cover the same failed-init pool
    for n, arm in arms.items():
        if set(arm) != seeds:
            print(f"WARNING {n}: pool mismatch ({len(arm)} vs {len(seeds)})")
    for s in TARGETS:
        if str(s) in "-".join(arms):
            target = target or TARGETS[s]

    print(f"pool = {len(seeds)} failed inits; target = {target}")
    print("\n== per-arm rescue curves (cumulative scenes rescued by try k) ==")
    for n, arm in arms.items():
        row = [sum(1 for v in arm.values() if v is not None and v <= k)
               for k in range(1, 11)]
        print(f"  {n:24s} " + " ".join(f"{r:3d}" for r in row))

    names = list(arms)
    best = []
    # search integer splits (order matters only per-arm; arms fixed set)
    grids = [range(0, 11)] * len(names)
    for combo in product(*grids):
        if sum(combo) != 10:
            continue
        rescued = 0
        for scene in seeds:
            ok = any(arms[n][scene] is not None and arms[n][scene] <= k
                     for n, k in zip(names, combo))
            rescued += ok
        best.append((rescued, dict(zip(names, combo))))
    best.sort(key=lambda x: -x[0])
    seen_splits = set()
    print(f"\n== best splits (top 10, dedup by split) ==")
    for r, sp in best:
        key = tuple(sorted(sp.items()))
        if key in seen_splits:
            continue
        seen_splits.add(key)
        flag = "  <-- MEETS TARGET" if target and r >= target else ""
        print(f"  {r:3d}/{len(seeds)}  {sp}{flag}")
        if len(seen_splits) >= 10:
            break
    # reference: each arm alone at 10 tries
    print(f"\n== arms alone (k=10) ==")
    for n, arm in arms.items():
        print(f"  {n:24s} {sum(1 for v in arm.values() if v is not None)}")


if __name__ == "__main__":
    main()
