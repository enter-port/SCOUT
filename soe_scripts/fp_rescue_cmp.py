# -*- coding: utf-8 -*-
"""fp_rescue_cmp.py -- rescued-set fingerprint comparison across rescue runs.

Usage (server):
  python /tmp/fp_rescue_cmp.py <run_dir1> <run_dir2> [...]

Prints, per run: per-scene groups (init_idx, fp, n_trajs, n_succ) sorted by
init_idx, rescued count; then pairwise set comparisons (same / subset /
superset / diff with per-scene success counts).  Caliber = rescue storage:
all.hdf5 = 20 core demos + per failed scene (successful retries if any,
else retry-0 only); group order follows the json's failed_init_indices.
"""
import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np

CORE_N = 20


def fp_of(arr):
    return hashlib.md5(np.asarray(arr, dtype=np.float64).tobytes()).hexdigest()[:12]


def load_run(run_dir):
    jpath = sorted(glob.glob(os.path.join(run_dir, "log", "*.json")))
    j = json.load(open(jpath[0])) if jpath else {}
    failed = j.get("failed_init_indices", [])
    rows = []
    with h5py.File(os.path.join(run_dir, "all.hdf5"), "r") as f:
        ids = sorted(int(k.split("_")[-1]) for k in f["data"]
                     if k.startswith("demo_") and k.split("_")[-1].isdigit())
        for i in ids:
            if i < CORE_N:
                continue
            g = f[f"data/demo_{i}"]
            succ = bool(g["success"][0]) if "success" in g else None
            rows.append((fp_of(g["states"][0]), succ))
    # group by consecutive fingerprint runs (storage writes scene blocks in order)
    groups, cur = [], None
    for fp, succ in rows:
        if fp != cur:
            groups.append({"fp": fp, "succ": []})
            cur = fp
        groups[-1]["succ"].append(succ)
    # map group -> init idx via failed_init_indices order
    scenes = {}
    for k, grp in enumerate(groups):
        init = failed[k] if k < len(failed) else None
        n_succ = sum(1 for s in grp["succ"] if s)
        scenes[init] = {"fp": grp["fp"], "n_tries": len(grp["succ"]),
                        "n_succ": n_succ, "rescued": n_succ > 0}
    return j, scenes


def main():
    runs = []
    for d in sys.argv[1:]:
        j, scenes = load_run(d)
        rescued = {k: v for k, v in scenes.items() if v["rescued"]}
        runs.append((d, j, scenes, rescued))
        print(f"\n=== {os.path.basename(d.rstrip('/'))} "
              f"(={d})")
        print(f"    baseline_solved {j.get('baseline_solved')} "
              f"n_failed {j.get('n_failed')} rescued {len(rescued)} "
              f"pass@{j.get('explore_try_times')} {j.get('pass_at_5')} "
              f"jerk {j.get('avg_jerk'):.4f}")
        print(f"    failed_init_indices ({len(j.get('failed_init_indices', []))}): "
              f"{j.get('failed_init_indices')}")
        for init in sorted(scenes, key=lambda x: (x is None, x)):
            v = scenes[init]
            mark = "R" if v["rescued"] else "."
            print(f"    [{mark}] init {init:>3} fp {v['fp']} "
                  f"succ {v['n_succ']}/{v['n_tries']}")
    # pairwise comparisons
    for a in range(len(runs)):
        for b in range(a + 1, len(runs)):
            da, ja, _, ra = runs[a]
            db, jb, _, rb = runs[b]
            print(f"\n### compare {os.path.basename(da.rstrip('/'))} vs "
                  f"{os.path.basename(db.rstrip('/'))}")
            fa = {v["fp"]: (k, v) for k, v in ra.items()}
            fb = {v["fp"]: (k, v) for k, v in rb.items()}
            only_a = set(fa) - set(fb)
            only_b = set(fb) - set(fa)
            both = set(fa) & set(fb)
            print(f"    rescued: {len(ra)} vs {len(rb)}; shared {len(both)}; "
                  f"lost_vs_B {len(only_a)}; new_vs_B {len(only_b)}")
            for fp in sorted(only_a):
                k, v = fa[fp]
                print(f"    LOST (in A not B): init {k} fp {fp} "
                      f"A succ {v['n_succ']}/{v['n_tries']}")
            for fp in sorted(only_b):
                k, v = fb[fp]
                print(f"    NEW  (in B not A): init {k} fp {fp} "
                      f"B succ {v['n_succ']}/{v['n_tries']}")
            if only_a == only_b == set():
                print("    -> IDENTICAL rescued fingerprint sets")
            elif only_a == set():
                print("    -> A rescued set is SUBSET of B")
            elif only_b == set():
                print("    -> A rescued set is SUPERSET of B")


if __name__ == "__main__":
    main()
