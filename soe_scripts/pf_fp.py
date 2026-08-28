# -*- coding: utf-8 -*-
"""pf_fp.py -- rescued-set fingerprint comparison for portfolio arms.

Strict guardrail (user order 08-28): an arm must rescue EVERY baseline
scene of the entropy-cost baseline (base3_screen, scenes {3,9,12,18});
losing any one of them = fail.  Placebo tier {9,18} is NOT a passing bar.

Method: rescue-mode success.hdf5 stores the SUCCESSFUL retries (plus the
core 20 demos); a scene's identity = md5 of its stored states[0] (demo fps
in all.hdf5 group order == failed_init_indices order).  We collect the
arm's rescued-scene fingerprints and check the baseline's four are all
present; extras = new scenes.
"""
import glob
import hashlib
import json
import os
import sys

import h5py
import numpy as np

CORE_N = 20
BASE = "/root/workspace/baojiachun/scout-rand/data/rand/base3_screen"


def fps_of(path):
    out = []
    with h5py.File(path, "r") as f:
        ids = sorted(int(k.split("_")[-1]) for k in f["data"]
                     if k.startswith("demo_") and k.split("_")[-1].isdigit())
        for i in ids:
            if i < CORE_N:
                continue
            g = f[f"data/demo_{i}"]
            fp = hashlib.md5(np.asarray(g["states"][0],
                                        dtype=np.float64).tobytes()).hexdigest()[:12]
            out.append((i, fp, bool(g["success"][0]) if "success" in g else None))
    return out


def scene_map(run_dir):
    """fingerprint -> init_idx via all.hdf5 group order + failed list."""
    jpath = sorted(glob.glob(os.path.join(run_dir, "log", "*.json")))
    j = json.load(open(jpath[0])) if jpath else {}
    failed = j.get("failed_init_indices", [])
    seen, order = set(), []
    with h5py.File(os.path.join(run_dir, "all.hdf5"), "r") as f:
        ids = sorted(int(k.split("_")[-1]) for k in f["data"]
                     if k.startswith("demo_") and k.split("_")[-1].isdigit())
        for i in ids:
            if i < CORE_N:
                continue
            g = f[f"data/demo_{i}"]
            fp = hashlib.md5(np.asarray(g["states"][0],
                                        dtype=np.float64).tobytes()).hexdigest()[:12]
            if fp not in seen:
                seen.add(fp)
                order.append(fp)
    return {fp: (failed[k] if k < len(failed) else None)
            for k, fp in enumerate(order)}, j


def main():
    base_fps = {}
    for i, fp, _ in fps_of(os.path.join(BASE, "success.hdf5")):
        base_fps.setdefault(fp, None)
    bm, bj = scene_map(BASE)
    base_scenes = {bm[fp]: fp for fp in base_fps if fp in bm}
    print(f"baseline base3_screen: rescued {bj.get('exploration_rescued')}"
          f"/{bj.get('n_failed')} pass@10 {bj.get('pass_at_5')}"
          f" failed={bj.get('failed_init_indices')}")
    print(f"  baseline rescued scenes: "
          + ", ".join(f"{s}->{fp[:8]}" for s, fp in sorted(base_scenes.items())))
    for tag in sys.argv[1:]:
        d = os.path.join("/root/workspace/baojiachun/scout-rand/data/rand", tag)
        sm, j = scene_map(d)
        arm_fps = {fp for _, fp, _ in fps_of(os.path.join(d, "success.hdf5"))}
        arm_scenes = sorted(sm[fp] for fp in arm_fps if fp in sm)
        missing = {s: fp for s, fp in base_scenes.items() if fp not in arm_fps}
        extra = [s for s in arm_scenes if s not in base_scenes.values()]
        print(f"\n{tag}: rescued {j.get('exploration_rescued')}"
              f"/{j.get('n_failed')} pass@10 {j.get('pass_at_5')}"
              f" jerk {j.get('avg_jerk'):.3f} mean_inject? (stdout)"
              f" failed={j.get('failed_init_indices')}")
        print(f"  arm rescued scenes: {arm_scenes}")
        verdict = "PASS" if not missing else "FAIL"
        print(f"  guardrail(strict): {verdict}"
              + (f"  LOST={{{','.join(str(s) for s in sorted(missing))}}}"
                 if missing else "  (all 4 baseline scenes kept)")
              + (f"  NEW={{{','.join(map(str, sorted(extra)))}}}" if extra else ""))


if __name__ == "__main__":
    main()
