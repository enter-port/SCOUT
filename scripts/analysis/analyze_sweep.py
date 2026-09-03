"""Per-scene overlap analysis for exploit-guided runs vs the pure-DP baseline.

DP baseline = chain r4 rollout json (DP-SCOUT-exp3/299.ckpt on the same 100
seed-42 scenes): failed_init_indices = the 34 scenes pure DP fails.
Each run dir: all.hdf5 = 20 core demos + N explore rollouts, scene i <- demo_{20+i}.
Usage: python analyze_sweep.py RUN_DIR [RUN_DIR ...]
"""
import json
import sys

import h5py

BASE = ("/root/workspace/baojiachun/scout-entropy/data/"
        "2026_8_26_entropy/SQUARE-entropy-s233/square")
R4 = BASE + "/rollout/SCOUT-exp4/log/square_SCOUT_rollout_exp4.json"
CORE_OFFSET = 20


def dp_fail_set():
    with open(R4) as f:
        return set(json.load(f)["failed_init_indices"])


def scene_success(path):
    out = {}
    with h5py.File(path, "r") as f:
        for k in f["data"].keys():
            if not k.startswith("demo"):
                continue
            i = int(k.split("_")[1]) - CORE_OFFSET
            if i < 0:
                continue
            g = f["data/" + k]
            v = None
            for key in ("success", "done", "dones"):
                if key in g:
                    v = int(g[key][-1])
                    break
            out[i] = v
    return out


def main():
    dpf = dp_fail_set()
    dps = set(range(100)) - dpf
    print(f"DP: 66 solved / fails n={len(dpf)}")
    for run in sys.argv[1:]:
        sc = scene_success(run.rstrip("/") + "/all.hdf5")
        if not sc:
            print(f"{run}: no demos?")
            continue
        succ = set(i for i, v in sc.items() if v == 1)
        wins = sorted(dpf & succ)
        loses = sorted(dps - succ)
        # optional jerk from json
        jtxt = ""
        try:
            with open(run.rstrip("/") + "/log/square_SCOUT_rollout_exp1.json") as f:
                j = json.load(f)
            jtxt = f"  jerk={j.get('avg_jerk'):.4f}" if j.get("avg_jerk") else ""
        except Exception:
            pass
        print(f"\n{run}: {len(succ)}/{max(sc) + 1} solved{jtxt}")
        print(f"  wins  ({len(wins)}): {wins}")
        print(f"  loses ({len(loses)}): {loses}")
        net = len(wins) - len(loses)
        print(f"  net vs DP: {net:+d}  -> SR {66 + net}/100")


if __name__ == "__main__":
    main()
