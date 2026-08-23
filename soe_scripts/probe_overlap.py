"""Per-scene overlap of explore successes between the DP and SCOUT arms.

Both arms explore the SAME fresh scene set each round (seed i*1000+42), so a
rollout demo can be attributed to its scene via the initial state vector
(states[0], rounded to kill float noise). For each seed x round:
  both  = scenes solved by BOTH arms
  onlyD = scenes solved only by the plain-DP arm
  onlyS = scenes solved only by the SCOUT arm
If exploration were additively useful, onlyS > onlyD systematically.
"""
import glob
import h5py
import numpy as np

R = "/root/workspace/baojiachun/scout/data/2026_8_21"
seeds = ["233", "2333", "23333", "233333"]

def scenes(path):
    """set of rounded initial-state tuples of rollout demos in success.hdf5."""
    out = set()
    with h5py.File(path, "r") as f:
        n_core = None
        # rollout demos are appended after the core demos; identify them by the
        # scout_aug mask if present, else assume demos after the core count
        # (core count differs per seed; simpler: use ALL demos' first states --
        # core demos are identical across arms, so they land in `both` and
        # cancel out of onlyD/onlyS).
        for k in f["data"].keys():
            if not str(k).startswith("demo"):
                continue
            st = f[f"data/{k}/states"][0]
            out.add(tuple(np.round(np.asarray(st, dtype=np.float64), 4)))
    return out

print("seed   round |  both  onlyDP  onlySCOUT   (scene sets matched by init state)")
tot = {"both": 0, "onlyD": 0, "onlyS": 0}
for S in seeds:
    for N in range(1, 7):
        pd = f"{R}/CAN-exp1-{S}/can/rollout/DP-exp{N}/success.hdf5"
        ps = f"{R}/CAN-exp1-{S}/can/rollout/SCOUT-exp{N}/success.hdf5"
        if not (glob.glob(pd) and glob.glob(ps)):
            continue
        d, s = scenes(pd), scenes(ps)
        both, onlyD, onlyS = len(d & s), len(d - s), len(s - d)
        tot["both"] += both; tot["onlyD"] += onlyD; tot["onlyS"] += onlyS
        print(f"s{S:>6} r{N} | {both:5d} {onlyD:6d} {onlyS:10d}")
print("TOTALS:", tot)
