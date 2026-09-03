import hashlib
import itertools

import h5py
import numpy as np


def fp_set(path):
    s = set()
    with h5py.File(path, "r") as f:
        for name in f["data"].keys():
            if not str(name).startswith("demo"):
                continue
            st = f["data"][name]["states"][0]
            s.add(hashlib.md5(np.asarray(st).tobytes()).hexdigest())
    return s


E = "/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy/SQUARE-entropy-s233/square"
core = fp_set(E + "/rollout/square_core.hdf5")
print(f"core fingerprints: {len(core)}")
arms = {
    "atypical_r1": E + "/rollout/SCOUT-exp1/success.hdf5",
    "part_G1": "data/particle/sq_G1_ps0/square_SCOUT_success_exp0.hdf5",
    "part_G2": "data/particle/sq_G2_ps50/square_SCOUT_success_exp0.hdf5",
    "orb_s025": "data/particle/sq_orb_s025/square_SCOUT_success_exp0.hdf5",
    "orb_s050": "data/particle/sq_orb_s050/square_SCOUT_success_exp0.hdf5",
}
S = {k: fp_set(v) - core for k, v in arms.items()}
for k, v in S.items():
    print(f"{k}: {len(v)} rescued scenes (post-core-exclusion)")
print()
for a in S:
    for b in S:
        if a < b:
            print(f"{a} vs {b}: common {len(S[a] & S[b])}"
                  f"  only-{a} {len(S[a] - S[b])}  only-{b} {len(S[b] - S[a])}")
for combo in itertools.combinations(S, 3):
    u = set().union(*[S[c] for c in combo])
    print(f"union {combo} = {len(u)}")
allu = set().union(*S.values())
print(f"union ALL = {len(allu)}")
