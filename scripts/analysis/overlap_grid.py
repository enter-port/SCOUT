import hashlib

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


R26 = "/root/workspace/baojiachun/scout-entropy/data/2026_8_26_entropy"
# (seed, round) -> chain exp index for the atypical comparison (r1->exp1 ...)
CELLS = [
    ("233", 1), ("233", 2),          # r3 killed
    ("2333", 1), ("2333", 2),        # r3 killed
    ("23333", 1), ("23333", 2), ("23333", 3),
]
HIST_ATYP = {"233": [30, 21, 25], "2333": [17, 17, 20], "23333": [33, 32, 35]}
ORB = {"233": [36, 13, None], "2333": [24, 19, None], "23333": [28, 15, 26]}

for seed, r in CELLS:
    E = f"{R26}/SQUARE-entropy-s{seed}/square"
    core = fp_set(E + "/rollout/square_core.hdf5")
    cell_dir = "sq_orb_s025" if (seed, r) == ("233", 1) else f"sq_orb025_s{seed}_r{r}"
    orb = fp_set(f"data/particle/{cell_dir}/square_SCOUT_success_exp0.hdf5") - core
    aty = fp_set(E + f"/rollout/SCOUT-exp{r}/success.hdf5") - core
    print(f"s{seed} r{r}: orbit {len(orb)} (json {ORB[seed][r-1]})  "
          f"atypical {len(aty)} (json {HIST_ATYP[seed][r-1]})")
    print(f"    common {len(orb & aty)}  only-orbit {len(orb - aty)}  "
          f"only-atypical {len(aty - orb)}  "
          f"orbit keeps {len(orb & aty)}/{len(aty)} of atypical ridge")
