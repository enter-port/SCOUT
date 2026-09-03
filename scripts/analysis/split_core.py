"""Seeded core split (2026-08-21 user): reproducibly pick N demos from the
official robomimic dataset with ONE seed -- the same TSEED that controls all
training randomness (round.sh v3 round 0).

Selection:
    rng = np.random.default_rng(seed)
    ids = sorted(rng.choice(n_total, n, replace=False).tolist())

Output format is byte-compatible with experiments/scripts/extract_core_subset.py
(copy every top-level group except ``data``; copy ``data`` attrs; renumber the
kept demos demo_0..demo_{n-1}; refresh the ``total`` attr) so every downstream
consumer (LPB RobomimicReplayImageDataset, hdf5_writer merges, VIB zarr
conversion) treats it exactly like the old core files.

Usage: split_core.py <src.hdf5> <dst.hdf5> <n> <seed>
"""
import sys

import h5py
import numpy as np

src, dst, n, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
with h5py.File(src, "r") as fin:
    demos = sorted(fin["data"].keys(), key=lambda k: int(k.split("_")[-1]))
    if not 0 < n < len(demos):
        raise SystemExit(f"need 0 < n < {len(demos)} demos; got {n}")
    ids = sorted(int(i) for i in
                 np.random.default_rng(seed).choice(len(demos), n, replace=False))
    keep = [demos[i] for i in ids]
    total = int(sum(fin[f"data/{d}/actions"].shape[0] for d in keep))
    with h5py.File(dst, "w") as fout:
        for top in fin.keys():
            if top != "data":
                fin.copy(top, fout)
        g = fout.create_group("data")
        for k, v in fin["data"].attrs.items():
            g.attrs[k] = v
        for ni, d in enumerate(keep):
            fin.copy(f"data/{d}", g, name=f"demo_{ni}")
        g.attrs["total"] = total
print(f"OK wrote {dst}: seed={seed} picked source demo ids {ids} "
      f"-> {n} demos, total={total} steps")
