"""Build the stage-B UNION training set (reflection #4 top pick).

core + per-scene capped union of rescued trajectories across runs:
  * scene identity = first env state of the demo (states[0] bytes) -- demos
    from the same rescue scene share it across runs;
  * within a scene, sort by ascending action jerk (mean |delta^2 a|) so the
    tamest (most trainable) variants are kept first;
  * keep at most --cap demos per scene (default 4) -> every rescued scene
    carries 3-4 demos instead of 1-2 (attacks the round-1 sparsity
    bottleneck) while capping the wild (high-jerk s3/k5) share.

Usage (from the worktree root):
  python soe_scripts/make_union.py <core.hdf5> <out_dir> <src1.hdf5> <src2.hdf5> ... [--cap 4]
Writes <out_dir>/train.hdf5 and prints a per-scene summary.
"""
import sys

import numpy as np


def _did(k: str) -> int:
    return int(k.split("_")[-1])


def _jerk(actions: np.ndarray) -> float:
    if len(actions) < 3:
        return 0.0
    d2 = np.abs(np.diff(actions, n=2, axis=0))
    return float(d2.mean())


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cap = 4
    for a in sys.argv[1:]:
        if a.startswith("--cap"):
            cap = int(a.split("=", 1)[1]) if "=" in a else int(sys.argv[sys.argv.index(a) + 1])
    core_path, out_dir, *srcs = args
    out_path = f"{out_dir.rstrip('/')}/train.hdf5"

    import h5py
    import shutil

    with h5py.File(core_path, "r") as f:
        n_core = max(_did(k) for k in f["data"].keys()
                     if k.startswith("demo_") and k.split("_")[-1].isdigit()) + 1

    # pool every appended demo across sources, grouped by scene key
    scenes = {}          # key -> list of (jerk, src_idx, demo_name)
    src_files = [h5py.File(s, "r") for s in srcs]
    try:
        for si, rf in enumerate(src_files):
            for k in rf["data"].keys():
                if not (k.startswith("demo_") and k.split("_")[-1].isdigit()):
                    continue
                if _did(k) < n_core:
                    continue
                st = rf[f"data/{k}/states"][0]
                key = np.ascontiguousarray(st).tobytes()
                jerk = _jerk(rf[f"data/{k}/actions"][:])
                scenes.setdefault(key, []).append((jerk, si, k))
    finally:
        pass  # files stay open for the copy phase below

    selection = []
    for key, lst in scenes.items():
        lst.sort(key=lambda t: (t[0], t[1]))
        selection.extend(lst[:cap])
    # stable final order: scene discovery order, tame-first (already sorted)
    print(f"[make_union] scenes={len(scenes)} pooled={sum(len(v) for v in scenes.values())} "
          f"selected={len(selection)} (cap={cap}/scene)")

    shutil.copyfile(core_path, out_path)
    with h5py.File(out_path, "r+") as f:
        next_id = n_core
        copied = 0
        for jerk, si, k in selection:
            rf = src_files[si]
            rf.copy(f"data/{k}", f["data"], name=f"demo_{next_id}")
            next_id += 1
            copied += 1
        all_demos = sorted(k for k in f["data"].keys() if k.startswith("demo"))
        core_set = {k for k in f["data"].keys() if k.startswith("demo")}
        mask = np.array([_did(d) < n_core if d.split("_")[-1].isdigit() else True
                         for d in all_demos], dtype=bool)
        if "mask/scout_aug" in f:
            del f["mask/scout_aug"]
        g = f.create_group("mask/scout_aug")
        g.create_dataset("mask", data=mask)
        g.attrs["num"] = int(mask.sum())
        f.attrs["num_demos_added"] = copied
        print(f"[make_union] wrote {out_path}: core={n_core} rescued={copied} total={len(all_demos)}")
    for rf in src_files:
        rf.close()


if __name__ == "__main__":
    main()
