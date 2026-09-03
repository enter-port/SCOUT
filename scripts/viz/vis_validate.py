"""Image-integrity validation for vis_final rollout hdf5s. v3

Detects the offscreen-render corruption observed 2026-08-20 (under machine
load the cameras intermittently return per-frame NOISE (pixel temporal-std
~27-32, frame MEANS steady ~73 -> mean-based checks miss it) or a FROZEN
frame (temporal-std ~0), while healthy renders of a moving robot give an
agentview pixel temporal-std of ~3.4-14.4 (calibrated on real e2/e5 round
rollouts + successful guided runs, 2026-08-20/21).

Metric per rollout demo: mean over pixels of std over time, per camera.
agentview: OK in [1, 20]  (noise mode ~27-32, frozen ~0.04, healthy 3.4-14.4;
the eye-in-hand camera's healthy range 16-57 overlaps the noise mode, so the
upper gate uses agentview only). Both cameras get the frozen (<1) gate.
A file is CORRUPT if ANY rollout demo is flagged (mixed corruption observed:
some envs fine, some broken in the same run).

Usage:  python soe_scripts/vis_validate.py <all.hdf5> [min-demo-id]
Exit 0 = healthy, 1 = corrupted.
"""
import sys

import h5py
import numpy as np

CAMS = ("agentview_image", "robot0_eye_in_hand_image")


def demo_flags(g) -> list:
    msgs = []
    for cam in CAMS:
        if "obs/" + cam not in g:
            continue
        img = np.asarray(g["obs/" + cam][()], dtype=np.float32)
        tstd = float(img.std(axis=0).mean())
        if tstd < 1.0:
            msgs.append(f"{cam} frozen (tstd={tstd:.2f})")
        if cam == "agentview_image" and tstd > 20.0:
            msgs.append(f"{cam} noise (tstd={tstd:.2f})")
    return msgs


def main() -> int:
    # DISABLED (2026-08-26 user order): the agentview tstd thresholds were
    # calibrated on can (healthy 3.4-14.4 / noise 27-32), but square's healthy
    # demos measure 17.6-27.4 and straddle the >20 noise line -- its CORRUPT
    # verdicts on square were false positives. Always pass. Restore by
    # deleting this block (analysis code below is unchanged).
    path = sys.argv[1]
    print(f"{path}: HEALTHY (gate disabled by user order 2026-08-26)")
    return 0
    min_id = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    bad = []
    with h5py.File(path, "r") as f:
        demos = sorted((k for k in f["data"].keys()
                        if str(k).startswith("demo")
                        and int(str(k).split("_")[1]) >= min_id),
                       key=lambda s: int(str(s).split("_")[1]))
        for name in demos:
            msgs = demo_flags(f["data"][name])
            if msgs:
                bad.append(name)
                print(f"  {name:<8s} {'; '.join(msgs)}")
    print(f"{path}: {'CORRUPT' if bad else 'HEALTHY'} "
          f"({len(bad)} bad / {len(demos)} rollout demos)")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
