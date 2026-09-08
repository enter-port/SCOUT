"""Build the SOE-format tool_hang core from SCOUT's own 40-demo core.

Why this exists (2026-09-07): SOE's ``convert_rel_actions_to_abs`` and
SCOUT's converter share lineage but diverge on each demo's FIRST action
(robosuite controller goal-sync after ``reset_to`` differs between the two
setups; measured on tool_hang AND on the historical can datasets: pos
deviation <= ~5cm at step 0 only, mid-demo identical, ~0.15% of steps >1cm).
The can/square SOE baselines trained on the replay-converted values and
behaved normally, so this is a provenance nuance, not a bug -- but the user
requirement for the tool_hang SOE baseline is "initial 40 demos identical to
TOOLHANG-9-5-orbit-s233", so we make the core VALUE-IDENTICAL to SCOUT's by
converting SCOUT's ``abs_actions`` (pos3|rotvec3|grip1) to SOE's 10-dim
(pos3|rot6d|grip1) with pure math -- no replay, no controller involvement.

Selection identity (same 40 demos) is separately proven by make_core_soe.py's
``--scout-core`` cross-check (demo count / per-demo lengths / total steps),
which must have run and PASSED before this script is used.

Layout mirrors make_core_soe.extract_core: copy every top-level group except
``data``/``mask`` (SCOUT's masks index a different selection), rebuild
``data`` with ``actions`` replaced by the 10-dim form (SCOUT's 7-dim
``abs_actions``/``actions`` keys ride along untouched -- SOE reads
output_keys=["actions"] only), and write a fresh ``mask/core_20`` with
robomimic's own writer. Atomic: writes ``<out>.tmp.hdf5`` then os.replace.
"""
import argparse
import os
import sys

import h5py
import numpy as np

sys.path.insert(0, "/root/workspace/baojiachun/SOE/simulation")
from rotation_transformer import RotationTransformer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scout-core", required=True,
                   help="SCOUT 40-demo core hdf5 (abs_actions 7-dim)")
    p.add_argument("--out", required=True,
                   help="SOE core hdf5 to write (actions 10-dim)")
    p.add_argument("--n", type=int, default=40)
    a = p.parse_args()

    rt = RotationTransformer(from_rep="axis_angle", to_rep="rotation_6d")
    tmp = a.out + ".tmp.hdf5"
    if os.path.exists(tmp):
        os.remove(tmp)

    total = 0
    with h5py.File(a.scout_core, "r") as fin, h5py.File(tmp, "w") as fout:
        for k in fin:
            if k in ("data", "mask"):
                continue
            fin.copy(k, fout)
        din, dout = fin["data"], fout.create_group("data")
        for k in din.attrs:
            dout.attrs[k] = din.attrs[k]
        keys = sorted(din.keys(), key=lambda s: int(s.split("_")[1]))
        assert len(keys) == a.n, "expected %d demos, got %d" % (a.n, len(keys))
        for k in keys:
            src = din[k]
            g = fout.create_group("data/" + k)
            for ak, av in src.attrs.items():
                g.attrs[ak] = av  # robomimic attrs (num_samples, camera_info, ...)
            for name in src:
                if name == "actions":
                    continue  # replaced by the 10-dim build below
                src.copy(name, g)
            aa = src["abs_actions"][:]  # (T, 7) pos3|rotvec3|grip1
            act = np.concatenate(
                [aa[:, :3], rt.forward(aa[:, 3:6]), aa[:, 6:7]], axis=-1)
            g.create_dataset("actions", data=act.astype(np.float64))
            g.attrs["num_samples"] = act.shape[0]  # create_hdf5_filter_key needs it
            total += act.shape[0]
        dout.attrs["total"] = total

    from robomimic.utils.file_utils import create_hdf5_filter_key
    create_hdf5_filter_key(
        hdf5_path=tmp,
        demo_keys=["demo_%d" % i for i in range(a.n)],
        key_name="core_20")
    os.replace(tmp, a.out)
    print("wrote %s: %d demos, %d steps (value-identical to %s)"
          % (a.out, a.n, total, a.scout_core))

    # self-verify: pos3+grip bitwise equal to SCOUT; rot6d inverts to input
    dmax = 0.0
    with h5py.File(a.out, "r") as f1, h5py.File(a.scout_core, "r") as f2:
        keys = sorted(f1["data"].keys(), key=lambda s: int(s.split("_")[1]))
        for k in keys:
            a1 = f1["data"][k]["actions"][:]
            a2 = f2["data"][k]["abs_actions"][:]
            dmax = max(dmax, float(
                np.abs(a1[:, [0, 1, 2, 9]] - a2[:, [0, 1, 2, 6]]).max()))
            rv = rt.inverse(a1[:, 3:9])
            dmax = max(dmax, float(np.abs(rv - a2[:, 3:6]).max()))
    assert dmax < 1e-6, "value-identity check FAILED: %.2e" % dmax
    print("value-identity vs SCOUT core: %.2e (PASS)" % dmax)


if __name__ == "__main__":
    main()
