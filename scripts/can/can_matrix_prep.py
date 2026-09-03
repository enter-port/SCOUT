"""Rebuild the per-round training-data banks for the CAN exploit matrix.

DP-SCOUT-exp{N} was trained on SCOUT-exp{N}/success_accum.hdf5 (round.log
[2/3] lines: ds=.../SCOUT-exp{N}/success_accum.hdf5 -> DP-SCOUT-exp{N}) =
core + success_1..N appended demos. Those accum files were pruned in the
08-29 cleanup, so rebuild them with the same merge function the chain used.
Each round's success.hdf5 is itself core + that round's appended rescue
successes (demo ids >= n_core, verified 08-30: s233 exp1 = 64 demos =
core 20 + 44 rescued), and merge_accumulated_hdf5 copies core once plus
each round's APPENDED demos only -- so the output equals the chain's
success_accum byte-composition.
Sentinel: demo count must equal n_core + sum(success_k - n_core).

Usage: can_matrix_prep.py SEED   (SEED in 233/2333/23333)
"""
import os
import sys

sys.path.insert(0, "/root/workspace/baojiachun/scout-exploit")

import h5py

from scout.eval.hdf5_writer import merge_accumulated_hdf5

ENT = "/root/workspace/baojiachun/scout-entropy/data/2026_8_21_entropy"
REPO = "/root/workspace/baojiachun/scout-exploit"


def ndemos(p):
    with h5py.File(p, "r") as f:
        return len(f["data"].keys())


def main():
    seed = sys.argv[1]
    base = f"{ENT}/CAN-entropy-s{seed}/can"
    core = f"{base}/rollout/can_core.hdf5"
    out = f"{REPO}/data/exploit_can_matrix/s{seed}"
    os.makedirs(out, exist_ok=True)
    ncore = ndemos(core)
    srcs = []
    for n in range(1, 7):
        srcs.append(f"{base}/rollout/SCOUT-exp{n}/success.hdf5")
        dst = f"{out}/bank_accum_r{n}.hdf5"
        expect = ncore + sum(ndemos(s) - ncore for s in srcs)
        if os.path.exists(dst):
            got = ndemos(dst)
            status = "OK" if got == expect else "MISMATCH"
            print(f"s{seed} r{n}: exists {got} demos (expect {expect}) {status}")
            continue
        info = merge_accumulated_hdf5(core, list(srcs), dst)
        got = ndemos(dst)
        assert got == expect, f"sentinel failed s{seed} r{n}: {got} != {expect}"
        print(f"s{seed} r{n}: {got} demos OK ({info})", flush=True)


if __name__ == "__main__":
    main()
