"""Rebuild r3 success_accum for the DIAGNOSTIC r3-bank arm (user decision
pending; clearly-labeled upper-bound experiment -- contains r3 rescue
successes collected ON the 100 test scenes).

r3 success_accum = core + SCOUT-exp{1,2,3}/success.hdf5 appended demos
(exactly what round_entropy.sh [2/3] builds for the DP arm's round-4 data).
"""
import sys

sys.path.insert(0, "/root/workspace/baojiachun/scout-exploit")

from scout.eval.hdf5_writer import merge_accumulated_hdf5

BASE = ("/root/workspace/baojiachun/scout-entropy/data/"
        "2026_8_26_entropy/SQUARE-entropy-s233/square")
CORE = BASE + "/rollout/square_core.hdf5"
SUCCS = [BASE + f"/rollout/SCOUT-exp{i}/success.hdf5" for i in (1, 2, 3)]
OUT = ("/root/workspace/baojiachun/scout-exploit/data/"
       "exploit_sq233_r3/bank_r3accum.hdf5")

import os
for p in SUCCS:
    assert os.path.exists(p), p
info = merge_accumulated_hdf5(CORE, SUCCS, OUT)
print(f"[r3accum] {info} -> {OUT}")
