#!/usr/bin/env python
"""_smoke_shard_hb.py -- shard_heartbeat parser smoke (2026-09-04).

run_rollout's explore heartbeat line grew a ``solved=<rescued>/<fini>``
field; this checks the reporter regex + tail_last_hb against all three
line generations AND the pass@10 aggregation the reporter derives from
them. Pure stdlib, no /proc or wandb needed.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_heartbeat import HB_RE, tail_last_hb  # noqa: E402


def _write_tail(lines):
    f = tempfile.NamedTemporaryFile("wb", suffix=".stdout", delete=False)
    f.write(("\n".join(lines) + "\n").encode())
    f.close()
    return f.name


def main():
    # 1. regex on the three line generations
    new = (b"[explore-hb] 58/130 collected=10 solved=3/12 "
           b"rss=21.3G avail=756.6G elapsed=3551s")
    old = (b"[explore-hb] 58/130 collected=10 "
           b"rss=21.3G avail=756.6G elapsed=3551s")
    bare = b"[explore-hb] 5/130 rss=21.3G avail=756.6G elapsed=120s"
    m = HB_RE.search(new)
    assert m and m.group(1) == b"explore", "new line: no match"
    assert (m.group(2), m.group(3)) == (b"58", b"130")
    assert (m.group(4), m.group(5), m.group(6)) == (b"10", b"3", b"12"), \
        f"new line: solved groups wrong: {m.groups()}"
    m = HB_RE.search(old)
    assert m and m.group(4) == b"10" and m.group(5) is None \
        and m.group(6) is None, f"old line: should have coll, no solved: {m.groups()}"
    m = HB_RE.search(bare)
    assert m and m.group(4) is None and m.group(5) is None, \
        f"bare line: only done/total expected: {m.groups()}"
    # non-explore tags (baseline/eval hb) must be ignored by tail_last_hb
    mixed = [
        "[base-hb] 3/100 collected=1 rss=1.0G avail=2.0G elapsed=10s",
        old.decode(),
        new.decode(),
        "[eval-hb] 7/100 solved=9/9 rss=1.0G avail=2.0G elapsed=10s",
    ]
    p = _write_tail(mixed)
    got = tail_last_hb(p)
    assert got == (58, 130, 10, 3, 12), f"tail_last_hb last-line-wins: {got}"
    os.unlink(p)
    # only-old-lines file -> solved/fini None (back-compat: pre-09-04 worker)
    p = _write_tail([old.decode(), old.decode()])
    got = tail_last_hb(p)
    assert got == (58, 130, 10, None, None), f"old-only file: {got}"
    os.unlink(p)
    # empty / no-hb file -> None
    p = _write_tail(["nothing here"])
    assert tail_last_hb(p) is None, "no-hb file must return None"
    os.unlink(p)
    # tail window: hb line pushed BEYOND the last TAIL_BYTES must be missed;
    # the line closest to the end is what counts
    import shard_heartbeat as sh
    filler = ["x" * 200] * (sh.TAIL_BYTES // 200 + 10)
    p = _write_tail([old.decode()] + filler + [new.decode()])
    assert tail_last_hb(p) == (58, 130, 10, 3, 12)
    os.unlink(p)
    p = _write_tail([new.decode()] + filler)   # only line out of window
    assert tail_last_hb(p) is None, "out-of-window line must not be seen"
    os.unlink(p)
    # torn LAST line (writer mid-append): degrade to the previous whole line
    p = _write_tail([old.decode(),
                     new.decode().replace("solved=3/12", "solved=3/")])
    got = tail_last_hb(p)
    assert got == (58, 130, 10, None, None), f"torn tail: {got}"
    os.unlink(p)

    # 2. aggregation arithmetic as the reporter computes it (tuple layout)
    shards = {0: (58, 130, 10, 3, 12), 1: (52, 190, 18, 5, 17),
              2: (52, 220, None, None, None)}      # an old-format straggler
    solved_sum = sum(h[3] for h in shards.values() if h[3] is not None)
    fini_sum = sum(h[4] for h in shards.values() if h[4] is not None)
    assert (solved_sum, fini_sum) == (8, 29), (solved_sum, fini_sum)
    assert abs(solved_sum / fini_sum - 8 / 29) < 1e-12
    # straggler-only case: fini_sum 0 -> payload keys suppressed (division
    # guard), no ZeroDivisionError anywhere
    shards_old = {0: (5, 130, 1, None, None)}
    solved_sum = sum(h[3] for h in shards_old.values() if h[3] is not None)
    fini_sum = sum(h[4] for h in shards_old.values() if h[4] is not None)
    assert fini_sum == 0

    print("[smoke-shard-hb] all parser/aggregation checks OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
