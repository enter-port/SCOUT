#!/usr/bin/env python
"""Sharded-explore heartbeat reporter (2026-09-02 user order).

Why: sharded phase-B workers run with --no-wandb, so during the HOURS the
rescue explore runs there was zero observable progress -- an OOM'd worker was
indistinguishable from a healthy one until someone noticed hours later.

What it does (every --interval s):
  * parses each shard stdout's LAST ``[explore-hb] done/total collected=...``
    line (printed unconditionally by rollout_vec's progress_cb since today);
  * scans /proc for this campaign's workers (cmdline has ``--scene-slice``
    AND the --match substring, e.g. the round's rollout dir) and reads each
    worker's VmRSS + state -- worker->shard mapping comes from the
    ``--scene-slice <slot>:<P>`` token in the same cmdline;
  * reads /proc/meminfo (MemAvailable/MemTotal);
  * wandb-logs ``explore_hb/*`` keys into the round's phase-A run
    (resume=must; auto-step keeps the shared run's step axis monotonic) and
    appends one grep-able line to --log-file.

Lifecycle: exits 0 when --stop-file (the merged all.hdf5) appears; exits 3
after --missing-polls consecutive polls with zero matching workers (crashed/
finished-without-merge); exits 4 at --max-min. SIGTERM -> wandb.finish() and
exit 0 (the round driver kills it when phase B returns). A wandb failure
never kills the reporter -- it keeps writing heartbeat.log.
"""
import argparse
import glob
import os
import re
import signal
import sys
import time

HB_RE = re.compile(
    rb"\[(\w+)-hb\] (\d+)/(\d+)(?: collected=(\d+))?(?: succ=(\d+))?")
TAIL_BYTES = 8192


def tail_last_hb(path):
    """(done, total, collected) from the last explore-hb line in `path`,
    or None if the worker has not printed one yet."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - TAIL_BYTES))
            chunk = f.read()
    except OSError:
        return None
    best = None
    for m in HB_RE.finditer(chunk):
        if m.group(1) == b"explore":
            best = m
    if best is None:
        return None
    done, total = int(best.group(2)), int(best.group(3))
    coll = int(best.group(4)) if best.group(4) is not None else None
    return done, total, coll


def find_workers(match):
    """This campaign's shard workers: [{pid, slot, P, rss_gb, state}]."""
    out = []
    me = os.getpid()
    needle = match.encode()
    try:
        procs = os.listdir("/proc")
    except OSError:                    # non-Linux (local dev) -- no workers
        return out
    for p in procs:
        if not p.isdigit() or int(p) == me:
            continue
        try:
            with open(f"/proc/{p}/cmdline", "rb") as f:
                cmd = f.read()
        except OSError:
            continue
        if b"--scene-slice" not in cmd or needle not in cmd:
            continue
        # /proc cmdline args are NUL-separated: parse argv properly instead
        # of regexing the raw buffer (a space-based regex never matches).
        slot = P = None
        try:
            argv = cmd.split(b"\0")
            k = argv.index(b"--scene-slice")
            slot, P = (int(x) for x in argv[k + 1].split(b":"))
        except (ValueError, IndexError):
            pass
        rss, state = None, "?"
        try:
            with open(f"/proc/{p}/status") as f:      # text mode -> str
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss = int(line.split()[1]) / 1e6   # kB -> GB
                    elif line.startswith("State:"):
                        state = line.split()[1]
        except OSError:
            pass
        out.append({"pid": int(p), "slot": slot, "P": P,
                    "rss": rss, "state": state})
    return sorted(out, key=lambda w: (w["slot"] if w["slot"] is not None
                                      else 1 << 30, w["pid"]))


def sys_mem():
    """(avail_gb, total_gb) from /proc/meminfo; (None, None) elsewhere."""
    avail = total = None
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    avail = int(line.split()[1]) / 1e6
                elif line.startswith("MemTotal:"):
                    total = int(line.split()[1]) / 1e6
    except OSError:
        pass
    return avail, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--shard-glob", required=True,
                    help="e.g. <RDIR>/log/shard*.stdout")
    ap.add_argument("--match", required=True,
                    help="substring identifying this round's workers in "
                         "/proc cmdlines (e.g. the rollout dir)")
    ap.add_argument("--stop-file", required=True,
                    help="exit 0 when this exists (the merged all.hdf5)")
    ap.add_argument("--log-file", required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("--missing-polls", type=int, default=5,
                    help="exit 3 after N consecutive polls with 0 workers")
    ap.add_argument("--max-min", type=int, default=240)
    args = ap.parse_args()

    run = None
    try:
        import wandb
        run = wandb.init(id=args.run_id, project=args.project,
                         resume="must")
        print(f"[hb] wandb resumed {args.project}/{args.run_id}", flush=True)
    except Exception as e:                      # never kill the reporter
        print(f"[hb] WARN: wandb unavailable ({e}) -- heartbeat.log only",
              flush=True)

    stop = {"flag": False}

    def _term(signum, frame):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGINT, _term)

    t0 = time.time()
    missing = 0
    poll = 0
    rc = 0
    while True:
        poll += 1
        workers = find_workers(args.match)
        stdouts = sorted(glob.glob(args.shard_glob))
        shard_hb = {}
        for path in stdouts:
            m = re.search(r"shard(\d+)\.stdout$", os.path.basename(path))
            if not m:
                continue
            hb = tail_last_hb(path)
            if hb is not None:
                shard_hb[int(m.group(1))] = hb
        avail, total = sys_mem()

        done_sum = sum(h[0] for h in shard_hb.values())
        total_sum = sum(h[1] for h in shard_hb.values())
        stamp = time.strftime("%F %T")
        rss_txt = ",".join(
            "{}:{}/{}".format(
                w["slot"],
                "?" if w["rss"] is None else "{:.1f}G".format(w["rss"]),
                w["state"])
            for w in workers) or "-"
        sys_txt = ((f"{avail:.1f}/{total:.0f}G"
                    f"({100 * (1 - avail / total):.0f}% used)")
                   if avail is not None and total else "?")
        line = (f"{stamp} poll={poll} alive={len(workers)} "
                f"done={done_sum}/{total_sum if total_sum else '?'} "
                f"shards={len(shard_hb)} rss[{rss_txt}] sys={sys_txt}")
        try:
            with open(args.log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

        if run is not None:
            payload = {
                "explore_hb/alive": len(workers),
                "explore_hb/done": done_sum,
                "explore_hb/total": total_sum,
                "explore_hb/n_shards_reporting": len(shard_hb),
            }
            if avail is not None:
                payload["explore_hb/sys_avail_gb"] = round(avail, 1)
            if total is not None:
                payload["explore_hb/sys_total_gb"] = round(total, 1)
                if avail is not None:
                    payload["explore_hb/sys_used_pct"] = round(
                        100 * (1 - avail / total), 1)
            for slot, (d, t, c) in shard_hb.items():
                payload[f"explore_hb/shard{slot}_done"] = d
                if c is not None:
                    payload[f"explore_hb/shard{slot}_collected"] = c
            for w in workers:
                if w["slot"] is not None and w["rss"] is not None:
                    payload[f"explore_hb/shard{w['slot']}_rss_gb"] = \
                        round(w["rss"], 1)
            try:
                run.log(payload)
            except Exception as e:
                print(f"[hb] WARN: wandb.log failed ({e})", flush=True)

        if os.path.exists(args.stop_file):
            print(f"[hb] stop-file present ({args.stop_file}) -- exit 0",
                  flush=True)
            rc = 0
            break
        if stop["flag"]:
            print("[hb] SIGTERM -- exit 0", flush=True)
            rc = 0
            break
        missing = missing + 1 if not workers else 0
        if poll > 3 and missing >= args.missing_polls:
            print(f"[hb] no matching workers for {missing} polls -- "
                  f"workers gone (crashed or finished without merge); "
                  f"exit 3", flush=True)
            rc = 3
            break
        if time.time() - t0 > args.max_min * 60:
            print(f"[hb] max-min {args.max_min} reached -- exit 4",
                  flush=True)
            rc = 4
            break
        time.sleep(args.interval)

    if run is not None:
        try:
            run.finish(exit_code=rc)
        except Exception:
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
