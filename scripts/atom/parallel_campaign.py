"""Run shared seed bases, then independent campaign atoms on assigned GPUs.

Parent implementation: scripts.atom.base_train and scripts.atom.campaign.
Added for registered TOOL_HANG-2026-09-25 core40 campaign. This module only
schedules existing atoms; it imposes no exploration lock or memory threshold.
Manifest/config files live in the registered experiment's data directory.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

from . import base_train, campaign
from .common import Context, ROOT, lock, read_json, write_json


def status_path(manifest, name):
    return Path(manifest["root"]) / "status" / f"{name}.json"


def active(state, name, manifest_path):
    if state.get("status") != "running":
        return False
    try:
        cmdline = Path(f"/proc/{int(state['pid'])}/cmdline").read_bytes().split(b"\0")
        return (b"scripts.atom.parallel_campaign" in cmdline
                and str(manifest_path).encode() in cmdline and name.encode() in cmdline)
    except (OSError, KeyError, ValueError):
        return False


def job(manifest_path, name):
    manifest = read_json(manifest_path)
    spec = manifest["jobs"][name]
    ctx = Context(read_json(spec["config"]))
    ctx.root.mkdir(parents=True, exist_ok=True)
    with lock(ctx.root / ".chain.lock"):
        state = dict(status="running", pid=os.getpid(), kind=spec["kind"],
                     config=spec["config"], seed=ctx.seed, gpu=ctx.gpu,
                     started_at=time.time(), attempt=uuid.uuid4().hex)
        write_json(status_path(manifest, name), state)
        try:
            result = base_train.prepare(ctx) if spec["kind"] == "base" else campaign.run(ctx)
        except BaseException:
            state.update(status="failed", finished_at=time.time(), error=traceback.format_exc())
            write_json(status_path(manifest, name), state)
            raise
        state.update(status="completed", finished_at=time.time(), result=result)
        write_json(status_path(manifest, name), state)


def launch_and_wait(manifest_path, manifest, names):
    processes = {}
    for name in names:
        path = status_path(manifest, name)
        state = read_json(path) if path.exists() else {}
        if state.get("status") == "completed" or active(state, name, manifest_path):
            continue
        log = Path(manifest["root"]) / "logs" / f"{name}-{time.time_ns()}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w") as stream:
            processes[name] = subprocess.Popen(
                [sys.executable, "-u", "-m", "scripts.atom.parallel_campaign",
                 "job", "--manifest", str(manifest_path), "--name", name],
                cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    while True:
        states = {}
        for name in names:
            path = status_path(manifest, name)
            state = read_json(path) if path.exists() else {}
            process = processes.get(name)
            if process is not None and process.poll() is None:
                states[name] = "running"
                continue
            if state.get("status") == "completed":
                states[name] = "completed"
            elif active(state, name, manifest_path):
                states[name] = "running"
            else:
                if state.get("status") != "failed":
                    state.update(status="failed", error="Job exited without a completion receipt",
                                 detected_at=time.time())
                    write_json(path, state)
                states[name] = "failed"
        if all(s != "running" for s in states.values()):
            if any(s == "failed" for s in states.values()):
                raise RuntimeError(f"Failed jobs (healthy jobs were allowed to finish): {states}")
            return
        time.sleep(5)


def supervise(manifest_path):
    manifest = read_json(manifest_path)
    root = Path(manifest["root"])
    with lock(root / ".supervisor.lock"):
        bases = [name for name, spec in manifest["jobs"].items() if spec["kind"] == "base"]
        launch_and_wait(manifest_path, manifest, bases)
        chains = []
        for name, spec in manifest["jobs"].items():
            if spec["kind"] != "chain":
                continue
            pair = read_json(status_path(manifest, spec["base_job"]))["result"]
            Context(read_json(manifest["jobs"][spec["base_job"]]["config"])).require(pair["dp"], pair["dyn"])
            config_path = Path(spec["config"])
            config = read_json(config_path)
            for key in ("dp", "dyn"):
                field = "base_" + key
                if field in config and config[field] != pair[key]:
                    raise ValueError(f"Refusing to change an existing base checkpoint: {config_path}")
                config[field] = pair[key]
            if config != read_json(config_path):
                write_json(config_path, config)
            chains.append(name)
        launch_and_wait(manifest_path, manifest, chains)
        write_json(root / "completed.json", {"completed_at": time.time(), "jobs": chains})


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("action", choices=["supervise", "job"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--name")
    args = parser.parse_args()
    if args.action == "job":
        if not args.name:
            parser.error("job requires --name")
        job(args.manifest.resolve(), args.name)
    else:
        supervise(args.manifest.resolve())


if __name__ == "__main__":
    main()
