#!/usr/bin/env python3
"""Check whether the six formal TOOL_HANG jobs are still running.

The check intentionally reports job liveness only. It does not read metrics,
GPU utilization, memory, logs, or rollout results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import psutil


SEEDS = (233, 2333, 23333)
ARMS = ("dp", "aty")


def resolve(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def identity_matches(seed: int, arm: str, state: dict, root: Path, manifest: dict) -> bool:
    try:
        process = psutil.Process(int(state["pid"]))
        cmd = process.cmdline()
    except (KeyError, TypeError, ValueError, psutil.Error):
        return False
    if not cmd:
        return False
    text = " ".join(cmd)
    job = f"s{seed}-{arm}"
    expected = manifest.get("jobs", {}).get(job, {})
    config_value = expected.get("config")
    state_config_value = state.get("config")
    if not state_config_value:
        return False
    state_config = resolve(state_config_value)
    if not state_config.is_file():
        return False
    config = resolve(config_value) if config_value else None
    exact_config = config == state_config if config else False
    variant_config = arm == "aty" and state_config.name.startswith(f"s{seed}-aty")
    if not (exact_config or variant_config):
        return False
    if "scripts.atom.parallel_campaign" in text:
        return exact_config and job in text and any(Path(x).name == "manifest.json" for x in cmd)
    if arm == "dp" and "scripts.atom.dp_handoff_recovery" in text:
        return f"s{seed}-dp" in text
    if arm == "aty" and "scripts.atom.dyn_variant_campaign" in text:
        return variant_config
    return False


def check(root: Path) -> dict:
    root = resolve(root)
    try:
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        manifest = {}
    jobs = {}
    for seed in SEEDS:
        for arm in ARMS:
            key = f"s{seed}-{arm}"
            path = root / "status" / f"s{seed}-{arm}.json"
            try:
                state = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                state = {}
            alive = state.get("status") == "running"
            if alive:
                try:
                    alive = psutil.pid_exists(int(state["pid"]))
                except (KeyError, TypeError, ValueError):
                    alive = False
            matched = identity_matches(seed, arm, state, root, manifest) if alive else False
            jobs[key] = bool(alive and matched)
    return {"all_running": all(jobs.values()), "jobs": jobs}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    result = check(args.root)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_running"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
