#!/usr/bin/env python3
"""Validate paired pass@5 summaries and report rescue counts.

Read-only by default. --out writes a compact JSON report; no rollout files
or checkpoints are changed. A missing arm is pending, never a zero score.
"""
import argparse
import json
import math
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def task_rows(directory):
    baseline_path = directory / "dp/summary.json"
    if not baseline_path.exists():
        return []
    baseline = read(baseline_path)
    metadata_path = directory / "checkpoints.json"
    if not metadata_path.exists():
        metadata_path = directory / "diagnostics/natural.json"
    metadata = read(metadata_path)
    if baseline.get("guided") != 0:
        raise ValueError(f"DP control unexpectedly uses guidance: {directory}")
    failures = set(baseline["failed_init_indices"])
    dp_solved = {d["init"] for d in baseline["explore_detail"] if d["solved"]}
    if len(dp_solved) != baseline["exploration_rescued"]:
        raise ValueError(f"DP scene count mismatch: {directory}")
    rows = []
    for path in sorted(directory.glob("*/summary.json")):
        if path.parent.name in ("eval", "action_sensitivity", "action_response", "dp_diversity"):
            continue
        result = read(path)
        is_dp = path.parent.name == "dp"
        if result.get("guided") != (0 if is_dp else 1):
            raise ValueError(f"Unexpected guidance mode: {path}")
        if not is_dp and result.get("vib_ckpt") != metadata["vib_ckpt"]:
            raise ValueError(f"Different dyn checkpoint from declared pair: {path}")
        if not is_dp:
            manifest_path = path.parent / "manifest.json"
            if manifest_path.exists():
                manifest = read(manifest_path)
                calibrated = manifest["calibration"]
                command = manifest["command"]
                actual_cap = float(command[command.index("--atypical-cap") + 1])
            else:
                calibrated = read(directory / "diagnostics" / f"{path.parent.name}.json")
                actual_cap = read(path.parent / "kl_probe.json")["kappa"]
            guidance = result["guidance"]
            if (guidance["guide"] != "atypical" or guidance["eta_dimless"] != 0
                    or not math.isclose(guidance["guidance_scale"], calibrated["eta"], rel_tol=1e-9)
                    or not math.isclose(actual_cap, calibrated["kappa"], rel_tol=1e-9)):
                raise ValueError(f"Rollout guidance disagrees with the declared calibration: {path}")
        for key in ("dp_ckpt", "core_hdf5", "baseline_solved", "n_failed"):
            if result[key] != baseline[key]:
                raise ValueError(f"Unpaired {key}: {path}")
        for key, expected in (("n_init_states", 100), ("seed", 42),
                              ("explore_try_times", 5), ("shards_merged", 4)):
            if result[key] != expected:
                raise ValueError(f"Protocol mismatch {key}: {path}")
        if set(result["failed_init_indices"]) != failures:
            raise ValueError(f"Different frozen failure set: {path}")
        detail = result["explore_detail"]
        if len(detail) != len(failures) or {d["init"] for d in detail} != failures:
            raise ValueError(f"Incomplete scene accounting: {path}")
        solved = {d["init"] for d in detail if d["solved"]}
        count = len(solved)
        if count != result["exploration_rescued"]:
            raise ValueError(f"Rescue count mismatch: {path}")
        if abs(result["pass_at_5"] - (result["baseline_solved"] + count) / 100) > 1e-8:
            raise ValueError(f"pass@5 mismatch: {path}")
        dp_count = len(dp_solved)
        verdict = "excellent" if count > 1.5 * dp_count else "minimum" if count > dp_count else "not_above_DP"
        if path.parent.name == "dp":
            verdict = "control"
        rows.append(dict(task=baseline["task"], arm=path.parent.name,
                         initial_success=result["baseline_solved"], rescued=count,
                         dp_rescued=dp_count, ratio=count / dp_count if dp_count else None,
                         pass_at_5=result["pass_at_5"], verdict=verdict,
                         n_failed=len(failures), minimum_threshold=dp_count + 1,
                         excellent_threshold=math.floor(1.5 * dp_count) + 1,
                         minimum_attainable=len(failures) > dp_count,
                         excellent_attainable=len(failures) > 1.5 * dp_count,
                         gain_vs_dp=len(solved - dp_solved), loss_vs_dp=len(dp_solved - solved),
                         solved_indices=sorted(solved), source=str(path)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="directory containing task subdirectories")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    rows = []
    for task in ("can", "square", "coffee", "threading", "tool_hang"):
        rows.extend(task_rows(args.root / task))
    print("| Task | Arm | Rescued | DP | Ratio | pass@5 | Verdict | Gained / lost vs DP |")
    print("|---|---|---:|---:|---:|---:|---|---:|")
    for row in rows:
        ratio = f"{row['ratio']:.3f}" if row["ratio"] is not None else "undefined"
        print(f"| {row['task']} | {row['arm']} | {row['rescued']} | {row['dp_rescued']} | "
              f"{ratio} | {row['pass_at_5']:.2f} | {row['verdict']} | "
              f"{row['gain_vs_dp']} / {row['loss_vs_dp']} |")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2) + "\n")


if __name__ == "__main__":
    main()
