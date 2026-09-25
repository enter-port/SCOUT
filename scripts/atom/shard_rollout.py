"""Run scene-sharded rescue workers and merge their outputs."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def _suffix(path, index, count):
    path = Path(path)
    return path.with_name(f"{path.stem}-shard{index}of{count}{path.suffix}")


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) < 7 or argv[5] != "--":
        raise SystemExit("usage: python -m scripts.atom.shard_rollout P OUT_JSON OUT_SUCCESS OUT_ALL CORE -- ARGS...")
    workers = int(argv[0])
    out_json, out_success, out_all, core = map(Path, argv[1:5])
    args = argv[6:]
    output_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    output_parser.add_argument("--output-dir")
    output_options, _ = output_parser.parse_known_args(args)
    if workers < 1:
        raise SystemExit("workers must be positive")
    for path in (out_json, out_success, out_all):
        path.parent.mkdir(parents=True, exist_ok=True)

    procs = []
    try:
        for index in range(workers):
            command = [sys.executable, "-m", "scout.eval.run_rollout",
                       "--scene-slice", f"{index}:{workers}",
                       "--output-json", str(out_json), *args]
            log = out_json.parent / f"shard{index}.stdout"
            stream = log.open("w", encoding="utf-8")
            procs.append((subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT), stream))
        failed = False
        for process, _ in procs:
            if process.wait() != 0:
                failed = True
        if failed:
            raise SystemExit("one or more shard workers failed; outputs were not merged")
    finally:
        for process, stream in procs:
            if process.poll() is None:
                process.terminate()
            stream.close()

    jsons = [_suffix(out_json, i, workers) for i in range(workers)]
    successes = [_suffix(out_success, i, workers) for i in range(workers)]
    alls = [_suffix(out_all, i, workers) for i in range(workers)]
    command = [sys.executable, "-m", "scout.eval.merge_sharded",
               "--jsons", *map(str, jsons), "--out-json", str(out_json),
               "--success-hdf5s", *map(str, successes), "--out-success", str(out_success),
               "--all-hdf5s", *map(str, alls), "--out-all", str(out_all),
               "--core-hdf5", str(core)]
    subprocess.run(command, check=True)

    if os.environ.get("CLEANUP_SHARDS") == "1":
        for path in [*jsons, *successes, *alls,
                     *[out_json.parent / f"shard{i}.stdout" for i in range(workers)]]:
            if path.exists():
                path.unlink()
        if output_options.output_dir:
            # run_rollout appends the tag to --output-dir, creating siblings,
            # independently of where --output-success/--output-all point.
            directory = Path(output_options.output_dir)
            for index in range(workers):
                candidate = Path(str(directory) + f"-shard{index}of{workers}")
                if (not candidate.is_symlink() and candidate.is_dir()
                        and candidate.resolve().parent == directory.resolve().parent):
                    shutil.rmtree(candidate)
    print(f"[shard_rollout] merged -> {out_json} {out_success} {out_all}", flush=True)


if __name__ == "__main__":
    main()
