"""Configuration, subprocesses and durable stage receipts (no ML imports)."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def round_name(arm, number):
    return f"{'SCOUT-aty' if arm == 'ATY' else arm}-round{number}"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", required=True, help="campaign JSON; paths relative to repository")
    p.add_argument("--dry-run", action="store_true", help="print commands without writing files")
    return p


class Context:
    def __init__(self, config, dry_run=False):
        self.c = dict(config)
        self.dry = dry_run
        self.task = self.c["task"]
        if self.task not in {"can", "coffee", "coffee_prep", "lift", "square", "threading", "tool_hang", "transport"}:
            raise ValueError(f"unsupported task: {self.task}")
        self.seed = int(self.c.get("seed", 233))
        self.gpu = str(self.c["gpu"])
        self.py = self.c.get("python", sys.executable)
        self.root = self.path(self.c["output_dir"])
        self.core = self.path(self.c["core_hdf5"])
        self.eval_config = self.path(self.c.get("eval_config", f"configs/{self.task}/eval.yaml"))
        self.project = self.c.get("wandb_project", self.root.name)
        self.wandb = self.c.get("wandb_mode", "disabled")
        if self.wandb not in {"disabled", "offline", "online"}:
            raise ValueError("wandb_mode must be disabled, offline or online")
        self.env = dict(os.environ, CUDA_VISIBLE_DEVICES=self.gpu, SCOUT_RENDER_GPU=self.gpu,
                        MUJOCO_GL="egl", TMPDIR="/tmp", CUBLAS_WORKSPACE_CONFIG=":4096:8",
                        PYTHONUNBUFFERED="1", WANDB_MODE=self.wandb)
        # Do not accidentally attach a new campaign to the launching shell's run.
        self.env.pop("WANDB_RUN_ID", None)
        self.env.pop("WANDB_RESUME", None)
        for key in ("dp", "dyn", "rollout", "calib", "grid"):
            if not isinstance(self.c.get(key, {}), dict):
                raise ValueError(f"{key} must be an object")
        for section, keys in (("dp", ("base_epochs", "epochs", "batch_size", "checkpoint_every")),
                              ("dyn", ("base_epochs", "epochs", "batch_size", "steps_per_epoch"))):
            if any(int(self.c.get(section, {}).get(k, 1)) < 1 for k in keys):
                raise ValueError(f"{section} training budgets must be positive")
        dose = self.c.get("dose", {"eta": 3., "kappa": 2.5})
        if any(not math.isfinite(float(dose[k])) or float(dose[k]) <= 0 for k in ("eta", "kappa")):
            raise ValueError("dose eta/kappa must be finite and positive")
        if not self.dry:
            self.require(self.core, self.eval_config)

    @classmethod
    def from_args(cls, args):
        return cls(read_json(args.config), args.dry_run)

    def path(self, value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else ROOT / p

    def require(self, *paths):
        if not self.dry:
            for p in paths:
                if not Path(p).is_file():
                    raise FileNotFoundError(p)

    def run(self, cmd, log, extra_env=None):
        cmd = list(map(str, cmd))
        print(f"{'DRY_RUN' if self.dry else 'RUN'}: {shlex.join(cmd)} -> {log}", flush=True)
        if self.dry:
            return
        Path(log).parent.mkdir(parents=True, exist_ok=True)
        with Path(log).open("w", encoding="utf-8") as stream:
            subprocess.run(cmd, cwd=ROOT, env={**self.env, **(extra_env or {})},
                           stdout=stream, stderr=subprocess.STDOUT, check=True)

    def module(self, name, args, log, **kwargs):
        self.run([self.py, "-m", name, *args], log, **kwargs)

    def training_options(self, section, *, base=False, overrides=None):
        """Return per-stage options with campaign values overlaid by overrides."""
        values = dict(self.c.get(section, {}))
        if overrides:
            values.update(overrides)
        if base:
            if "base_epochs" in values:
                values["epochs"] = values["base_epochs"]
            if "base_checkpoint_every" in values:
                values["checkpoint_every"] = values["base_checkpoint_every"]
        return values

    def stage(self, directory, spec, action):
        """Reuse only successful, matching stages; keep failed attempts for diagnosis."""
        directory = Path(directory)
        signature = hashlib.sha256(json.dumps({"config": self.c, "inputs": spec},
                                              sort_keys=True, default=str).encode()).hexdigest()
        receipt = directory / "done.json"
        if self.dry:
            return action(directory / "planned")
        directory.mkdir(parents=True, exist_ok=True)
        with lock(directory / ".lock"):
            if receipt.exists():
                saved = read_json(receipt)
                if saved["signature"] != signature:
                    raise ValueError(f"configuration/inputs changed: use a new output_dir ({receipt})")
                self.require(*saved["result"].get("artifacts", []))
                print(f"REUSE: {directory}", flush=True)
                return saved["result"]
            attempt = directory / ("attempt-" + uuid.uuid4().hex[:10])
            attempt.mkdir()
            result = action(attempt)
            result = json.loads(json.dumps(result, default=str))
            self.require(*result.get("artifacts", []))
            write_json(receipt, {"signature": signature, "inputs": spec, "result": result})
            return result


@contextmanager
def lock(path):
    """Kernel releases the lock on process exit, including interrupted runs."""
    with Path(path).open("a+b") as f:
        if os.name == "posix":
            import fcntl
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            import msvcrt
            f.write(b"0")
            f.flush()
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            if os.name == "posix":
                fcntl.flock(f, fcntl.LOCK_UN)
            else:
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


def checkpoint(directory, pattern, dry=False):
    if dry:
        return str(Path(directory) / ("checkpoints/planned.ckpt" if "checkpoints" in pattern else "planned/scout_vib.ckpt"))
    candidates = list(Path(directory).glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"no checkpoint: {directory}/{pattern}")
    return str(max(candidates, key=lambda p: p.stat().st_mtime_ns))


def dataset(ctx, output, paths):
    """Core once + the explicitly supplied rounds, never a glob over future rounds."""
    paths = [str(p) for p in paths if p]
    if ctx.dry:
        print(f"DRY_RUN: accumulate core={ctx.core} rounds={paths} -> {output}")
    else:
        from scout.eval.hdf5_writer import merge_accumulated_hdf5
        merge_accumulated_hdf5(str(ctx.core), paths, str(output))
    return str(output)


def finish(result):
    print(json.dumps(result, indent=2, default=str))
