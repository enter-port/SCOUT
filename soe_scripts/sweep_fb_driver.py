#!/usr/bin/env python
"""fb/beta/lambda sweep on the COLLAPSED setting (can r1 all_accum data).

Goal (user 2026-08-18): with the network structure unchanged, find the
parameter combo that keeps z alive on the explore-retrain data that killed
fb=0.005 (relu_alive -> 0.000, guidance exactly 0). Each run <= 300s wall
(feature cache hit; 100 epochs is enough -- the collapse was visible by ep 42).

Runs sequentially on one GPU; results appended to summary.tsv.
"""
import copy
import os
import re
import subprocess
import sys
import time

import yaml

REPO = "/root/workspace/baojiachun/scout"
PY = "/root/workspace/baojiachun/.venv/bin/python"
BASE_CFG = f"{REPO}/data/experiment2/can/train/dyn/dyn-SCOUT-exp1/config.yaml"
OUT = "/root/workspace/baojiachun/diag_vib/sweep_fb"
GPU = sys.argv[1] if len(sys.argv) > 1 else "4"

RUNS = [
    # (name, overrides)  -- baseline repro first
    ("base_fb0005_b1e-4_l5", {}),
    ("fb002",  {"free_bits": 0.02}),
    ("fb005",  {"free_bits": 0.05}),
    ("fb010",  {"free_bits": 0.10}),
    ("b3e-5",  {"beta": 3.0e-5}),
    ("b1e-5",  {"beta": 1.0e-5}),
    ("l1",     {"failure_weight": 1}),
    ("fb002_b3e-5", {"free_bits": 0.02, "beta": 3.0e-5}),
    ("wd0",    {"_wd": 0.0}),
    ("lr3e-4", {"_lr": 3.0e-4}),
]


def run_one(name, ov):
    cfg = copy.deepcopy(BASE)
    cfg["num_epochs"] = 100
    cfg["val_every"] = 1000            # skip val in screening
    cfg["use_wandb"] = False
    cfg["save_dir"] = f"{OUT}/{name}"
    cfg["free_bits"] = ov.get("free_bits", 0.005)
    cfg["beta"] = ov.get("beta", 1.0e-4)
    cfg["failure_weight"] = ov.get("failure_weight", 5)
    if "_wd" in ov:
        cfg["optimizer"]["params"]["weight_decay"] = ov["_wd"]
    if "_lr" in ov:
        cfg["optimizer"]["params"]["lr"] = ov["_lr"]
    cpath = f"{OUT}/{name}.config.yaml"
    with open(cpath, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    log = f"{OUT}/{name}.log"
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=GPU, MUJOCO_GL="egl", TMPDIR="/tmp")
    t0 = time.time()
    with open(log, "w") as lf:
        rc = subprocess.call(
            ["timeout", "300", PY, "-m", "scout.train_vib", "--config", cpath],
            cwd=REPO, env=env, stdout=lf, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    txt = open(log, errors="ignore").read()
    alive = re.findall(r"relu_alive ([0-9.]+)", txt)
    done = re.search(r"=== done \| .*latent_mse=([0-9.e-]+) kl=([0-9.e-]+) \|μ\|=([0-9.e-]+)", txt)
    gchk = re.search(r"\[guidance-check\] \|dNLL/da\| on a real batch = ([0-9.e-]+)", txt)
    row = [name, f"rc={rc}", f"{dt:.0f}s",
           f"alive_last={alive[-1] if alive else '?'}",
           f"alive_min={min(alive) if alive else '?'}",
           f"kl={done.group(2) if done else '?'}",
           f"mu={done.group(3) if done else '?'}",
           f"mse={done.group(1) if done else '?'}",
           f"grad={gchk.group(1) if gchk else '?'}"]
    line = "\t".join(row)
    print("[sweep] " + line, flush=True)
    with open(f"{OUT}/summary.tsv", "a") as f:
        f.write(line + "\n")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    with open(BASE_CFG) as f:
        BASE = yaml.safe_load(f)
    print(f"[sweep] base: beta={BASE.get('beta')} fb={BASE.get('free_bits')} "
          f"lam={BASE.get('failure_weight')} zarr={BASE['dataset']['zarr_path']}", flush=True)
    for name, ov in RUNS:
        run_one(name, ov)
    print("[sweep] ALL DONE", flush=True)
