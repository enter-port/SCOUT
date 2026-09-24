#!/usr/bin/env python3
"""Preserve a successful base cap's quantile in the unguided DP KL drift.

This is identifiable only when the base cap lies inside the natural KL
distribution. It is not appropriate when virtually all natural KL is below
the cap, as happens for several tasks guided only late in denoising.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-source", type=Path, required=True)
    parser.add_argument("--round-source", type=Path, required=True)
    parser.add_argument("--base-kappa", type=float, default=2.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.base_kappa) or args.base_kappa <= 0:
        parser.error("base-kappa must be finite and positive")
    if args.out.exists():
        raise FileExistsError(args.out)
    meta, arrays = [], []
    for source in (args.base_source, args.round_source):
        directory = source / "diagnostics"
        d = json.loads((directory / "natural.json").read_text())
        if d["eta"] != 0 or d["rng_seed"] != 0 or d["B"] != 128:
            raise ValueError("Expected the same unguided core protocol")
        a = np.load(directory / "natural.npz")["kl"]
        if not np.isfinite(a).all() or a.size == 0:
            raise ValueError("Invalid natural KL measurements")
        meta.append(d)
        arrays.append(a)
    if any(meta[0][key] != meta[1][key] for key in ("task", "core_hdf5", "n_steps")):
        raise ValueError("Base and round probes must share task, core data and guidance schedule")
    quantile = float((arrays[0] <= args.base_kappa).mean())
    if not .05 < quantile < .95:
        raise ValueError(f"Base cap quantile {quantile:g} is too extreme to identify a reliable transfer")
    cap = float(np.quantile(arrays[1], quantile))
    if not np.isfinite(cap) or cap <= 0:
        raise ValueError("Transferred cap must be finite and positive")
    out = {key: meta[1][key] for key in ("task", "dp_ckpt", "vib_ckpt", "core_hdf5")}
    out.update(method="preserve_unguided_KL_quantile", kappa=cap, base_kappa=args.base_kappa,
               reference_quantile=quantile, natural_C_base=float(arrays[0].mean()),
               natural_C_round=float(arrays[1].mean()),
               base_dp_ckpt=meta[0]["dp_ckpt"], base_vib_ckpt=meta[0]["vib_ckpt"],
               base_probe=str(args.base_source / "diagnostics/natural.json"),
               round_probe=str(args.round_source / "diagnostics/natural.json"),
               next_step="Calibrate eta at this kappa to R=0.01, then validate paired pass@5")
    args.out.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
