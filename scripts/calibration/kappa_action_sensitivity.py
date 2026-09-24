#!/usr/bin/env python3
"""Core-action KL sensitivity to isotropic perturbations in DP action units.

CPU only. This measures a candidate calibration statistic, not a success
predictor: S(epsilon) = mean KL / epsilon**2. sqrt(kappa/S) is a local,
direction-averaged equivalent action radius. Test epsilon dependence before
treating the local quadratic approximation as valid.
"""
import argparse
import json
import os
from pathlib import Path
import sys

# Some checkpoint factories choose CUDA internally. Hide it before importing
# torch so this diagnostic cannot allocate on another person's GPU.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def run_task(source, epsilons, directions, output_name="action_sensitivity", metadata_file=None):
    import h5py
    import numpy as np
    import torch
    from scout.eval.factories import load_cfg, make_scout_vib_factory
    from scout.eval.rollout import make_obs_adapter
    from scout.guidance.expert_bank import _default_aa_to_6d
    from diffusion_policy.model.common.normalizer import LinearNormalizer

    metadata = json.loads((metadata_file or source / "diagnostics/natural.json").read_text())
    task = metadata["task"]
    output = source / output_name
    output.mkdir(exist_ok=True)
    if (output / "summary.json").exists():
        raise FileExistsError(output / "summary.json")
    cfg = load_cfg(str(ROOT / f"configs/eval_{task}_entropy.yaml"))
    cfg.vib.base_dp_ckpt, cfg.vib.ckpt_path = metadata["dp_ckpt"], metadata["vib_ckpt"]
    views, proprio = list(cfg.eval.view_names), list(cfg.eval.proprio_keys)
    # The original ResNet checkpoint loader has no map_location argument.
    # Limit this override to model construction in this diagnostic process.
    original_load = torch.load

    def load_cpu(*args, **kwargs):
        kwargs["map_location"] = "cpu"
        return original_load(*args, **kwargs)

    torch.load = load_cpu
    try:
        vib = make_scout_vib_factory(cfg, torch.device("cpu"))(metadata["vib_ckpt"])
    finally:
        torch.load = original_load
    vib.eval()
    normalizer_file = Path(metadata["dp_ckpt"]).parents[1] / "normalizer.pth"
    normalizer = LinearNormalizer()
    if normalizer_file.is_file():
        normalizer.load_state_dict(torch.load(normalizer_file, map_location="cpu"))
        normalizer_source = str(normalizer_file)
    else:
        payload = torch.load(metadata["dp_ckpt"], map_location="cpu")
        if "state_dicts" in payload:
            slots = payload["state_dicts"]
            weights = slots.get("model", slots.get("policy", slots.get("ema_model")))
        else:
            weights = payload.get("model", payload.get("policy", payload.get("state_dict")))
        saved = {key[len("normalizer."):]: value for key, value in weights.items()
                 if key.startswith("normalizer.")}
        if not saved:
            raise ValueError("DP checkpoint has no fitted normalizer")
        normalizer.load_state_dict(saved)
        normalizer_source = metadata["dp_ckpt"] + ":normalizer"
        del payload, weights, saved
    obs_rows, chunks, indices = [], [], []
    with h5py.File(metadata["core_hdf5"], "r") as f:
        demos = sorted(f["data"].keys())
        stride = max(1, sum(len(f[f"data/{d}/abs_actions"]) for d in demos) // (128 * 4))
        for demo in demos:
            g = f[f"data/{demo}"]
            raw_actions = g["abs_actions"][:]
            if raw_actions.shape[-1] in [7, 14]:
                raw_actions = _default_aa_to_6d(raw_actions)
            elif raw_actions.shape[-1] not in [10, 20]:
                raise ValueError("Unsupported core action representation")
            per_step = raw_actions.shape[-1]
            if vib.vib_enc.action_dim % per_step:
                raise ValueError("Encoder chunk dimension is not divisible by action dimension")
            horizon = vib.vib_enc.action_dim // per_step
            for t in range(1, len(raw_actions) - 1, stride):
                if len(obs_rows) == 128:
                    break
                row = {v: np.moveaxis(g[f"obs/{v}"][t:t+1], -1, 1).astype(np.float32) for v in views}
                row.update({k: g[f"obs/{k}"][t:t+1].astype(np.float32) for k in proprio})
                chunk = raw_actions[t:t+horizon]
                if len(chunk) < horizon:
                    chunk = np.pad(chunk, ((0, horizon - len(chunk)), (0, 0)), mode="edge")
                obs_rows.append(row)
                chunks.append(chunk)
                indices.append([demo, t])
            if len(obs_rows) == 128:
                break
    obs = {key: torch.as_tensor(np.stack([row[key] for row in obs_rows])) for key in views + proprio}
    actions = torch.as_tensor(np.stack(chunks), dtype=torch.float32)
    adapter = make_obs_adapter(views, proprio)
    with torch.inference_mode():
        # Small CPU batches limit temporary activation memory.
        states = []
        for start in range(0, len(obs_rows), 16):
            states.append(vib.encode(adapter({key: value[start:start+16] for key, value in obs.items()})))
        state = torch.cat(states)
        normalized = normalizer["action"].normalize(actions)
        reconstructed = normalizer["action"].unnormalize(normalized)
        if not torch.allclose(actions, reconstructed, atol=1e-5, rtol=1e-5):
            raise ValueError("Action normalizer round trip failed")
        mu0, lv0 = vib.vib_enc(state, reconstructed.flatten(1))
        mu0, lv0 = mu0.double(), lv0.double()
        generator = torch.Generator().manual_seed(0)
        noise = torch.randn((directions, *normalized.shape), generator=generator)
        noise /= noise.square().mean(dim=(-2, -1), keepdim=True).sqrt()
        results, raw = [], {}
        for epsilon in epsilons:
            samples = []
            for direction in noise:
                perturbed = normalizer["action"].unnormalize(normalized + epsilon * direction)
                mu, lv = vib.vib_enc(state, perturbed.flatten(1))
                delta_lv = lv.double() - lv0
                kl = .5 * ((mu.double() - mu0).square() * torch.exp(-lv0)
                           + torch.expm1(delta_lv) - delta_lv).sum(-1)
                samples.append(kl.numpy())
            values = np.stack(samples)
            if not np.isfinite(values).all() or (values < -1e-12).any():
                raise ValueError("Invalid KL sensitivity measurement")
            sensitivity = values.mean(axis=0) / epsilon ** 2
            record = {"epsilon_rms": epsilon, "mean_KL": float(values.mean()),
                      "S_mean": float(sensitivity.mean()), "S_median": float(np.median(sensitivity)),
                      "S_p10": float(np.quantile(sensitivity, .1)), "S_p90": float(np.quantile(sensitivity, .9))}
            results.append(record)
            raw[f"kl_eps_{epsilon:g}"] = values
            print(f"[{task}] sensitivity {record}", flush=True)
    summary = {key: metadata[key] for key in ["task", "dp_ckpt", "vib_ckpt", "core_hdf5"]}
    summary.update(method="normalized_core_action_isotropic_sensitivity" if output_name == "action_sensitivity"
                   else "normalized_core_action_finite_response", B=len(obs_rows),
                   directions=directions, action_chunk_steps=horizon, unclipped=True,
                   action_representation="position_rotation6d_gripper",
                   normalizer_source=normalizer_source,
                   core_indices=indices, results=results, device="cpu")
    np.savez_compressed(output / "samples.npz", **raw)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", required=True,
                        choices=["can", "square", "coffee", "threading", "tool_hang"])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[.01, .05, .1])
    parser.add_argument("--directions", type=int, default=16)
    parser.add_argument("--output-name", choices=["action_sensitivity", "action_response"], default="action_sensitivity")
    parser.add_argument("--metadata-name", default="diagnostics/natural.json",
                        help="checkpoint manifest path relative to each task directory")
    args = parser.parse_args()
    if args.directions < 1 or any(not 0 < x < float("inf") for x in args.epsilons):
        parser.error("directions and finite epsilons must be positive")
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if torch.cuda.is_available():
        raise RuntimeError("This diagnostic must not use CUDA")
    source_root = args.root.resolve()
    runtime = source_root.parent / "sensitivity_runtime"
    runtime.mkdir(exist_ok=True)
    os.chdir(runtime)
    for task in args.tasks:
        run_task(source_root / task, args.epsilons, args.directions, args.output_name,
                 source_root / task / args.metadata_name)


if __name__ == "__main__":
    main()
