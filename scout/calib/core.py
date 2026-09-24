"""Shared frozen observations, model setup, and paired guidance measurements."""
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
IDENTITY_KEYS = ("task", "dp_ckpt", "vib_ckpt", "core_hdf5")
TASKS = ("can", "coffee", "coffee_prep", "lift", "square", "threading", "tool_hang", "transport")


def check_gpu(gpu, uuid):
    """Check the physical GPU before loading CUDA; preserve the research guard."""
    if gpu is None or uuid is None:
        raise ValueError("New measurements require an explicit GPU index and UUID")
    reading = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu), "--query-gpu=uuid,memory.used",
        "--format=csv,noheader,nounits"], text=True).strip().split(",")
    if (reading[0].strip() != uuid or int(reading[1]) >= 128
            or uuid == "GPU-349c44c7-da75-dba2-5a49-8f9a19a84832"):
        raise RuntimeError(f"GPU identity/occupancy check failed: {reading}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def read_core_batch(path, views, proprio, batch_size=128, *, scale_order="sorted"):
    """Return numpy observations and mean |raw abs_actions| over the whole core.

    The frozen batch uses sorted demos and (t-1,t) frames. Legacy eta uses HDF5
    iteration order for the scale reduction; C-only probes need no scale.
    Keeping that distinction also preserves floating point reduction order.
    """
    import h5py
    import numpy as np

    if batch_size < 1 or scale_order not in ("sorted", "stored", None):
        raise ValueError("Invalid batch size or action scale order")
    observations = []
    with h5py.File(path, "r") as f:
        demos = sorted(f["data"].keys())
        stride = max(1, sum(len(f[f"data/{d}/abs_actions"]) for d in demos)
                     // (batch_size * 4))
        for demo in demos:
            group = f[f"data/{demo}"]
            for t in range(1, len(group["abs_actions"]) - 1, stride):
                if len(observations) >= batch_size:
                    break
                obs = {}
                for key in views:
                    frames = np.stack([group[f"obs/{key}"][t - 1], group[f"obs/{key}"][t]])
                    obs[key] = np.moveaxis(frames, -1, 1).astype(np.float32) / 255.0
                for key in proprio:
                    obs[key] = np.stack([group[f"obs/{key}"][t - 1],
                                         group[f"obs/{key}"][t]]).astype(np.float32)
                observations.append(obs)
            if len(observations) >= batch_size:
                break
        scale = None
        if scale_order is not None:
            order = demos if scale_order == "sorted" else f["data"].keys()
            scale = float(np.abs(np.concatenate([f[f"data/{d}/abs_actions"][:] for d in order])).mean())
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError("Invalid core action scale")
    if not observations:
        raise ValueError("Core contains no two-frame observations")
    return {key: np.stack([row[key] for row in observations])
            for key in list(views) + list(proprio)}, scale


def prepare_core(eval_config, core_hdf5, batch_size=128, *, scale_order="sorted"):
    import torch
    from scout.eval.factories import load_cfg

    cfg = load_cfg(str(eval_config))
    arrays, scale = read_core_batch(core_hdf5, list(cfg.eval.view_names),
                                   list(cfg.eval.proprio_keys), batch_size, scale_order=scale_order)
    obs = {key: torch.as_tensor(value).to(torch.device("cuda")) for key, value in arrays.items()}
    return cfg, obs, scale


def load_model_pair(cfg, dp_ckpt, vib_ckpt):
    import torch
    from scout.eval.factories import make_lpb_dp_factory, make_scout_vib_factory
    from scout.eval.rollout import make_action_bridge, make_obs_adapter

    device = torch.device("cuda")
    dp = make_lpb_dp_factory(device)(str(dp_ckpt))
    dp.eval()
    cfg.vib.ckpt_path, cfg.vib.base_dp_ckpt = str(vib_ckpt), str(dp_ckpt)
    vib = make_scout_vib_factory(cfg, device)(str(vib_ckpt))
    vib.eval()
    views, proprio = list(cfg.eval.view_names), list(cfg.eval.proprio_keys)
    adapt = make_obs_adapter(views, proprio)

    def obs_adapter(values):
        return adapt({key: value * 255.0 if key in views else value for key, value in values.items()})

    return dp, vib, make_action_bridge(dp), obs_adapter


def measure_guidance(dp, planner, obs, eta, scale=None, *, record_kl=True,
                     record_injection=True, record_noisy=False, seed=0, step_mean_c=False):
    """One paired denoise. Measure uncapped KL and the actual capped gradient.

    step_mean_c preserves the legacy C reduction: Torch mean per step, then
    numpy mean of Python floats. Research probes reduce the full numpy array.
    Hooks are restored even if sampling fails. Planner lifetime is caller owned.
    """
    import numpy as np
    import torch

    rows, step_means, injections, noisy = [], [], [], []
    original_kl, original_step = planner._kl_backward, planner.guided_step

    def record_cost(trajectory, x0_hat, current_obs=None):
        kl, grad = original_kl(trajectory, x0_hat, current_obs)
        if step_mean_c:
            step_means.append(float(kl.detach().mean()))
        else:
            rows.append(kl.detach().cpu().numpy().copy())
        return kl, grad

    def record_step(trajectory, x0_hat, current_obs=None, noise_scale=1.0):
        result = original_step(trajectory, x0_hat, current_obs, noise_scale)
        injections.append(eta * float(noise_scale) * float(result[0].detach().abs().mean()))
        if record_noisy:
            noisy.append(float(trajectory.detach().abs().mean()))
        return result

    if record_kl:
        planner._kl_backward = record_cost
    if record_injection:
        planner.guided_step = record_step
    dp.guidance_scale = eta
    try:
        torch.manual_seed(seed)
        dp.predict_action_dyn_guided(obs)
    finally:
        planner._kl_backward, planner.guided_step = original_kl, original_step
    result = {}
    if record_kl:
        if step_mean_c:
            result.update(C_mean=float(np.mean(step_means)) if step_means else 0.0,
                          n_steps=len(step_means))
        else:
            kl = np.stack(rows)
            if not np.isfinite(kl).all():
                raise ValueError("Nonfinite core KL")
            result.update(kl=kl, C_mean=float(kl.mean()), n_steps=len(rows))
    if record_injection:
        inj = np.asarray(injections, dtype=float)
        if not len(inj) or not np.isfinite(inj).all():
            raise ValueError("Missing or nonfinite guidance injection")
        result.update(injection=inj, R_mean=float(inj.mean() / scale))
    if record_noisy:
        result["noisy_scale"] = np.asarray(noisy)
    return result
