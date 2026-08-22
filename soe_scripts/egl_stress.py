"""Render-corruption stress harness (user 2026-08-22).

N robomimic envs stepping RANDOM actions -- NO policy, NO VIB, NO torch
compute on the GPU. Isolates the MuJoCo/EGL offscreen-render layer exactly as
the rollout drives it (same env factory, same obs keys). Per env: seeded
reset, T steps, collect agentview frames, classify by temporal-std
(frozen tstd<1 / noise tstd>20 / healthy) -- the same metric as
soe_scripts/vis_validate.py.

Usage:
    env CUDA_VISIBLE_DEVICES=<id> MUJOCO_GL=egl TMPDIR=/tmp \
        python soe_scripts/egl_stress.py <n_envs> <steps> [base_seed]

Run from the repo root. Prints a per-env table + verdict counts; any
OpenGL/EGL traceback goes to stderr (capture with 2>&1).
"""
import os
import sys
import time

import numpy as np


def main():
    n_envs, steps = int(sys.argv[1]), int(sys.argv[2])
    base_seed = int(sys.argv[3]) if len(sys.argv) > 3 else 42
    episodes = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    use_torch = len(sys.argv) > 5 and sys.argv[5] == "--torch"
    sys.path.insert(0, os.getcwd())

    import yaml
    from easydict import EasyDict
    from omegaconf import OmegaConf
    from scout.eval.rollout import make_robomimic_env_factory

    # production-like in-process CUDA compute: every real rollout ran with
    # torch + the policy on the SAME GPU as the EGL rendering.
    big = None
    if use_torch:
        import torch
        torch.cuda.init()
        big = torch.randn(2048, 2048, device="cuda")

    def cuda_churn():
        if big is not None:
            import torch
            y = big @ big                       # ~ GFLOP matmul per tick
            _ = (y * 1e-6).sum().item()         # force sync
            del y

    OmegaConf.register_new_resolver("eval", eval, replace=True)
    core = "data/2026_8_21/CAN-exp1-2333/can/rollout/can_core.hdf5"
    with open("configs/base_dp_can_image.yaml") as f:
        lpb = OmegaConf.create(yaml.safe_load(f))
    fac = make_robomimic_env_factory(
        dataset_path=core,
        shape_meta=OmegaConf.to_container(lpb.shape_meta, resolve=True),
        n_obs_steps=int(OmegaConf.select(lpb, "n_obs_steps", default=2)),
        abs_action=True,
        render_obs_key="agentview_image",
    )
    rng = np.random.default_rng(0)

    def make_envs(n):
        return [fac() for _ in range(n)]

    def run_phase(envs, steps_, seed0, collect_from=None):
        frames_ = [[] for _ in range(len(envs))]
        obs = [envs[i].reset(seed=seed0 + i) for i in range(len(envs))]
        for t in range(steps_):
            cuda_churn()
            for i, env in enumerate(envs):
                a = rng.uniform(-0.3, 0.3, size=10)
                obs[i], _, _, _ = env.step(a)
                if collect_from is None or t >= collect_from:
                    frames_[i].append(np.asarray(obs[i]["agentview_image"]))
        return frames_

    frames = [[] for _ in range(n_envs)]
    t0 = time.time()
    two_phase = episodes < 0            # episodes<0 -> two-phase mode
    lazy_gc = episodes == -2            # -2: drop gen1 WITHOUT gc.collect()
    if two_phase:
        # production shape: eval-phase envs are DROPPED (runner goes out of
        # scope -> lazy GC destroys their EGL contexts), then explore-phase
        # envs are created and must render correctly afterwards.
        import gc
        gen1 = make_envs(n_envs)
        run_phase(gen1, steps, base_seed)
        del gen1
        if not lazy_gc:
            gc.collect()
        gen2 = make_envs(n_envs)
        frames = run_phase(gen2, steps, base_seed + 10_000)
        del gen2
        if lazy_gc:
            gc.collect()
    else:
        envs = make_envs(n_envs)
        for ep in range(max(episodes, 1)):
            frames = run_phase(envs, steps, base_seed + ep * n_envs)
    dt = time.time() - t0

    verdicts = []
    for i, fr in enumerate(frames):
        arr = np.stack(fr).astype(np.float32)          # (T,...)
        scale = 255.0 if arr.max() <= 1.5 else 1.0     # adapter may return [0,1]
        tstd = float(arr.std(axis=0).mean()) * scale
        consec = float(np.abs(np.diff(arr, axis=0)).max()) * scale
        v = "FROZEN" if tstd < 1.0 else ("NOISE" if tstd > 20.0 else "ok")
        verdicts.append(v)
        print(f"env{i:02d} tstd={tstd:6.2f} consec_maxdiff={consec:6.2f} "
              f"mean={arr.mean() * scale:6.1f} {v}", flush=True)
    n_bad = sum(1 for v in verdicts if v != "ok")
    print(f"SUMMARY gpu_env={os.environ.get('CUDA_VISIBLE_DEVICES')} "
          f"n_envs={n_envs} steps={steps} eps={episodes} "
          f"torch={int(use_torch)} time={dt:.0f}s "
          f"BAD={n_bad}/{n_envs}", flush=True)


if __name__ == "__main__":
    main()
