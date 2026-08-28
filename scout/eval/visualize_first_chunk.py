"""First-action-chunk trajectory visualization (user 2026-08-27).

Existing tools (``visualize_trajectories`` / ``vis_grid``) overlay FULL
horizon rollouts. This tool answers a different question: for ONE fixed init
scene (``env.reset(seed=<seed>)``), sample N independent policy draws and
execute ONLY the FIRST denoised action chunk (``n_action_steps`` steps) of
each, then overlay the N resulting EE trajectories on the shared top-down
render -- unguided (``predict_action``) and guided (entropy-cost atypical,
``predict_action_dyn_guided``) side by side. This shows the stochastic spread
of exactly the chunk the guidance steers.

Reuses the production machinery unchanged: ``factories`` for DP/VIB/env,
``RolloutPipeline._attach_planner`` for the guided planner wiring, the
``_VecRunner`` chunk executor (batched inference, per-rollout z, chunk
semantics identical to a real rollout's first chunk), and the
``visualize_trajectories`` renderer. Purely additive -- no existing file is
modified.

Usage (server, from the scout repo root; MUJOCO_GL=egl):

    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 python -m scout.eval.visualize_first_chunk \
        --config configs/eval_can_entropy.yaml --task can \
        --base-dp-ckpt <DP-SCOUT-exp4>/checkpoints/299.ckpt \
        --vib-ckpt    <dyn-SCOUT-exp4>/<ts>/scout_vib.ckpt \
        --core-hdf5   <...>/rollout/can_core.hdf5 \
        --seed 42 --n 20 --guide atypical --atypical-cap 2.5 \
        --out-dir data/vis_first_chunk/can_s233_round4
"""
from __future__ import annotations

import argparse
import json
import os
from collections import deque
from typing import List

import numpy as np
import torch


def _eef_of(obs: dict) -> np.ndarray:
    """First frame of the stacked (n_obs_steps, 3) EE-pos obs."""
    return np.asarray(obs["robot0_eef_pos"], dtype=np.float64)[0]


def run_arm(dp, env_factory, state0, n: int, n_envs: int, n_action_steps: int,
            device, guided: bool) -> List[np.ndarray]:
    """N independent first-chunk rollouts from the SAME init state.

    horizon = n_action_steps -> the runner finalizes right after the first
    chunk executes (no second denoise call), and each rollout's diffusion
    noise is drawn independently (per-row randn in the batched sample).
    """
    from scout.eval.rollout_vec import _VecRunner

    trajs, chunks = [], []
    runner = _VecRunner(
        dp, env_factory, n_envs=n_envs, n_action_steps=n_action_steps,
        horizon=n_action_steps, device=device, guided=guided,
        on_done=lambda slot: (trajs.append(slot.traj),
                              chunks.append(np.array(slot.chunk))),
    )
    jobs = deque((state0, i, 0) for i in range(n))
    runner.run(jobs, record_obs=True)
    runner.close()
    assert len(trajs) == n, f"expected {n} trajs, got {len(trajs)}"
    return trajs, chunks


def eef_paths(trajs) -> List[np.ndarray]:
    """Per-traj EE path covering the executed chunk + the post-step endpoint.

    traj['obs'][t] is the PRE-step obs of step t; append the final
    next_obs' last stacked frame so the path includes where the last action
    of the chunk actually brought the EE.
    """
    out = []
    for t in trajs:
        pts = [_eef_of(o) for o in t["obs"]]
        pts.append(np.asarray(t["next_obs"][-1]["robot0_eef_pos"],
                              dtype=np.float64)[-1])
        out.append(np.stack(pts, axis=0))
    return out


def proposal_paths(trajs, chunks) -> List[np.ndarray]:
    """SOE-style (Fig. 4/5) 'action proposal' points: the COMMANDED absolute
    EE waypoints of the denoised chunk itself, no sim execution.

    The chunk rows are 10-d abs actions (pos3|rot6|grip1); the position
    component is the world-frame EE target. Prepend the chunk's conditioning
    EE position (pre-step obs of step 0) so the path starts where the arm is.
    """
    out = []
    for t, c in zip(trajs, chunks):
        start = _eef_of(t["obs"][0])
        out.append(np.concatenate([start[None, :], c[:, :3]], axis=0))
    return out


def chunk_jerk(path: np.ndarray) -> float:
    """Mean 3rd-difference magnitude over the chunk (production jerk metric)."""
    if len(path) < 4:
        return 0.0
    a = np.diff(path, axis=0)
    j = np.diff(a, axis=0, n=2)
    return float(np.mean(np.linalg.norm(j, axis=-1)))


def endpoint_spread(paths) -> float:
    """Mean pairwise distance between chunk endpoints."""
    e = np.stack([p[-1] for p in paths], axis=0)
    d = np.linalg.norm(e[:, None, :] - e[None, :, :], axis=-1)
    n = len(e)
    return float(d.sum() / max(n * (n - 1), 1))


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--base-dp-ckpt", required=True)
    p.add_argument("--vib-ckpt", required=True)
    p.add_argument("--core-hdf5", required=True)
    p.add_argument("--guide", default="atypical",
                   help="'off' | 'atypical' | rand registry idea "
                        "(scout-rand: rand_mjitter/rand_shell/...)")
    p.add_argument("--atypical-cap", type=float, default=2.5)
    p.add_argument("--rand-kwargs", default="",
                   help="k=v,k=v passthrough into entropy_kwargs (e.g. "
                        "'rand_b=0.5' for rand_mjitter; floats auto-parsed)")
    p.add_argument("--seed", type=int, default=42,
                   help="env reset seed -> the ONE init scene")
    p.add_argument("--n", type=int, default=20, help="rollouts per arm")
    p.add_argument("--n-envs", type=int, default=10)
    p.add_argument("--res", type=int, default=768)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--cuda-visible-devices", default=None)
    args = p.parse_args()

    if args.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from scout.eval.factories import (
        load_cfg, make_default_env_factory, make_lpb_dp_factory,
        make_scout_vib_factory,
    )

    cfg = load_cfg(args.config)
    cfg.base_dp.initial_ckpt_path = args.base_dp_ckpt
    cfg.vib.ckpt_path = args.vib_ckpt
    cfg.vib.base_dp_ckpt = args.base_dp_ckpt
    cfg.dataset.path = args.core_hdf5

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dp = make_lpb_dp_factory(device)(args.base_dp_ckpt)
    n_action_steps = int(getattr(dp, "n_action_steps", 1))
    env_factory = make_default_env_factory(cfg)
    print(f"[first_chunk] task={args.task} guide={args.guide} "
          f"n={args.n} seed={args.seed} chunk={n_action_steps} device={device}")

    # ---- the ONE fixed init scene -------------------------------------- #
    env0 = env_factory()
    obs0 = env0.reset(seed=args.seed)
    state0 = env0.get_state()
    start0 = _eef_of(obs0)
    if hasattr(env0, "close"):
        env0.close()
    print(f"[first_chunk] init scene seed={args.seed} "
          f"eef0={np.round(start0, 4)}")

    # ---- guided planner wiring (production path) ----------------------- #
    rand_ek = {}
    for kv in filter(None, args.rand_kwargs.split(",")):
        k, _, v = kv.partition("=")
        try:
            rand_ek[k.strip()] = float(v)
        except ValueError:
            rand_ek[k.strip()] = v.strip()
    if args.guide != "off":
        from scout.eval.rollout_pipeline import RolloutPipeline
        vib = make_scout_vib_factory(cfg, device)(args.vib_ckpt)
        pipeline = RolloutPipeline(
            cfg=cfg, dp_factory=None, scout_vib_factory=None,
            env_factory=env_factory, device=device, guided=True,
            guide_mode=args.guide,
            entropy_kwargs={"atypical_cap": args.atypical_cap, **rand_ek},
        )
        pipeline._attach_planner(dp, vib)

    # ---- run both arms --------------------------------------------------- #
    arms = {}
    for label, guided in (("unguided", False), ("guided", args.guide != "off")):
        reset = getattr(dp, "reset", None)
        if callable(reset):
            reset()
        trajs, chunks = run_arm(dp, env_factory, state0, args.n, args.n_envs,
                                n_action_steps, device, guided)
        paths = eef_paths(trajs)
        props = proposal_paths(trajs, chunks)
        starts = np.stack([q[0] for q in paths], axis=0)
        arms[label] = {
            "paths": paths,
            "props": props,
            "start_dev_max": float(np.max(np.linalg.norm(starts - start0,
                                                         axis=-1))),
            "endpoint_spread": endpoint_spread(paths),
            "jerk": float(np.mean([chunk_jerk(q) for q in paths])),
            "prop_endpoint_spread": endpoint_spread(props),
            "prop_jerk": float(np.mean([chunk_jerk(q) for q in props])),
            "n_steps": int(min(len(q) for q in paths)),
        }
        print(f"[first_chunk] {label}: start_dev_max="
              f"{arms[label]['start_dev_max']:.2e} "
              f"endpoint_spread={arms[label]['endpoint_spread']:.4f} "
              f"prop_endpoint_spread={arms[label]['prop_endpoint_spread']:.4f} "
              f"jerk={arms[label]['jerk']:.4f} "
              f"steps={arms[label]['n_steps']}")

    # ---- render (shared top-down camera over the union of both arms) ---- #
    from scout.eval.visualize_trajectories import (
        add_gradient_colorbar, draw_gradient_path, make_env,
        mark_endpoints, project_to_pixels, render_initial_views,
        trajectory_cmap,
    )

    fit = np.concatenate([q for a in arms.values()
                          for q in a["paths"] + a["props"]], axis=0)
    venv = make_env(args.core_hdf5)
    img_agent, img_top, cam_pos, fov = render_initial_views(
        venv, state0, fit, res=args.res)
    if hasattr(venv, "close"):
        venv.close()
    h_px, w_px = img_top.shape[:2]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cmap = trajectory_cmap()
    fig, axes = plt.subplots(2, 3, figsize=(19.2, 13.5),
                             gridspec_kw={"width_ratios": [1.0, 1.25, 1.25]})
    axes[0][0].imshow(img_agent)
    axes[0][0].set_title(f"initial scene (seed {args.seed}, agentview)")
    axes[0][0].set_xticks([]); axes[0][0].set_yticks([])
    # bottom-left: the bare top-down scene every panel overlays on (reference
    # for where the gripper / can are; the camera is fitted to the union of
    # all trajectory + proposal points, so it crops tightly around them)
    axes[1][0].imshow(img_top)
    axes[1][0].set_title("reference: bare top-down (camera = fit of all "
                         "points)", fontsize=11)
    axes[1][0].set_xticks([]); axes[1][0].set_yticks([])

    for (row, what, key) in ((0, "executed (sim, 8 env steps)", "paths"),
                             (1, "action proposals (commanded waypoints)", "props")):
        for col, label in enumerate(("unguided", "guided"), start=1):
            ax = axes[row][col]
            px = [project_to_pixels(q, cam_pos, fov, w_px, h_px)
                  for q in arms[label][key]]
            ax.imshow(img_top)
            ax.set_xlim(0, w_px); ax.set_ylim(h_px, 0)
            ax.set_xticks([]); ax.set_yticks([])
            for q in px:
                draw_gradient_path(ax, q, cmap, lw=1.7, alpha=0.9)
                mark_endpoints(ax, q[0], q[-1])
            a = arms[label]
            spread = (a["endpoint_spread"] if key == "paths"
                      else a["prop_endpoint_spread"])
            jerk = a["jerk"] if key == "paths" else a["prop_jerk"]
            ax.set_title(f"{label}, {what}\nendpt spread {spread:.3f} m, "
                         f"jerk {jerk:.3f}", fontsize=11)
    add_gradient_colorbar(fig, axes[-1][-1], cmap)
    fig.suptitle(f"{args.task} first-action-chunk ({n_action_steps} steps) -- "
                 f"seed {args.seed}, {args.n} draws/arm, guide={args.guide} "
                 f"{(' '.join(f'{k}={v}' for k, v in rand_ek.items())) if rand_ek else f'cap={args.atypical_cap}'}\n"
                 f"top row = executed in sim; "
                 f"bottom row = SOE-style action proposals (commanded EE "
                 f"targets)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_png = os.path.join(out_dir, "first_chunk_viz.png")
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    plt.imsave(os.path.join(out_dir, "scene_agentview.png"), img_agent)
    plt.imsave(os.path.join(out_dir, "scene_topdown.png"), img_top)

    np.savez_compressed(
        os.path.join(out_dir, "paths.npz"),
        **{f"{lbl}_{kind}_{i}": q for lbl, a in arms.items()
           for kind in ("exec", "prop")
           for i, q in enumerate(a["paths"] if kind == "exec" else a["props"])})
    summary = {
        "task": args.task, "guide": args.guide, "seed": args.seed,
        "n": args.n, "chunk_steps": n_action_steps,
        "dp_ckpt": args.base_dp_ckpt, "vib_ckpt": args.vib_ckpt,
        "eef0": start0.tolist(),
        **{f"{lbl}/{k}": v for lbl, a in arms.items()
           for k, v in a.items() if k not in ("paths", "props")},
    }
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[first_chunk] saved: {out_png}")
    print(f"[first_chunk] done")


if __name__ == "__main__":
    main()
